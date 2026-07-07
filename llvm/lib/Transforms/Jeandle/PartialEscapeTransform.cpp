//===- PartialEscapeTransform.cpp - PEA (transform pass) ------------------===//
//
// Copyright (c) 2026, the Jeandle-LLVM Authors. All Rights Reserved.
//
// Part of the Jeandle-LLVM project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Consume the PEAResult from PartialEscapeAnalysis and apply its effects in
// two ordered passes driven by Effect::isCfgKill() (Jeandle's analog of
// Graal's EffectList.apply(graph, obsoleteNodes, cfgKills)).
//
//   Pass 1 (non-cfgKill): ReplaceLoad, ReplaceCall, EliminateStore,
//   Materialize, CreatePHI — applied per-block in RPO via EffectList::apply,
//   which sorts by SeqNo and dispatches each effect's apply() through
//   TransformContext.
//
//   Pass 2 (cfgKill): EliminateAllocation — rewrites an invoke alloc into an
//   unconditional branch to the normal dest (dropping the unwind edge), or
//   plain-erases a call alloc. isCfgKill() is true ONLY for this effect
//   (matching Graal's deleteNode(WithExceptionNode)/killIfBranch).
//
// Between the passes a point-sensitive resolution sub-pass rewrites each
// surviving original-allocation use to the materialize NewInv / merge PHI
// that dominates it (Jeandle's analog of Graal's per-point `aliases` map).
// After both passes: ConstantFoldTerminator, a trivially-dead sweep, and
// EliminateUnreachableBlocks.
//
// At each escape point MaterializeEffect::apply emits a new Hotspot_JIT
// InvokeInst, replays tracked field stores, and re-emits surviving
// monitorenters; EliminateAllocation in Pass 2 then erases the now-dead
// original. The invoke reuses the original alloc's unwind dest when the
// original was itself an invoke, else synthesizes a minimal landingpad+resume
// block, and tags the return with java-klass, java-klass-exact, and nonnull.
//
// Lock cascade: when an object escapes with locks held, the lock model DELETES
// the folded monitorenters and captures the surviving unbalanced enters into
// MaterializeEffect::Locks (sorted ascending by bytecode depth), re-emitting
// them right after the new allocation (Graal: synthetic MonitorEnterNodes at
// the CommitAllocationNode). Matching downstream exits survive in IR with
// operands RAUW'd onto the new invoke.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Jeandle/PartialEscapeTransform.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/Jeandle/PartialEscape.h"
#include "llvm/Analysis/Jeandle/PartialEscapeAnalysis.h"
#include "llvm/Analysis/Jeandle/PartialEscapeUtils.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Jeandle/Attributes.h"
#include "llvm/IR/Jeandle/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

static bool eraseAllocation(Instruction *Target) {
  assert(Target && "EliminateAllocation target must be non-null");

  if (auto *II = dyn_cast<InvokeInst>(Target)) {
    // The transform is the last line of defense: dropping the unwind edge is
    // only sound because the analyzer proved this allocation never observably
    // escapes, and a Jeandle allocation intrinsic's exception edge handles
    // OOM only (unobservable — re-thrown identically by the materialized
    // invoke or, for NeverEscapes, never taken). Assert the analyzer's
    // NeverEscapes contract so a future misclassification of a side-effecting
    // invoke as eliminable fails loudly instead of silently miscompiling.
    assert(jeandle::pea::isJeandleAllocation(II) &&
           "EliminateAllocation may only drop the unwind edge of a Jeandle "
           "allocation intrinsic (OOM-only, unobservable throw)");
    BasicBlock *Normal = II->getNormalDest();
    BasicBlock *Unwind = II->getUnwindDest();
    BasicBlock *Parent = II->getParent();

    // Null out any remaining uses before erasing.
    if (!II->use_empty())
      II->replaceAllUsesWith(PoisonValue::get(II->getType()));

    // Remove the unwind edge: update any PHIs in the unwind block so they
    // forget the predecessor we're about to drop.
    Unwind->removePredecessor(Parent, /*KeepOneInputPHIs=*/true);

    // Replace the invoke with an unconditional branch to the normal dest.
    BranchInst::Create(Normal, Parent);
    II->eraseFromParent();
    return true;
  }

  if (auto *CI = dyn_cast<CallInst>(Target)) {
    // Same NeverEscapes contract as the invoke branch above (a call-form
    // allocation has no unwind edge to drop, but the eliminability guarantee
    // is identical).
    assert(jeandle::pea::isJeandleAllocation(CI) &&
           "EliminateAllocation target must be a Jeandle allocation intrinsic");
    if (!CI->use_empty())
      CI->replaceAllUsesWith(PoisonValue::get(CI->getType()));
    CI->eraseFromParent();
    return true;
  }

  return false;
}

// Synthesize a minimal unwind-destination block at the end of F for use as the
// unwind destination of a materialization invoke. The frontend always emits the
// function's personality so this is well-formed. When the materialization site
// sits inside a Windows-EH funclet (EnclosingPad != null), the unwind dest must
// itself be a funclet pad nested in the enclosing pad — a plain landingpad is
// illegal inside a funclet — so we emit `cleanuppad within %EnclosingPad` +
// `cleanupret ... unwind to caller`. Otherwise emit the landingpad+resume form.
static BasicBlock *createMinimalUnwindBlock(Function &F,
                                            FuncletPadInst *EnclosingPad) {
  LLVMContext &Ctx = F.getContext();
  BasicBlock *BB = BasicBlock::Create(Ctx, "pea.unwind", &F);
  IRBuilder<> B(BB);
  if (EnclosingPad) {
    CleanupPadInst *CP = B.CreateCleanupPad(EnclosingPad);
    B.CreateCleanupRet(CP, /*UnwindBB=*/nullptr); // unwind to caller
  } else {
    LandingPadInst *LP = B.CreateLandingPad(Type::getInt64Ty(Ctx), 0, "pea.lp");
    LP->setCleanup(true);
    B.CreateResume(LP);
  }
  return BB;
}

// Pick (or synthesize) the unwind destination for a materialization invoke.
// Strategy 1: reuse the original allocation's unwind dest if the original was
// itself an InvokeInst (it's guaranteed landingpad-compatible because the
// frontend created it for OOM handling) AND the materialization site is not
// inside a funclet — reuse would be illegal when the new invoke is funclet-
// nested but the reused dest belongs to a different (or no) funclet. The dest
// must also have no PHIs: IRBuilder::CreateInvoke does not add a PHI incoming
// for the new predecessor, so reusing a PHI-carrying dest would leave the new
// (materialization-OOM) predecessor without a matching incoming. Strategy 2
// (fallback): synthesize a minimal unwind block, funclet-aware when needed.
static BasicBlock *findOrSynthesizeUnwindDest(Function &F, CallBase *OrigAlloc,
                                              FuncletPadInst *EnclosingPad) {
  if (!EnclosingPad)
    if (auto *OrigInv = dyn_cast<InvokeInst>(OrigAlloc))
      if (BasicBlock *UD = OrigInv->getUnwindDest())
        if (UD->phis().empty())
          return UD;
  return createMinimalUnwindBlock(F, EnclosingPad);
}

// Emit the materialization sequence for a single Materialize effect: split the
// containing block at the MaterializeEffect's InsertBefore so the new
// materialization is the terminator, emit a hotspotcc InvokeInst, replay
// tracked field stores at the top of the normal-dest block, and record the
// materialization (OrigAlloc → NewInv) in Defs for the point-sensitive
// resolution sub-pass (OrigAlloc is not RAUW'd inline). The same OrigAlloc may
// be materialized multiple times (mixed-state merge synthesizing a per-pred
// materialization on each virtual incoming): record each (analyzer-recorded-
// pred-block, OrigAlloc) → NewInv in MatPerBlock (CreatePHI picks the right
// per-incoming NewInv) and Origin → MatCont in BlockRename (so the PHI's
// incoming-block names the post-split merge-pred). Lock cascade: the surviving
// unbalanced enters were captured by the analyzer into the MaterializeEffect's
// Locks (sorted ascending by bytecode depth) and are re-emitted right after
// the field stores here.
//
// The materialized invoke is structurally identical to a frontend allocation
// site (hotspotcc `jeandle.new_instance` / `jeandle.new_array`, addrspace(1)
// return, exception edge), so the downstream GC-statepoint pipeline
// (PEA → InsertGCBarriers → ... → RewriteStatepointsForGC) wraps it
// uniformly with gc.statepoint/gc.result/gc.relocate; splitBasicBlock is
// SSA-preserving and the materialized pointer dominates every use in MatCont.
// See `partial-escape/310_full_pipeline_statepoint.ll`.

// Eager-update hook: call this BEFORE erasing `Dying` from IR. Re-aims every
// MaterializeEffect whose InsertBefore == Dying to `Next` (the normal-flow
// successor in the SAME block — for a non-terminator that is
// Target->getNextNode(); for an invoke terminator it is the `br` created by
// BranchInst::Create, captured as II->getNextNode() after Create but before
// erase). This keeps the WeakTrackingVH alive so applyMaterialize never sees
// a null InsertBefore. Mirrors Graal's "fixed deleted -> use node.next()"
// (PartialEscapeClosure.java:310-329, MATERIALIZE_ALL); Graal mutates IR
// during analysis so its `fixed` is always the live post-fold node, whereas
// Jeandle's analysis/transform split captures a stale IP that a sibling erase
// can invalidate — this hook is the transform-time equivalent of Graal's
// re-derivation. Re-indexes each dependent into Next's bucket so a future
// erase of Next chains correctly.
static void relocateDependentMaterializes(
    DenseMap<Instruction *, SmallVector<jeandle::MaterializeEffect *, 4>>
        &Dependents,
    Instruction *Dying, Instruction *Next) {
  if (!Next || Next == Dying)
    return;
  auto It = Dependents.find(Dying);
  if (It == Dependents.end())
    return;
  // Move the bucket out and erase Dying's entry BEFORE any insert: inserting
  // into Dependents[Next] below could rehash and invalidate `It`.
  SmallVector<jeandle::MaterializeEffect *, 4> Bucket = std::move(It->second);
  Dependents.erase(It);
  auto &NextBucket = Dependents[Next];
  for (jeandle::MaterializeEffect *M : Bucket) {
    M->setInsertBefore(Next);
    NextBucket.push_back(M);
  }
}

static void applyMaterialize(
    Function &F, const jeandle::PEAResult &Result,
    const jeandle::MaterializeEffect &E,
    DenseMap<Value *, Value *> &NewAllocFor,
    DenseMap<std::pair<BasicBlock *, Value *>, Value *> &MatPerBlock,
    DenseMap<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *> &BlockRename,
    DenseMap<Value *, SmallVector<Value *, 4>> &Defs,
    DenseMap<const jeandle::MaterializeEffect *, InvokeInst *> &NewInvOf,
    const DenseSet<const jeandle::MaterializeEffect *> &IsCascadeTail,
    const DenseMap<const jeandle::MaterializeEffect *, Instruction *>
        &CascadeKeyOf,
    const DenseMap<Instruction *,
                   SmallVector<const jeandle::MaterializeEffect *, 4>>
        &CascadeGroups) {
  assert(E.ObjID != jeandle::InvalidObjectID);
  assert(E.Target && "Materialize effect must carry the original allocation");

  jeandle::VirtualObject &VObj = *Result.VirtualObjects[E.ObjID];
  CallBase *OrigAlloc = VObj.AllocationCall;
  assert(OrigAlloc == E.Target);

  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  const DataLayout &DL = M->getDataLayout();

  // The eager-update hook (relocateDependentMaterializes, called from
  // ReplaceLoad/ReplaceCall/EliminateStore before erasing) re-aims every
  // dependent Materialize to the next instruction before the erase nulls this
  // WeakTrackingVH, so InsertBefore is always live at apply time. Assert it,
  // rather than silently hoisting to the alloc's normal-dest (the old fallback
  // was unsound: premature escape + SSA-dominance-unsound replay of field
  // stores for values defined between the normal-dest head and the escape
  // point). This ports Graal's `fixed`-never-erased guarantee (Trap 9), now
  // enforced by the eager update rather than merely assumed. Keep the
  // WeakTrackingVH so a future erase path that misses the hook surfaces as a
  // clean null-assert instead of use-after-free.
  // TODO(jeandle-pea): if a future erase path is added, add the hook there too.
  Instruction *InsertBefore = dyn_cast_or_null<Instruction>(E.InsertBefore);
  assert(InsertBefore &&
         "Materialize InsertBefore was erased — an erase path "
         "missed the eager-update hook (see relocateDependentMaterializes)");
  BasicBlock *Origin = InsertBefore->getParent();
  // Capture the analyzer's recorded pred BB at this effect (which may
  // already have been renamed by an earlier applyMaterialize at the same
  // pred) for the MatPerBlock key.
  BasicBlock *AnalyzerRecordedPred = E.Block;

  // Step 1: pick the allocation function.
  const char *FnName =
      VObj.isInstance() ? "jeandle.new_instance" : "jeandle.new_array";
  Function *AllocFn = M->getFunction(FnName);
  assert(AllocFn && "alloc function not declared in module");

  // Step 2: build args. Rebuild the klass pointer constant (rather than reuse
  // the original op) so we don't carry a stale operand reference.
  Value *Arg0 = ConstantExpr::getIntToPtr(
      ConstantInt::get(Type::getInt64Ty(Ctx),
                       static_cast<uint64_t>(VObj.Klass)),
      PointerType::get(Ctx, /*AS=*/0));
  Type *I32 = Type::getInt32Ty(Ctx);
  SmallVector<Value *, 5> AllocArgs;
  AllocArgs.push_back(Arg0);
  if (VObj.isInstance()) {
    AllocArgs.push_back(ConstantInt::get(I32, VObj.SizeInBytes));
  } else {
    assert(AllocFn->arg_size() == 5 &&
           "jeandle.new_array must use the 5-arg protocol");
    AllocArgs.push_back(ConstantInt::get(I32, VObj.ArrayLength));
    AllocArgs.push_back(ConstantInt::get(I32, VObj.SizeInBytes));
    AllocArgs.push_back(ConstantInt::get(I32, VObj.ArrayBaseOffset));
    AllocArgs.push_back(ConstantInt::get(I32, VObj.ArrayLengthLimit));
  }

  // Step 3: determine the enclosing EH funclet pad of the materialization site
  // BEFORE the split below — colorEHFunclets requires well-formed IR (every
  // block terminated), but Step 4 transiently leaves Origin without a
  // terminator. The new invoke is emitted at the END of `Origin` (Step 6), so
  // it belongs to ORIGIN's funclet; the split does not change Origin's funclet
  // membership. A funclet pad sits at the ENTRY of a funclet, never at an
  // arbitrary block head, so we resolve the pad from Origin's own funclet
  // color rather than scanning a post-split block head. Gated on a funclet
  // personality so it is a true no-op (no O(F) coloring) on the current
  // non-funclet target — Jeandle is not on Windows, but the standing
  // IR-defensiveness rule requires tolerating any legal IR.
  FuncletPadInst *EnclosingFuncletPad = nullptr;
  if (F.hasPersonalityFn() &&
      isFuncletEHPersonality(classifyEHPersonality(F.getPersonalityFn()))) {
    DenseMap<BasicBlock *, ColorVector> BlockColors = colorEHFunclets(F);
    auto CIt = BlockColors.find(Origin);
    if (CIt != BlockColors.end() && !CIt->second.empty())
      // A block in a funclet has exactly one color (its funclet's entry pad);
      // the entry block colors to itself (no pad). front() is that color.
      EnclosingFuncletPad =
          dyn_cast<FuncletPadInst>(CIt->second.front()->getFirstNonPHI());
  }

  // Step 3b: find or synthesize the unwind destination (funclet-aware when the
  // materialization site is inside a funclet).
  BasicBlock *UnwindDest =
      findOrSynthesizeUnwindDest(F, OrigAlloc, EnclosingFuncletPad);

  // Step 4: split the origin block at InsertBefore. SplitBlock leaves Origin
  // ending with an unconditional br to the new MatCont block; we drop that
  // terminator because the materialization invoke will take its place.
  BasicBlock *MatCont = Origin->splitBasicBlock(InsertBefore, "mat.cont");
  Origin->getTerminator()->eraseFromParent();

  // Step 5: collect operand bundles from the recorded source (escape-point
  // CallBase or original allocation). Drop "deopt": copying it would plant
  // OrigAlloc into NewInv's own bundle (the source CB's deopt slot for the VO
  // holds OrigAlloc), which the resolution sub-pass would rewrite to NewInv —
  // a self-reference the verifier rejects.
  // TODO(jeandle-deopt): see applyMaterialize().
  // Preserve every non-deopt bundle (funclet, gc-transition, cfguardtarget,
  // ptrauth, kcfi, ...). The funclet-bundle synthesis below handles the
  // Windows-EH case for the materialization site itself.
  SmallVector<OperandBundleDef, 4> Bundles;
  if (E.DeoptBundleSource) {
    if (auto *CBSrc = dyn_cast<CallBase>(E.DeoptBundleSource)) {
      SmallVector<OperandBundleDef, 4> All;
      CBSrc->getOperandBundlesAsDefs(All);
      for (OperandBundleDef &OBD : All)
        if (OBD.getTag() != "deopt")
          Bundles.emplace_back(std::move(OBD));
    }
  }
  // Attach the funclet bundle computed pre-split (Step 3b) when the
  // materialization site sits inside an EH funclet and the recorded source
  // didn't already supply one.
  bool HasFunclet = false;
  for (const OperandBundleDef &BD : Bundles)
    if (BD.getTag() == "funclet") {
      HasFunclet = true;
      break;
    }
  if (!HasFunclet && EnclosingFuncletPad)
    Bundles.emplace_back("funclet", static_cast<Value *>(EnclosingFuncletPad));

  // Step 6: emit the InvokeInst at the end of Origin.
  IRBuilder<> B(Origin);
  if (InsertBefore->getDebugLoc())
    B.SetCurrentDebugLocation(InsertBefore->getDebugLoc());
  InvokeInst *NewInv = B.CreateInvoke(AllocFn, /*NormalDest=*/MatCont,
                                      /*UnwindDest=*/UnwindDest, AllocArgs,
                                      Bundles, "pea.mat");
  NewInv->setCallingConv(CallingConv::Hotspot_JIT);
  // Copy metadata and merge attrs from the original allocation so downstream
  // RewriteStatepointsForGC / GC barriers don't see weaker output (lost
  // !prof, !alias.scope, !noalias, !jeandle.bytecodeindex, nofree/nosync/cold).
  // Metadata first; addRetAttr below then takes precedence. The re-emitted
  // invoke is rebuilt from VirtualObject structural fields and matches the
  // allocation function's full arity, so the per-param argument attrs in
  // OrigAttrs align one-to-one and are safe to reuse; return attrs are added
  // explicitly below.
  NewInv->copyMetadata(*OrigAlloc, /*WL=*/{});
  AttributeList OrigAttrs = OrigAlloc->getAttributes();
  AttributeList CurAttrs = NewInv->getAttributes();
  AttrBuilder RetAB(Ctx, CurAttrs.getRetAttrs());
  NewInv->setAttributes(OrigAttrs.addRetAttributes(Ctx, RetAB));
  // Carry forward the precise return klass. Added after the merge so they
  // override the same Kind slot from the original.
  NewInv->addRetAttr(Attribute::get(Ctx, jeandle::Attribute::JavaKlass,
                                    std::to_string(VObj.Klass)));
  NewInv->addRetAttr(Attribute::get(Ctx, jeandle::Attribute::JavaKlassExact));
  NewInv->addRetAttr(Attribute::get(Ctx, Attribute::NonNull));

  // Record this effect's NewInv keyed by the effect itself (NOT by OrigAlloc,
  // which is last-write-wins across per-pred materializations of the same
  // object) so the cascade-tail field-store replay below stores into the
  // correct per-effect allocation.
  NewInvOf[&E] = NewInv;

  // Step 7: replay tracked field stores. Only the cascade TAIL emits, and it
  // emits the WHOLE cascade's stores here (every sibling NewInv already exists
  // and, the chain being straight-line, dominates this tail block). Non-tail
  // members defer their stores to their tail. This is what makes a back-edge
  // field (p.g = o where o is materialized later in the same cascade) resolve:
  // the store lands in the tail — dominated by o's NewInv — instead of in an
  // earlier block where o's NewInv could not dominate it. Graal commits every
  // object at one escape point in a single CommitAllocationNode; emitting all
  // cascade invokes before any field store is the Jeandle analog. The lock
  // re-emit below shares this SB so within the tail, stores precede locks.
  IRBuilder<> SB(MatCont, MatCont->getFirstInsertionPt());
  if (InsertBefore->getDebugLoc())
    SB.SetCurrentDebugLocation(InsertBefore->getDebugLoc());
  Type *I8 = Type::getInt8Ty(Ctx);
  if (IsCascadeTail.count(&E)) {
    Instruction *Key = CascadeKeyOf.lookup(&E);
    auto It = CascadeGroups.find(Key);
    const jeandle::MaterializeEffect *Self = &E;
    ArrayRef<const jeandle::MaterializeEffect *> Members =
        It != CascadeGroups.end()
            ? ArrayRef<const jeandle::MaterializeEffect *>(It->second)
            : ArrayRef<const jeandle::MaterializeEffect *>(&Self, 1);
    for (const jeandle::MaterializeEffect *M : Members) {
      InvokeInst *Base = NewInvOf.lookup(M);
      assert(Base &&
             "every cascade member's NewInv must be recorded before the "
             "tail emits its field stores");
      for (const auto &FE : M->FieldEntries) {
        Value *V = nullptr;
        if (FE.Value.isScalar()) {
          V = FE.Value.getScalar();
        } else if (FE.Value.isMaterializedRef()) {
          // Field value is an inner/peer virtual's OrigAlloc (the analyzer
          // records OrigAlloc on both the live and per-pred paths). Emit it
          // here; the point-sensitive resolution sub-pass
          // (resolveMaterializedUses) rewrites this store's value to the NewInv
          // that dominates it (Jeandle's analog of Graal getAliasAndResolve).
          // Forward, self-referential, AND cyclic back-edge fields all resolve
          // because the store sits in the cascade tail, dominated by every
          // peer NewInv. Eager substitution via NewAllocFor would be
          // last-write-wins and miscompile multi-materialization cases.
          V = FE.Value.getMaterialized();
        } else {
          // The analyzer rewrites every VirtualRef into MaterializedRef during
          // recursive prerequisite materialization. Unknown entries are
          // filtered out at snapshot time. Hitting either tag here is a
          // contract violation.
          assert(false && "VirtualRef field entries must have been rewritten "
                          "to MaterializedRef during analysis");
          continue;
        }
        if (!V)
          continue;
        Value *Slot = SB.CreateInBoundsGEP(I8, Base, SB.getInt64(FE.Offset),
                                           "pea.matslot");
        // Natural alignment = the field type's store size rounded up to a power
        // of two (ptr addrspace(1) -> heap pointer width, i64/double -> 8,
        // i32/float -> 4, i16 -> 2, i8 -> 1). The replayed stores are
        // atomic-unordered, and atomic accesses MUST be naturally aligned (an
        // under-aligned atomic store lowers to a libcall or is rejected by the
        // backend). Note we deliberately do NOT use getABITypeAlign: the ABI
        // alignment of a type may legally be SMALLER than its size (e.g. i64
        // has ABI align 4 under LLVM's default datalayout), which is too weak
        // for an atomic. The store size is derived from the DataLayout, so this
        // stays correct under a future compressed-oop / 32-bit heap model and
        // matches the frontend's natural-aligned emission (and
        // VirtualObject::FieldDesc::ByteSize).
        uint64_t StoreSz = DL.getTypeStoreSize(V->getType()).getFixedValue();
        Align NaturalAlign(llvm::PowerOf2Ceil(StoreSz ? StoreSz : 1));
        StoreInst *S = SB.CreateAlignedStore(V, Slot, NaturalAlign);
        // Java heap stores are atomic-unordered (matches jeandle-jdk emission).
        S->setAtomic(AtomicOrdering::Unordered);
      }
    }
  }

  // Re-emit surviving monitorenters at the materialize point. Graal flattens
  // every lock materialized at one point into a single CommitAllocationNode and
  // lowers them globally sorted ascending by lock depth; the analyzer captured
  // each VO's locks per-effect, and PEAResult::computeEscapePointLocks merged
  // them per escape point (the shared InsertBefore of this cascade group) into
  // Result.EscapePointLocks. This effect emits the merged list ONCE — iff it is
  // the highest-SeqNo effect at its escape point (by then every sibling's
  // NewInv is in NewInvOf, so each lock's receiver resolves via
  // NewInvOf[ML.SourceEffect]; chained block-splits also leave this effect's
  // MatCont last at the escape call, so emitting here after the field stores
  // precedes the escape and follows every sibling's field stores). Emitting
  // per-effect here would mis-order re-entrant interleaved cascades. This is
  // unified across live-path AND per-pred cascades: after the critical-edge
  // pre-pass re-aims per-pred effects onto a shared split-edge block (or, on a
  // single-succ pred, they already share the terminator), a per-pred cascade
  // is structurally identical to a live-path cascade — straight-line NewInv
  // chain, tail emits — so the tail's MatCont IS dominated by every sibling
  // NewInv. If the escape-point key is unresolved (the escape call was erased
  // by a sibling effect), fall back to this effect's own locks.
  auto EmitLock = [&](Value *Recv, Function *Callee,
                      ArrayRef<Value *> NonReceiverArgs) {
    if (!Callee)
      return;
    SmallVector<Value *, 4> Args;
    Args.push_back(Recv);
    for (Value *A : NonReceiverArgs)
      Args.push_back(A);
    CallInst *Enter = SB.CreateCall(Callee, Args);
    Enter->setCallingConv(CallingConv::Hotspot_JIT);
  };
  // Use the ORIGINAL escape-point instruction (captured once at pre-scan in
  // CascadeKeyOf) as the lock-re-emit key, NOT the (possibly re-aimed)
  // E.InsertBefore. MaxSeqForEscapePoint/EscapePointLocks are keyed by the
  // original InsertBefore (populated by computeEscapePointLocks before
  // Pass 1); after the eager-update hook re-aimed E.InsertBefore to the
  // next instruction, re-reading it would miss the lookup and fall back to
  // per-effect (mis-ordering re-entrant locks for Case-A cascades). The
  // captured key may dangle after an erase, but the frozen-map lookup is
  // pointer-value-only (no deref) — safe, identical to the cascade replay
  // (CascadeKeyOf[&E] -> CascadeGroups) below. This keying is used for BOTH
  // live-path and per-pred effects (per-pred effects' split-block first
  // non-PHI is never a sibling erase target — the split block hosts only
  // per-pred Materialize effects — so CascadeKeyOf stays valid for them).
  Instruction *EscapeKey = CascadeKeyOf.lookup(&E);
  auto MaxIt = EscapeKey ? Result.MaxSeqForEscapePoint.find(EscapeKey)
                         : Result.MaxSeqForEscapePoint.end();
  if (MaxIt != Result.MaxSeqForEscapePoint.end() && E.SeqNo == MaxIt->second) {
    // Cascade emitter (live-path OR per-pred): emit the escape point's
    // globally-merged lock list once. Every sibling's NewInv is in NewInvOf
    // (lower SeqNo, already applied) and this effect's own was just registered,
    // so each lock's receiver resolves per-effect.
    auto It = Result.EscapePointLocks.find(EscapeKey);
    if (It != Result.EscapePointLocks.end())
      for (const jeandle::MergedLock &ML : It->second) {
        // Every cascade member's applyMaterialize reaches NewInvOf[&E] = NewInv
        // (PartialEscapeTransform.cpp:378) before the tail emits — no early
        // return between entry and :378, only asserts — so the per-effect
        // receiver is always populated. Assert it rather than fall back to
        // NewAllocFor[OrigAlloc], which is last-write-wins across per-pred
        // materializations of the same object and could pick a non-dominating
        // NewInv. Mirrors the cascade field-store replay assert at :380-382.
        auto NIt = NewInvOf.find(ML.SourceEffect);
        assert(NIt != NewInvOf.end() && "every cascade member's NewInv must be "
                                        "recorded before the tail emits locks");
        EmitLock(NIt->second, ML.Callee, ML.NonReceiverArgs);
      }
  } else if (MaxIt == Result.MaxSeqForEscapePoint.end()) {
    // Not a cascade group (single-effect escape point, or the escape call
    // was erased and no merged list was recorded): emit this effect's own
    // locks per-effect. A sibling in an actual cascade (MaxIt found,
    // SeqNo != max) emits nothing here — the tail handles the whole group.
    for (const jeandle::MaterializedLock &ML : E.Locks)
      EmitLock(NewInv, ML.Callee, ML.NonReceiverArgs);
  }

  // Record this materialization in NewAllocFor so any later applyMaterialize
  // can rewrite a recorded MaterializedRef referencing OrigAlloc to the live
  // NewInv (field-store replay of a nested virtual).
  NewAllocFor[OrigAlloc] = NewInv;
  // Per-pred mapping for CreatePHI to pick the right NewInv per incoming.
  // Also record the block split so the PHI's incoming-block resolves to the
  // post-split MatCont.
  MatPerBlock[{AnalyzerRecordedPred, OrigAlloc}] = NewInv;
  // Per-pred-distinct placeholder resolution (Graal: a distinct
  // AllocatedObjectNode per materialize). When the analyzer carried a per-pred
  // placeholder Value* (one per (pred, ObjectID)), record the mapping under
  // that placeholder so CreatePHI incomings and any inherited materialized
  // value resolve to THIS pred's own NewInv — never the global last-write-wins
  // NewAllocFor[OrigAlloc]. The placeholder is never inserted into IR; it is
  // only ever looked up here.
  if (E.PerPredPlaceholder) {
    MatPerBlock[{AnalyzerRecordedPred, E.PerPredPlaceholder}] = NewInv;
    NewAllocFor[E.PerPredPlaceholder] = NewInv;
  }
  BlockRename[{Origin, E.TargetMergeBB}] = MatCont;

  // Record this NewInv as a definition point of OrigAlloc. OrigAlloc is not
  // RAUW'd inline — the point-sensitive resolution sub-pass (run after
  // Pass 1, once the CFG is stable and a fresh DominatorTree is available)
  // rewrites each surviving OrigAlloc use to the unique dominating def (this
  // NewInv, a sibling per-pred NewInv, or a merge PHI). This mirrors Graal's
  // per-point alias resolution (getAlias/getAliasAndResolve) and is what makes
  // escape-point (non-dominating) materialization SSA-sound. Deopt-bundle
  // operands are scrubbed to a typed null in the sub-pass (PEA stays
  // deopt-agnostic; see resolveMaterializedUses).
  Defs[OrigAlloc].push_back(NewInv);
}

// Bundles the Function, the analysis result, and the shared per-apply maps so
// each Effect subclass's apply() is self-contained (Jeandle's adaptation of
// Graal's `apply(StructuredGraph graph, ArrayList<Node> obsoleteNodes)` — LLVM
// mutates a Function, not a StructuredGraph, and carries the alias/def maps the
// analysis could not populate because it cannot mutate IR).
struct jeandle::TransformContext {
  Function &F;
  jeandle::PEAResult &Result;
  DenseMap<Value *, Value *> &NewAllocFor;
  DenseMap<std::pair<BasicBlock *, Value *>, Value *> &MatPerBlock;
  DenseMap<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *> &BlockRename;
  DenseMap<Value *, SmallVector<Value *, 4>> &Defs;
  bool &Changed;

  // Cascade coordination for cyclic-field materialization (Graal's single
  // CommitAllocationNode model). Within a cascade (>= 2 Materialize effects
  // sharing one escape-point InsertBefore, plus each singleton as a group of
  // one), every NewInv invoke is emitted before any of the group's field
  // stores: only the cascade TAIL replays stores, and it replays the WHOLE
  // group's into its MatCont (the cascade's final block, dominated by every
  // NewInv). That makes a back-edge field (referencing a peer materialized
  // later in the cascade) resolve through the point-sensitive resolution sub-
  // pass instead of lowering to poison. Populated by run() before the Pass-1
  // RPO loop; NewInvOf is filled incrementally as each Materialize applies.
  //   CascadeGroups  escape-point InsertBefore -> member effects (SeqNo-sorted)
  //   CascadeKeyOf   effect -> its InsertBefore (captured once at pre-scan; the
  //                  WeakTrackingVH is never re-read for cascade membership)
  //   IsCascadeTail  the max-SeqNo member of each group (a singleton IS its own
  //                  tail, so its behavior is identical to per-effect emission)
  //   NewInvOf       effect -> the InvokeInst it emitted (Phase A records it)
  DenseMap<Instruction *, SmallVector<const jeandle::MaterializeEffect *, 4>>
      &CascadeGroups;
  DenseMap<const jeandle::MaterializeEffect *, Instruction *> &CascadeKeyOf;
  DenseSet<const jeandle::MaterializeEffect *> &IsCascadeTail;
  DenseMap<const jeandle::MaterializeEffect *, InvokeInst *> &NewInvOf;
  // Reverse index of CascadeKeyOf: escape-point InsertBefore -> the non-const
  // MaterializeEffect list keyed on it. Built in the pre-scan alongside
  // CascadeKeyOf. Consumed by relocateDependentMaterializes: when an erase
  // effect (ReplaceLoad/ReplaceCall/EliminateStore) deletes an instruction
  // that is one or more Materializes' InsertBefore, it re-aims each to the
  // next instruction before the erase nulls the WeakTrackingVH. This is the
  // Jeandle analog of Graal's "fixed deleted -> use node.next()" (MATERIALIZE_
  // ALL, PartialEscapeClosure.java:310-329) and keeps InsertBefore live at
  // apply time so applyMaterialize can assert it. Non-const MaterializeEffect*
  // because the hook calls setInsertBefore. Re-indexed to the next
  // instruction's bucket for chained erases.
  DenseMap<Instruction *, SmallVector<jeandle::MaterializeEffect *, 4>>
      &InsertBeforeDependents;
};

void jeandle::ReplaceLoadEffect::apply(jeandle::TransformContext &Ctx) {
  if (!Target || !Replacement)
    return;
  Value *Repl = Replacement;
  // When the replacement's stamp is wider than the original, a Pi node would
  // normally be injected. LLVM has no per-Value stamp at this layer — the
  // closest analogue is the load-only metadata the original load may have
  // carried. Transfer those (only when both sides are LoadInsts and the
  // Replacement is missing the kind) so downstream LLVM passes do not lose the
  // narrower-than-default knowledge after RAUW.
  if (auto *TargetLoad = dyn_cast<LoadInst>(Target)) {
    if (auto *ReplLoad = dyn_cast<LoadInst>(Repl)) {
      static constexpr unsigned PreservableKinds[] = {
          LLVMContext::MD_nonnull,
          LLVMContext::MD_dereferenceable,
          LLVMContext::MD_dereferenceable_or_null,
          LLVMContext::MD_align,
          LLVMContext::MD_invariant_load,
          LLVMContext::MD_noundef,
      };
      for (unsigned K : PreservableKinds) {
        if (ReplLoad->getMetadata(K))
          continue; // already at least as precise; do not overwrite.
        if (MDNode *MD = TargetLoad->getMetadata(K))
          ReplLoad->setMetadata(K, MD);
      }
    }
  }
  // The analyzer may have synthesized an unparented coercion instruction as the
  // replacement (a same-bit-width `bitcast` reinterpretation). Splice it, and
  // any still-unparented operand, in postorder so each operand is parented
  // before its user; all land immediately before Target. A PHINode replacement
  // is owned by a CreatePHI effect that runs LATER in SeqNo order, so it is
  // treated as a leaf here (splicing it mid-block is illegal).
  if (auto *RI = dyn_cast<Instruction>(Repl); RI && !isa<PHINode>(RI)) {
    SmallVector<Instruction *, 4> Stack;
    SmallPtrSet<Instruction *, 4> Visited;
    if (RI->getParent() == nullptr && Visited.insert(RI).second)
      Stack.push_back(RI);
    SmallVector<Instruction *, 4> PostOrder;
    struct Frame {
      Instruction *I;
      unsigned NextOpIdx;
    };
    SmallVector<Frame, 4> Frames;
    if (!Stack.empty()) {
      Frames.push_back({Stack.back(), 0});
      while (!Frames.empty()) {
        Frame &Top = Frames.back();
        if (Top.NextOpIdx < Top.I->getNumOperands()) {
          Value *Op = Top.I->getOperand(Top.NextOpIdx++);
          if (auto *OpI = dyn_cast<Instruction>(Op)) {
            if (OpI->getParent() == nullptr && !isa<PHINode>(OpI) &&
                Visited.insert(OpI).second) {
              Frames.push_back({OpI, 0});
            }
          }
        } else {
          PostOrder.push_back(Top.I);
          Frames.pop_back();
        }
      }
    }
    for (Instruction *I : PostOrder) {
      if (I->getParent() == nullptr)
        I->insertBefore(Target->getIterator());
    }
  }
  if (!Target->use_empty())
    Target->replaceAllUsesWith(Repl);
  // Re-aim any Materialize keyed on `Target` to its next instruction before
  // the erase nulls the WeakTrackingVH. (Loads are never block terminators,
  // so getNextNode() is the in-block normal-flow successor.)
  relocateDependentMaterializes(Ctx.InsertBeforeDependents, Target,
                                Target->getNextNode());
  Target->eraseFromParent();
  Ctx.Changed = true;
}

void jeandle::ReplaceCallEffect::apply(jeandle::TransformContext &Ctx) {
  // JavaOp folded against a virtual receiver: non-void results are replaced
  // with a constant and the call erased; void JavaOps use a null Replacement to
  // request deletion only. Folded results that feed `br i1` leave constant
  // terminators cleaned up by ConstantFoldTerminator in run().
  if (!Target)
    return;
  if (Replacement) {
    Target->replaceAllUsesWith(Replacement);
  } else if (!Target->use_empty()) {
    return;
  }
  // For InvokeInst we cannot simply erase — the unwind edge must be dropped
  // first. JavaOp folds emit `call` (not invoke) calls in practice; defensively
  // handle invokes.
  if (auto *II = dyn_cast<InvokeInst>(Target)) {
    BasicBlock *Normal = II->getNormalDest();
    BasicBlock *Unwind = II->getUnwindDest();
    BasicBlock *Parent = II->getParent();
    Unwind->removePredecessor(Parent, /*KeepOneInputPHIs=*/true);
    BranchInst::Create(Normal, Parent);
    // Re-aim any Materialize keyed on `II` to the freshly-created `br` (II's
    // normal successor in the SAME block) before erasing II. This must use the
    // `br` (II->getNextNode() after the Create), NOT
    // Normal->getFirstNonPHIOrDbg — the latter lives in the (multi-pred)
    // normal-dest block and would split the merge, mis-placing the materialize
    // on every predecessor's path.
    relocateDependentMaterializes(Ctx.InsertBeforeDependents, II,
                                  II->getNextNode());
    II->eraseFromParent();
  } else {
    // Re-aim any Materialize keyed on `Target` to its next instruction before
    // the erase nulls the WeakTrackingVH.
    relocateDependentMaterializes(Ctx.InsertBeforeDependents, Target,
                                  Target->getNextNode());
    Target->eraseFromParent();
  }
  Ctx.Changed = true;
}

void jeandle::EliminateStoreEffect::apply(jeandle::TransformContext &Ctx) {
  if (!Target)
    return;
  // Re-aim any Materialize keyed on `Target` to its next instruction before
  // the erase nulls the WeakTrackingVH. Defensive: EliminateStore and
  // Materialize-at-store are mutually exclusive by the processStore dispatch,
  // so this never fires today, but a store CAN be a Materialize IP (value-side
  // fall-through), so the hook is future-proof.
  relocateDependentMaterializes(Ctx.InsertBeforeDependents, Target,
                                Target->getNextNode());
  Target->eraseFromParent();
  Ctx.Changed = true;
}

void jeandle::EliminateAllocationEffect::apply(jeandle::TransformContext &Ctx) {
  if (eraseAllocation(Target))
    Ctx.Changed = true;
}

void jeandle::MaterializeEffect::apply(jeandle::TransformContext &Ctx) {
  // Emit the materialization sequence. The original allocation's uses are
  // resolved later by the point-sensitive resolution sub-pass (not RAUW'd
  // inline); the allocation itself is erased by EliminateAllocation in Pass 2.
  // The cascade maps drive the two-phase field-store replay (cyclic fields).
  applyMaterialize(Ctx.F, Ctx.Result, *this, Ctx.NewAllocFor, Ctx.MatPerBlock,
                   Ctx.BlockRename, Ctx.Defs, Ctx.NewInvOf, Ctx.IsCascadeTail,
                   Ctx.CascadeKeyOf, Ctx.CascadeGroups);
  Ctx.Changed = true;
}

void jeandle::CreatePHIEffect::apply(jeandle::TransformContext &Ctx) {
  // Insert the unparented PHI created by the analyzer at the head of the merge
  // block (after any existing PHIs), and wire up its incoming values. For each
  // incoming (V, Pred): walk BlockRename to the live merge-pred; if V refers to
  // an OrigAlloc materialized at this (Pred, V), use the per-pred NewInv from
  // MatPerBlock, else fall back to NewAllocFor.
  PHINode *Phi = PhiInst;
  assert(Phi && "CreatePHI effect requires a PhiInst");
  assert(Phi->getParent() == nullptr &&
         "CreatePHI's PhiInst must be unparented at apply time");
  BasicBlock *MergeBB = Block;
  Phi->insertBefore(MergeBB->getFirstInsertionPt());
  assert(PHIIncomingValues.size() == PHIIncomingBlocks.size());
  for (unsigned I = 0; I < PHIIncomingValues.size(); ++I) {
    Value *V = PHIIncomingValues[I];
    BasicBlock *Pred = PHIIncomingBlocks[I];
    if (auto *VI = dyn_cast<Instruction>(V)) {
      auto It = Ctx.MatPerBlock.find({Pred, VI});
      if (It != Ctx.MatPerBlock.end()) {
        V = It->second;
      } else {
        auto It2 = Ctx.NewAllocFor.find(VI);
        if (It2 != Ctx.NewAllocFor.end())
          V = It2->second;
      }
    }
    // An unresolved per-pred placeholder reaches here when this pred's per-pred
    // Materialize was dropped — most commonly because the object is already
    // globally materialized (materialize-before-loops), so the redundant
    // per-pred materialize is elided and no MatPerBlock/NewAllocFor entry is
    // recorded. Leaving the placeholder as the incoming would plant a dangling,
    // never-defined value into the PHI (a latent verifier fault; surfaced when
    // a downstream use — e.g. the deepest-def resolution in
    // resolveMaterializedUses — keeps the PHI alive). Detect the placeholder
    // precisely via the analyzer's placeholder set (NOT by "unparented
    // PHINode": a loop field-PHI incoming can also be momentarily unparented
    // and must be left as-is), then fall back to the global materialization,
    // which exists precisely because the object is materialized; otherwise the
    // original allocation (valid IR that the resolution sub-pass / Pass 2 then
    // handles).
    if (Ctx.Result.PerPredMatPlaceholders.count(V) &&
        ObjID != jeandle::InvalidObjectID) {
      Value *OrigAlloc = Ctx.Result.VirtualObjects[ObjID]->AllocationCall;
      auto ItG = Ctx.NewAllocFor.find(OrigAlloc);
      V = (ItG != Ctx.NewAllocFor.end()) ? ItG->second : OrigAlloc;
    }
    // Resolve the live pred BB through BlockRename. Keyed by (LivePred,
    // this->Block) — CreatePHI is always per-pred, so this->Block IS the target
    // merge M; two per-pred mats from the same PH to different merges route
    // through their own split→MatCont chains. Fallback to (LivePred, null) so a
    // merge consuming an escape-point / Case-A NewInv (TargetMergeBB=null,
    // seeded under {Origin, null}) resolves e.g. then → mat.cont.
    BasicBlock *LivePred = Pred;
    while (true) {
      auto It = Ctx.BlockRename.find({LivePred, Block});
      if (It == Ctx.BlockRename.end()) {
        auto ItN = Ctx.BlockRename.find({LivePred, nullptr});
        if (ItN == Ctx.BlockRename.end())
          break;
        LivePred = ItN->second;
        continue;
      }
      LivePred = It->second;
    }
    Phi->addIncoming(V, LivePred);
  }
  // Register the freshly built PHI as a definition point of OrigAlloc so the
  // point-sensitive resolution sub-pass rewrites post-merge OrigAlloc uses onto
  // it (the dominance check restricts the rewrite to uses the PHI dominates).
  if (RAUWOrigToPHI && ObjID != jeandle::InvalidObjectID) {
    jeandle::VirtualObject &VObj = *Ctx.Result.VirtualObjects[ObjID];
    if (VObj.AllocationCall)
      Ctx.Defs[VObj.AllocationCall].push_back(Phi);
  }
  Ctx.Changed = true;
}

void jeandle::RewritePhiIncomingEffect::apply(jeandle::TransformContext &Ctx) {
  // Resolve this pred's freshly-materialized base (NewInv) from the per-pred
  // placeholder that the Materialize effect recorded into MatPerBlock.
  Value *NewInv = nullptr;
  auto It = Ctx.MatPerBlock.find({Pred, PerPredPlaceholder});
  if (It != Ctx.MatPerBlock.end())
    NewInv = It->second;
  else {
    auto It2 = Ctx.NewAllocFor.find(PerPredPlaceholder);
    if (It2 != Ctx.NewAllocFor.end())
      NewInv = It2->second;
  }
  if (!NewInv)
    return; // the Materialize was dropped (object ineligible) — nothing to
            // rewire.

  // Resolve the live merge-pred (e.g. latch -> ... -> MatCont) through the
  // block-split rename chain, mirroring CreatePHIEffect::apply. Keyed by
  // (LivePred, TargetMergeBB) — null for this effect (Case-A only: mat at PH
  // end, single MatCont, no critical-edge split).
  BasicBlock *LivePred = Pred;
  while (true) {
    auto R = Ctx.BlockRename.find({LivePred, TargetMergeBB});
    if (R == Ctx.BlockRename.end())
      break;
    LivePred = R->second;
  }

  // Re-derive the carried value over the materialized base at the
  // materialization point. NewInv is the materialization invoke; its result
  // dominates the normal-dest (MatCont), so the GEP goes there (same placement
  // as the field-replay slots in applyMaterialize). Offset 0 (bitcast/identity
  // carry) reuses NewInv directly.
  Value *Rederived = NewInv;
  if (ByteOffset != 0) {
    BasicBlock *MatCont;
    if (auto *II = dyn_cast<InvokeInst>(NewInv))
      MatCont = II->getNormalDest();
    else
      MatCont = cast<Instruction>(NewInv)->getParent();
    IRBuilder<> B(MatCont, MatCont->getFirstInsertionPt());
    Rederived = B.CreateInBoundsGEP(B.getInt8Ty(), NewInv,
                                    B.getInt64(ByteOffset), "pea.matoff");
  }

  // Rewire the carrying PHI's incoming for this predecessor. splitBasicBlock
  // in applyMaterialize already updated the incoming block to the post-split
  // MatCont; LivePred is resolved through BlockRename as a belt-and-suspenders.
  for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i)
    if (Phi->getIncomingBlock(i) == LivePred) {
      Phi->setIncomingValue(i, Rederived);
      break;
    }
  Ctx.Changed = true;
}

// Apply every effect where isCfgKill()==CfgKills, in SeqNo order (Jeandle's
// substitute for Graal's list-order — see Effect::SeqNo). Pass 1 calls this
// with CfgKills=false (every effect except EliminateAllocation); Pass 2 with
// CfgKills=true (EliminateAllocation only).
void jeandle::EffectList::apply(jeandle::TransformContext &Ctx, bool CfgKills) {
  SmallVector<jeandle::Effect *, 16> Order;
  Order.reserve(Effects.size());
  for (auto &E : Effects)
    Order.push_back(E.get());
  llvm::sort(Order, [](const jeandle::Effect *A, const jeandle::Effect *B) {
    return A->SeqNo < B->SeqNo;
  });
  for (jeandle::Effect *E : Order)
    if (E->isCfgKill() == CfgKills)
      E->apply(Ctx);
}

// Of two definitions that BOTH already dominate a target use, return true when
// Candidate is the closer (deeper / later) one. The dominating defs of a single
// use form a totally-ordered dominator chain (PEA uniqueness invariant), so the
// deepest is well-defined and unique. Iterating keeps the deepest: a new
// Candidate wins iff the current best dominates it (Candidate is strictly
// deeper, or later in the same block). This is Jeandle's explicit form of
// Graal's per-point alias resolution — Graal replaces the allocation node in
// place so each use automatically sees the most-recent materialized value
// (getAlias/getAliasAndResolve, PartialEscapeClosure.java ~1563-1584; the
// aliases map is reset per node, EffectsClosure.java:279). LLVM's Analysis/
// Transform split forbids mutating IR during analysis, so OrigAlloc persists as
// a real invoke and Jeandle must instead pick the most-recent materialized def
// among the surviving defs.
static bool isCloserDominatingDef(Value *Candidate, Value *Current,
                                  const DominatorTree &DT) {
  if (!Current || Candidate == Current)
    return !Current;
  auto *CandidateI = dyn_cast<Instruction>(Candidate);
  if (!CandidateI)
    return false;
  return DT.dominates(Current, CandidateI);
}

// Point-sensitive resolution of original-allocation uses — Jeandle's analog of
// Graal's per-point alias resolution (the `aliases` map / getAlias /
// getAliasAndResolve, which Graal maintains because it REPLACES the allocation
// node with a VirtualObjectNode during analysis). LLVM's Analysis/Transform
// split forbids mutating IR during analysis, so OrigAlloc persists as a real
// invoke until here; this pass makes its ROLE Graal-equivalent by resolving
// every surviving use to the def that dominates it, then Pass 2 erases the
// now-use-less allocation.
//
// Run AFTER Pass 1 (which has placed every materialize NewInv and merge PHI and
// settled the CFG via block splits), so a freshly-computed DominatorTree is
// valid. For each OrigAlloc use, pick the CLOSEST (deepest) def in
// Defs[OrigAlloc] that dominates it. Normally only one def dominates a use, but
// a later materialization or merge PHI may shadow an earlier dominating def; in
// that case the deeper/later def is the SSA value that represents this point.
// Deopt-bundle operands are scrubbed to a typed null (PEA stays
// deopt-agnostic).
static void
resolveMaterializedUses(Function &F,
                        DenseMap<Value *, SmallVector<Value *, 4>> &Defs) {
  if (Defs.empty())
    return;
  DominatorTree DT(F);

  for (auto &Kv : Defs) {
    Value *OrigAlloc = Kv.first;
    const SmallVector<Value *, 4> &DefList = Kv.second;
    if (OrigAlloc->use_empty())
      continue;
    Value *NullVO =
        ConstantPointerNull::get(cast<PointerType>(OrigAlloc->getType()));
    for (Use &U : llvm::make_early_inc_range(OrigAlloc->uses())) {
      // Pick the NEAREST (deepest) dominating definition, not merely the first
      // in DefList order. DefList insertion order = Pass-1 RPO apply order, so
      // a loop-header materialized-ptr PHI is inserted before a loop-body
      // NewInv; when both dominate the same use, "first" would wrongly thread
      // the header PHI (the previous iteration's merged pointer). Iterating to
      // the deepest converges on the unique dominator-tree leaf among the
      // dominating defs.
      Value *Dom = nullptr;
      for (Value *Def : DefList) {
        if (!DT.dominates(Def, U))
          continue;
#ifndef NDEBUG
        // PEA invariant: dominating defs of one use form a dominator chain. If
        // two were ever incomparable, "deepest" would be order-dependent — the
        // exact bug class this guard catches. Cheap to verify; piggybacks on
        // the loop.
        if (Dom && Def != Dom)
          assert(
              (DT.dominates(cast<Instruction>(Dom), cast<Instruction>(Def)) ||
               DT.dominates(cast<Instruction>(Def), cast<Instruction>(Dom))) &&
              "PEA: dominating defs of a use must be totally ordered");
#endif
        if (isCloserDominatingDef(Def, Dom, DT))
          Dom = Def;
      }
      if (!Dom)
        continue; // no dominating def; leave for Pass 2's poison RAUW.
      // Scrub deopt-bundle operands to a typed null rather than threading a
      // (possibly non-dominating) NewInv/PHI into a sibling's deopt bundle.
      if (auto *CB = dyn_cast<CallBase>(U.getUser())) {
        unsigned OpIdx = U.getOperandNo();
        if (CB->isBundleOperand(OpIdx) &&
            CB->getOperandBundleForOperand(OpIdx).isDeoptOperandBundle()) {
          U.set(NullVO);
          continue;
        }
      }
      U.set(Dom);
    }
  }
}

PreservedAnalyses PartialEscapeTransform::run(Function &F,
                                              FunctionAnalysisManager &FAM) {
  // Gate on jeandle.java_method_compilation.
  Module *M = F.getParent();
  if (!M || !M->getNamedMetadata(jeandle::Metadata::JavaMethodCompilation))
    return PreservedAnalyses::all();

  // Pre-Pass 1 may rewrite E.Block / E.InsertBefore for IsPerPred Materialize
  // effects sitting on critical-edge preds; take a non-const reference so we
  // can mutate the analysis result before applying it.
  auto &Result = FAM.getResult<PartialEscapeAnalysis>(F);
  if (!Result.hasOptimizationOpportunity())
    return PreservedAnalyses::all();

  bool Changed = false;

  // Map from each virtual object's original allocation to the new
  // materialized invoke produced by applyMaterialize. Populated in SeqNo
  // order across Pass 1 so that nested-virtual materializations record their
  // inner pointer before the outer's apply needs to look it up.
  DenseMap<Value *, Value *> NewAllocFor;
  // Per-pred (analyzer-recorded pred BB, OrigAlloc) → NewInv. Used by
  // CreatePHI to pick the right NewInv for each merge incoming when the same
  // OrigAlloc is materialized at multiple preds.
  DenseMap<std::pair<BasicBlock *, Value *>, Value *> MatPerBlock;
  // Block-split rename map: (Origin BasicBlock, target-merge BB) → next block
  // in the post-split MatCont chain. Used to resolve the analyzer-recorded PHI
  // incoming block (which named the original pred pre-split) to the live
  // merge-pred. Keyed by (PH, target-merge) so two per-pred materializes from
  // the same PH to DIFFERENT target merges (two split edges) do not collide —
  // each merge's CreatePHI incoming routes through its own edge's split→MatCont
  // chain. The target-merge is null for the Case-A / global path (mat at PH
  // end, single MatCont, no critical-edge split).
  DenseMap<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *> BlockRename;
  // Per-OrigAlloc definition points (every materialize NewInv + every merge
  // PHI) populated during Pass 1. Consumed by resolveMaterializedUses after
  // Pass 1 to rewrite each surviving OrigAlloc use to its dominating def.
  DenseMap<Value *, SmallVector<Value *, 4>> Defs;

  // -------------------------------------------------------------------------
  // PRE-PASS: split critical edges before per-pred materialisation.
  //
  // A per-pred Materialize replaces PH's terminator with a materialisation
  // invoke carrying an OOM unwind edge. If PH has multiple successors the OOM
  // would become observable on every PH→* edge — a Java-semantics change. So
  // for each IsPerPred Materialize on a PH with >1 successor, split the PH→S
  // edge where S is the target merge (MaterializeEffect::TargetMergeBB, the
  // merge whose MergeProcessor requested the per-pred mat), then re-aim the
  // per-pred Materialize effects onto the new edge-block PH' and seed
  // BlockRename[{PH, S}] = PH' so CreatePHI's BlockRename-chain walk (keyed by
  // (LivePred, this->Block=S)) routes the analyzer-recorded PH incoming through
  // PH' to MatCont. Keying by (PH, S) means two per-pred mats from the same PH
  // to different target merges (S1, S2) split two distinct edges and do not
  // collide.
  {
    struct EdgeKey {
      BasicBlock *PH;
      BasicBlock *S; // target merge
      unsigned SuccIdx;
    };
    SmallVector<EdgeKey, 4> Splits;
    DenseSet<std::pair<BasicBlock *, BasicBlock *>> SeenEdges;
    for (const auto &KvOut : Result.BlockEffects) {
      for (const auto &E : KvOut.second) {
        const auto *M = dyn_cast<jeandle::MaterializeEffect>(&E);
        if (!M || !M->IsPerPred)
          continue;
        BasicBlock *PH = E.Block;
        BasicBlock *S = M->TargetMergeBB;
        if (!PH || !S)
          continue;
        Instruction *Term = PH->getTerminator();
        if (!Term || Term->getNumSuccessors() <= 1)
          continue;
        // The critical-edge split is unconditional: the lock model deletes the
        // original monitorenter and re-emits it at the materialize point, so no
        // surviving enter sits in PH that would lose its dominating receiver if
        // the Materialize moved to a new edge block.
        unsigned SuccIdx = UINT_MAX;
        for (unsigned i = 0, n = Term->getNumSuccessors(); i < n; ++i) {
          if (Term->getSuccessor(i) == S) {
            SuccIdx = i;
            break;
          }
        }
        if (SuccIdx == UINT_MAX)
          continue; // S not a successor of PH — malformed.
        if (S->hasNPredecessors(1))
          continue; // single-pred S: no critical edge to split.
        auto Key = std::make_pair(PH, S);
        if (!SeenEdges.insert(Key).second)
          continue;
        Splits.push_back({PH, S, SuccIdx});
      }
    }
    DenseMap<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *> PHRename;
    for (const EdgeKey &K : Splits) {
      Instruction *Term = K.PH->getTerminator();
      BasicBlock *NewBB = SplitCriticalEdge(
          Term, K.SuccIdx,
          CriticalEdgeSplittingOptions().setMergeIdenticalEdges());
      if (!NewBB)
        continue;
      NewBB->setName("pea.crit.split");
      PHRename[{K.PH, K.S}] = NewBB;
      // Seed the transform's BlockRename so the CreatePHI handler's chain
      // walk (keyed by (LivePred, this->Block=S)) routes PH → NewBB and then
      // (after applyMaterialize) NewBB → MatCont.
      BlockRename[{K.PH, K.S}] = NewBB;
    }
    // Re-aim per-pred Materialize effects from their original PH bucket onto
    // the new edge-block. Also move the BlockEffects entry so the RPO walk
    // applies the effects at the correct block boundary. The Stay effects
    // remain in the Old bucket; each Move effect is spliced out, re-aimed via
    // the MaterializeEffect setters, and added to the New bucket. The
    // E.TargetMergeBB == M filter ensures a per-pred mat for (PH, M1) is NOT
    // moved onto M2's split edge when two merges share PH.
    if (!PHRename.empty()) {
      for (const auto &KvOut : PHRename) {
        BasicBlock *Old = KvOut.first.first;
        BasicBlock *S = KvOut.first.second;
        BasicBlock *New = KvOut.second;
        auto It = Result.BlockEffects.find(Old);
        if (It == Result.BlockEffects.end())
          continue;
        jeandle::EffectList &Src = It->second;
        jeandle::EffectList Move;
        size_t I = 0;
        while (I < Src.size()) {
          jeandle::Effect &E = Src[I];
          auto *M = dyn_cast<jeandle::MaterializeEffect>(&E);
          if (M && M->IsPerPred && E.Block == Old && M->TargetMergeBB == S) {
            M->setBlock(New);
            M->setInsertBefore(&*New->getFirstNonPHIOrDbg());
            Move.add(Src.spliceOut(I));
          } else {
            ++I;
          }
        }
        Result.BlockEffects[New].addAll(Move);
      }
    }
  }

  // Recompute RPOT AFTER any critical-edge splits so the new edge-blocks
  // are visited in Pass 1.
  ReversePostOrderTraversal<Function *> RPOT(&F);

  // Build the per-escape-point merged lock lists (one global depth-sort per
  // materialize point) before Pass 1 applies effects.
  Result.computeEscapePointLocks();

  // Cascade coordination for cyclic-field materialization: group every
  // Materialize effect by its shared escape-point InsertBefore so the transform
  // can emit every cascade NewInv before any field store (Jeandle's analog of
  // Graal's single CommitAllocationNode, which holds every object and every
  // initializer value at one point so cyclic peer references resolve). Capture
  // each effect's key ONCE here; never re-read the WeakTrackingVH InsertBefore
  // for cascade membership at apply time (a sibling effect may null it). The
  // max-SeqNo member of each group is the "tail" that replays the whole group's
  // field stores; a singleton (null or unique InsertBefore) is its own tail.
  // Per-pred effects were re-aimed onto their split edge blocks by the pre-pass
  // above, so per-pred cascades (one per edge block) key correctly here.
  DenseMap<Instruction *, SmallVector<const jeandle::MaterializeEffect *, 4>>
      CascadeGroups;
  DenseMap<const jeandle::MaterializeEffect *, Instruction *> CascadeKeyOf;
  DenseSet<const jeandle::MaterializeEffect *> IsCascadeTail;
  DenseMap<const jeandle::MaterializeEffect *, InvokeInst *> NewInvOf;
  DenseMap<Instruction *, SmallVector<jeandle::MaterializeEffect *, 4>>
      InsertBeforeDependents;
  {
    DenseMap<Instruction *, SmallVector<const jeandle::MaterializeEffect *, 4>>
        ByKey;
    for (auto &Kv : Result.BlockEffects)
      for (jeandle::Effect &E : Kv.second) {
        auto *M = dyn_cast<jeandle::MaterializeEffect>(&E);
        if (!M)
          continue;
        Instruction *Key = dyn_cast_or_null<Instruction>(M->InsertBefore);
        CascadeKeyOf[M] = Key;
        if (Key) {
          ByKey[Key].push_back(M);
          // Reverse index for the eager-update hook
          // (relocateDependentMaterializes): keyed by the live InsertBefore
          // instruction so an erase effect can find and re-aim every
          // dependent Materialize before nulling the WeakTrackingVH.
          InsertBeforeDependents[Key].push_back(M);
        } else
          // Degenerate (InsertBefore already null at pre-scan): the effect is
          // its own singleton tail and replays its own stores in its own
          // MatCont.
          IsCascadeTail.insert(M);
      }
    for (auto &Kv : ByKey) {
      llvm::sort(Kv.second, [](const jeandle::MaterializeEffect *A,
                               const jeandle::MaterializeEffect *B) {
        return A->SeqNo < B->SeqNo;
      });
      // The max-SeqNo member is applied last, so by the time it runs every
      // sibling NewInv exists and its MatCont is the cascade's final block.
      IsCascadeTail.insert(Kv.second.back());
      CascadeGroups[Kv.first] = std::move(Kv.second);
    }
  }

  // -------------------------------------------------------------------------
  // Pass 1: non-cfgKill effects (ReplaceLoad, ReplaceCall, EliminateStore,
  // Materialize, CreatePHI) — applied per-block in RPO via EffectList::apply,
  // which sorts by SeqNo and dispatches each effect's apply() through
  // TransformContext (Jeandle's analog of Graal's
  // apply(graph, obsoleteNodes, cfgKills=false)). isCfgKill() partitions the
  // two passes; EliminateAllocation is the only cfgKill, so it is skipped here.
  // -------------------------------------------------------------------------
  jeandle::TransformContext Ctx{F,
                                Result,
                                NewAllocFor,
                                MatPerBlock,
                                BlockRename,
                                Defs,
                                Changed,
                                CascadeGroups,
                                CascadeKeyOf,
                                IsCascadeTail,
                                NewInvOf,
                                InsertBeforeDependents};

  for (BasicBlock *BB : RPOT) {
    auto It = Result.BlockEffects.find(BB);
    if (It == Result.BlockEffects.end())
      continue;
    It->second.apply(Ctx, /*CfgKills=*/false);
  }

  // -------------------------------------------------------------------------
  // Resolution sub-pass: rewrite each surviving OrigAlloc use to its dominating
  // materialize NewInv / merge PHI (Jeandle's analog of Graal's point-sensitive
  // alias resolution). Runs after Pass 1 has settled the CFG; before Pass 2 so
  // the allocation becomes use-empty before EliminateAllocation erases it.
  // -------------------------------------------------------------------------
  if (!Defs.empty()) {
    resolveMaterializedUses(F, Defs);
    Changed = true;
  }

  // -------------------------------------------------------------------------
  // Pass 2: cfgKill effects (EliminateAllocation only) — same dispatch with
  // CfgKills=true. Runs after the resolution sub-pass so each allocation is
  // use-empty before it is erased.
  // -------------------------------------------------------------------------
  for (BasicBlock *BB : RPOT) {
    auto It = Result.BlockEffects.find(BB);
    if (It == Result.BlockEffects.end())
      continue;
    It->second.apply(Ctx, /*CfgKills=*/true);
  }

  // Erase parented Case-B alias PHIs that the analyzer flagged
  // as redundant for NeverEscapes VOs. Pass 2 above already RAUW'd
  // every OrigAlloc incoming to poison via eraseAllocation, so the
  // PHIs survive as `phi [poison, poison]`. Replace each with poison
  // and erase. WeakTrackingVH auto-nulls if some other code path
  // already deleted the PHI (e.g. an outer iteration's dead-block
  // sweep), so the null check below is load-bearing.
  for (auto &VH : Result.CaseBAliasedPhisToErase) {
    Value *V = VH;
    if (!V)
      continue;
    auto *Phi = dyn_cast<PHINode>(V);
    if (!Phi || !Phi->getParent())
      continue;
    Phi->replaceAllUsesWith(PoisonValue::get(Phi->getType()));
    Phi->eraseFromParent();
    Changed = true;
  }

  if (!Changed)
    return PreservedAnalyses::all();

  // Folded JavaOps may have left behind `br i1 true|false, ...` terminators
  // (monitorenter elision RAUWs the result to `true`, after which a
  // conditional branch on the result becomes constant). Use
  // ConstantFoldTerminator to collapse those before the unreachable-block
  // sweep so the slow-path blocks of synchronized regions get cleaned up.
  for (BasicBlock &BB : llvm::make_early_inc_range(F)) {
    ConstantFoldTerminator(&BB, /*DeleteDeadConditions=*/true,
                           /*TLI=*/nullptr, /*DTU=*/nullptr);
  }

  // Fold trivial PHIs. Loop-body partial escape builds a materializedValuePhi
  // at each enclosing loop header, but because a single allocation (the
  // preheader NewInv) is the materialized value on every path, each such phi
  // is trivial — phi(NewInv, NewInv), or for nested loops a dead cycle
  // phi(innerPhi, NewInv) where innerPhi is itself phi(self, outerPhi). The
  // trivially-dead sweep below cannot break that cycle (each phi is "used" by
  // the next), so fold them first: PHINode::hasConstantValue collapses both
  // phi(X,X) and phi(self, X) to X. Iterate to fixpoint so a fold that makes
  // an enclosing phi trivial is caught. (This mirrors what downstream
  // GVN/InstCombine would do; doing it here keeps the PEA output clean and is
  // required for the nested-loop dead cycle, which downstream sees later.)
  bool FoldedPhi = true;
  while (FoldedPhi) {
    FoldedPhi = false;
    for (BasicBlock &BB : F) {
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        auto *PN = dyn_cast<PHINode>(&I);
        if (!PN || PN->getNumIncomingValues() == 0)
          continue;
        if (Value *V = PN->hasConstantValue()) {
          PN->replaceAllUsesWith(V);
          PN->eraseFromParent();
          FoldedPhi = true;
        }
      }
    }
  }

  // Sweep trivially-dead instructions that became unused after our rewrites
  // (e.g., GEPs derived from eliminated allocations whose only users were the
  // loads/stores we replaced in Pass 1). Iterate to fixpoint so cascading
  // deaths are caught.
  bool LocalChanged = true;
  while (LocalChanged) {
    LocalChanged = false;
    for (BasicBlock &BB : llvm::make_early_inc_range(F)) {
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        if (isInstructionTriviallyDead(&I)) {
          I.eraseFromParent();
          LocalChanged = true;
        }
      }
    }
  }

  // Clean up unwind blocks that became unreachable after invoke→br rewrites
  // and slow-path blocks orphaned by ConstantFoldTerminator above.
  EliminateUnreachableBlocks(F);

  // Drop references on still-unparented OwnedInsts before verifyFunction: the
  // PEAResult destructor (which runs in the analysis manager, after the
  // verifier) would clean them up, but without this sweep an unparented helper
  // holding a use of a now-parented helper trips the verifier's "use list of X
  // is in IR but X's user is not". dropAllReferences() severs the use list
  // without freeing the value; the dtor still owns the WeakTrackingVH and does
  // the eventual deleteValue.
  for (WeakTrackingVH &VH : Result.OwnedInsts) {
    if (Value *V = VH) {
      if (auto *I = dyn_cast<Instruction>(V)) {
        if (!I->getParent())
          I->dropAllReferences();
      }
    }
  }

  // In debug builds, verify the rewritten IR so any PEA malformation (broken
  // SSA from a stale RAUW, mis-ordered CreatePHI vs per-pred Materialize,
  // value-side virtual leak, critical-edge replacement, missing funclet
  // bundle, ...) is caught here with an actionable message rather than later
  // in RewriteStatepointsForGC or assembly emission.
#ifndef NDEBUG
  if (verifyFunction(F, &errs())) {
    errs() << "PEA: produced malformed IR for " << F.getName() << "\n";
    llvm_unreachable("PartialEscapeTransform produced malformed IR");
  }
#endif

  return PreservedAnalyses::none();
}
