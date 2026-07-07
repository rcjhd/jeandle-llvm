//===- PartialEscape.h - PEA shared data structures -------------*- C++ -*-===//
//
// Copyright (c) 2026, the Jeandle-LLVM Authors. All Rights Reserved.
//
// Part of the Jeandle-LLVM project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared data structures used by both PartialEscapeAnalysis and
// PartialEscapeTransform. Light header: forward-declares LLVM IR
// classes and only includes ADT containers. Method bodies are in
// llvm/lib/Analysis/Jeandle/PartialEscape.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_JEANDLE_PARTIALESCAPE_H
#define LLVM_ANALYSIS_JEANDLE_PARTIALESCAPE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/ValueHandle.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace llvm {

class BasicBlock;
class CallBase;
class Constant;
class DataLayout;
class Function;
class GetElementPtrInst;
class Instruction;
class InvokeInst;
class LLVMContext;
class PHINode;
class raw_ostream;
class Type;
class Value;

namespace jeandle {

class AliasMap;
class Effect;
class EffectList;
class PEABlockState;
struct TransformContext;

// ===========================================================================
// 1.1  ObjectID and VirtualObject
// ===========================================================================

using ObjectID = unsigned;
static constexpr ObjectID InvalidObjectID = ~0u;

// Identity of a virtual monitorenter, carried per element of an
// ObjectState's live lock stack. EnterCall is the original
// jeandle.monitorenter call site (the effects-side anchor used when
// un-eliding the call on materialisation); BytecodeDepth is the lock-nesting
// ordering key at the enter site, sourced from the analyzer's monotonic
// NextLockEnterOrder (cached per call site in the analyzer's LockDepthCache,
// see PartialEscapeAnalysis.cpp). It is NOT the true Java-bytecode monitor
// depth — the frontend no longer attaches that — but an RPO-order proxy that
// is sound for every load-bearing use here (cascade `<`, re-emit sort,
// merge-time stack identity, strict-increasing assert): under the
// structured-locking assumption PEA relies on, "lock X acquired before Y
// while both live" <=> "X's enter dominates Y's" <=> "RPO visits X before Y".
// The analyzer-side mirror struct LockEnter (PartialEscapeAnalysis.cpp)
// additionally carries an Order tag for loop-fixpoint convergence checks,
// which compare CallBase identity only since Order is refreshed on every
// re-push.
struct MonitorIdRef {
  CallBase *EnterCall;
  uint32_t BytecodeDepth;

  bool operator==(const MonitorIdRef &O) const {
    return EnterCall == O.EnterCall && BytecodeDepth == O.BytecodeDepth;
  }
  bool operator!=(const MonitorIdRef &O) const { return !(*this == O); }
};

// Self-contained description of a monitorenter to re-emit at the materialize
// point. Graal analog: a synthetic MonitorEnterNode created at the
// CommitAllocationNode during lowering (DefaultJavaLoweringProvider
// finishAllocatedObjects), sorted ascending by lock depth. Jeandle captures
// this from the original enter call at ANALYSIS time because the lock model
// DELETES the original enter from IR — the transform cannot depend on
// the original call's lifetime. Callee is the jeandle.monitorenter_*
// function; NonReceiverArgs are operands 1..N (e.g. the BasicLock); the
// receiver (operand 0) is the freshly materialized pointer by construction.
// BytecodeDepth is the ascending re-emit sort key (Graal getLockDepth).
struct MaterializedLock {
  Function *Callee = nullptr;
  SmallVector<Value *, 2> NonReceiverArgs;
  uint32_t BytecodeDepth = 0;
};

// A lock re-emitted at a materialization point, tagged with its source
// MaterializeEffect so the transform can pick the right per-effect receiver
// for each lock when a cascade group's locks are merged and globally
// depth-sorted. SourceEffect is the per-effect receiver-lookup key (Jeandle's
// analog of Graal's `allocations[commit.getObjectIndex(monitorId)]`,
// DefaultJavaLoweringProvider.java:1152) — strictly more precise than an
// OrigAlloc key, which would be last-write-wins across per-pred
// materializations of the same object (see applyMaterialize NewAllocFor
// comment, PartialEscapeTransform.cpp:374-377). See
// PEAResult::EscapePointLocks.
class MaterializeEffect;
struct MergedLock {
  Function *Callee = nullptr;
  SmallVector<Value *, 2> NonReceiverArgs;
  uint32_t BytecodeDepth = 0;
  const MaterializeEffect *SourceEffect = nullptr; // per-effect receiver key
};

class VirtualObject {
public:
  enum ClassKind : uint8_t { Instance, Array };

  struct FieldDesc {
    int64_t Offset;
    Type *LLVMType;
    uint8_t ByteSize;
    bool IsReference;

    bool overlaps(int64_t Off, uint8_t Size) const {
      return Off < Offset + ByteSize && Offset < Off + Size;
    }
  };

private:
  const ObjectID ID;
  ClassKind Kind;

public:
  // The original `jeandle.new_instance`/`jeandle.new_array` invoke emitted by
  // the JDK abstract interpreter. This is Jeandle's analog of Graal's
  // VirtualObjectNode — the identity token that downstream IR uses reference.
  // LLVM-CONSTRAINED DIVERGENCE from Graal: Graal REPLACES the NewInstanceNode
  // with a VirtualObjectNode DURING analysis (tool.replaceWithVirtual) and
  // resolves uses through the point-sensitive `aliases` map; there is no
  // persistent "OrigAlloc". Jeandle's Analysis pass cannot mutate IR (LLVM's
  // Analysis/Transform split), so OrigAlloc persists as a real invoke until the
  // Transform, where it is either eliminated (NeverEscapes) or resolved per-
  // point to the materialized value (PartialEscapeTransform's
  // resolveMaterializedUses — the DT-driven analog of Graal's alias resolution,
  // computed at transform time because the analysis cannot mutate). Behaviorally
  // equivalent: OrigAlloc's role is a pure identity token, never a real
  // allocation in the final IR.
  CallBase *AllocationCall = nullptr;

  uintptr_t Klass = 0;
  uint32_t SizeInBytes = 0;

  uint32_t ArrayLength = 0;
  Type *ArrayElementType = nullptr;
  uint32_t ArrayIndexScale = 0;
  uint32_t ArrayBaseOffset = 0;
  uint32_t ArrayLengthLimit = 0;

  SmallVector<FieldDesc, 8> Fields;

  // A "synthetic" VirtualObject is created at a multi-pred merge by the
  // analyzer's processBlockPhis when every incoming resolves to a DIFFERENT
  // but COMPATIBLE virtual object (same Klass / kind / array dimensions /
  // lock state, and the per-pred allocations have no other observable
  // identity). Compatibility does NOT compare per-instance field count:
  // Fields is lazily populated (path-dependent), so its size is not a
  // structural invariant; the synthetic VO's Fields is the UNION of the
  // per-pred Fields. There is no backing allocation — AllocationCall is
  // non-null (cloned from the first per-pred VO) but MUST NOT be erased or
  // used as a Materialize effect target. SyntheticSourceIDs holds the
  // per-pred VOs in PHI-incoming order; SyntheticPhi is the merge-block
  // PHINode the VO is aliased to. A later attempt to materialize a synthetic
  // VO conservatively marks it and every per-pred source VO ineligible so
  // the original allocations and stores survive.
  // TODO(cascade-materialize): see PartialEscapeAnalysis.cpp materializeAt().
  bool IsSynthetic = false;
  SmallVector<ObjectID, 4> SyntheticSourceIDs;
  PHINode *SyntheticPhi = nullptr;

  VirtualObject(ObjectID id, ClassKind k, CallBase *alloc)
      : ID(id), Kind(k), AllocationCall(alloc) {}

  ObjectID getID() const { return ID; }
  ClassKind getKind() const { return Kind; }
  bool isInstance() const { return Kind == Instance; }
  bool isArray() const { return Kind == Array; }

  int getOrCreateFieldIndex(int64_t Offset, Type *Ty, const DataLayout &DL);

  // Result of matching a GEP against the array's element-address pattern.
  // Index is the (possibly symbolic) Value* that names the Java-level
  // element index; ElementType is the per-element LLVM type. Callers must
  // inspect Index: a ConstantInt yields a constant element index (combine
  // it with ArrayBaseOffset + scale for the canonical byte offset); a
  // symbolic index forces the caller to bail (materialize the array).
  struct ArrayElementGEPMatch {
    llvm::Value *Index;
    llvm::Type *ElementType;
  };
  std::optional<ArrayElementGEPMatch>
  matchArrayElementGEP(GetElementPtrInst *GEP,
                       const llvm::DataLayout &DL) const;

  // Copy the structural (non-identity, non-synthetic) fields from O into *this.
  // Shared by duplicate() and PEAResult::createVirtualObject() so the field
  // list lives in exactly one place (avoids drift when a field is added).
  void copyStructuralFieldsFrom(const VirtualObject &O);

  std::unique_ptr<VirtualObject> duplicate() const;
};

// ===========================================================================
// 1.2  FieldValue
// ===========================================================================

class FieldValue {
public:
  enum Tag : uint8_t { Unknown, Scalar, VirtualRef, MaterializedRef };

private:
  Tag T = Unknown;
  union U {
    Value *V;
    ObjectID Ref;
    U() : V(nullptr) {}
  } U;
  Type *DeclaredType = nullptr;

public:
  FieldValue() = default;

  static FieldValue unknown() { return {}; }

  static FieldValue scalar(Value *V);
  static FieldValue virtualRef(ObjectID ID, Type *RefTy);
  static FieldValue materializedRef(Value *Ptr);

  bool isUnknown() const { return T == Unknown; }
  bool isScalar() const { return T == Scalar; }
  bool isVirtualRef() const { return T == VirtualRef; }
  bool isMaterializedRef() const { return T == MaterializedRef; }

  Value *getScalar() const {
    assert(isScalar());
    return U.V;
  }
  ObjectID getVirtualRef() const {
    assert(isVirtualRef());
    return U.Ref;
  }
  Value *getMaterialized() const {
    assert(isMaterializedRef());
    return U.V;
  }
  Type *getDeclaredType() const { return DeclaredType; }

  static Constant *defaultFor(Type *FieldType);

  bool shallowEquals(const FieldValue &O) const;
};

// ===========================================================================
// 1.3  ObjectState
// ===========================================================================

class ObjectState {
public:
  enum StateKind : uint8_t { Virtual, Materialized };

private:
  StateKind Kind = Virtual;
  // Per-VO live monitor stack: each element is a MonitorIdRef identifying
  // the (enter-call, bytecode-depth) pair pushed by a folded monitorenter
  // on this VO whose matching monitorexit hasn't yet been seen on this
  // path. The EnterCall + BytecodeDepth fields are needed for un-elision
  // of the original calls on materialisation, the narrow cascade rule, and
  // merge-time stack-identity comparisons. The analyzer keeps a parallel
  // LiveLockEnters DenseMap keyed by ObjectID; this on-VO copy mirrors the
  // per-VO truth and is kept in lockstep by foldMonitorEnter /
  // foldMonitorExit.
  SmallVector<MonitorIdRef, 2> Locks;
  Value *MaterializedValue = nullptr;

public:
  // ObjectState carries ONLY the per-VO virtual/materialized flag, the live lock
  // stack, and (once materialized) the materialized pointer. Per-FIELD state
  // does NOT live here. Graal's ObjectState.entries is authoritative because
  // Graal propagates an ObjectState[] across the CFG inside
  // PartialEscapeBlockState; Jeandle cannot (LLVM's Analysis/Transform split +
  // SSA single-pass walk — see the STATE MODEL comment in
  // PartialEscapeAnalysis.cpp), so field values are tracked in the
  // analyzer-wide FieldStates DenseMap keyed by (ObjectID, byte-offset), and
  // ObjectState intentionally carries no field-storage member.
  // Copy/move/assign/dtor are implicitly generated: ObjectState is a plain bag
  // of value members, so PEABlockState's array-level copy-on-write (which
  // deep-copies every slot) needs no per-slot sharing annotation here.

  StateKind getKind() const { return Kind; }
  bool isVirtual() const { return Kind == Virtual; }
  bool isMaterialized() const { return Kind == Materialized; }

  Value *getMaterializedValue() const {
    assert(isMaterialized() && MaterializedValue);
    return MaterializedValue;
  }
  bool hasLocks() const { return !Locks.empty(); }
  // Element-wise lock-stack comparison used by the depth-aware merge-time
  // stack-identity check (mergeStates) and the pre-cascade in
  // foldMonitorEnter. Compares both EnterCall and BytecodeDepth. BytecodeDepth
  // is the Analyzer's NextLockEnterOrder proxy (cached per call site in
  // LockDepthCache, stable across loop-fixpoint iterations); the analyzer-side
  // BlockExitData snapshot identity (blockExitInfoEquivalent) compares only
  // EnterCall, since BytecodeDepth's underlying Order is refreshed on every
  // loop-fixpoint re-push.
  bool locksEqual(const ObjectState &Other) const {
    if (Locks.size() != Other.Locks.size())
      return false;
    for (size_t I = 0, E = Locks.size(); I < E; ++I)
      if (Locks[I] != Other.Locks[I])
        return false;
    return true;
  }

  // Graal analog: ObjectState.escape(ValueNode materialized). This is the
  // pure virtual->materialized STATE FLIP only — it does NOT emit a
  // Materialize effect or build an allocation invoke (that is the caller's
  // job, e.g. materializeAt / ensureMaterialized). Named `escape` after Graal
  // to avoid colliding with the real-materialization method family below.
  void escape(Value *Ptr) {
    assert(isVirtual());
    assert(Ptr);
    Kind = Materialized;
    MaterializedValue = Ptr;
    // Graal ObjectState.escape (ObjectState.java:195-202) retains the lock
    // state across the virtual->materialized flip — the materialized state
    // still reads locks for re-emit and the strict-lock cascade. We match
    // that: escape() does NOT clear Locks. Callers that need the live lock
    // state dropped (the live-path materializeAt, via ClearLockState ->
    // clearLocks) do so explicitly before flipping; the analyzer-side
    // LiveLockEnters/LockCounts maps remain the cross-block authority, so
    // any retained on-VO locks are informational only.
  }

  void addLock(MonitorIdRef M) {
    assert(isVirtual());
    // Graal ObjectState.addLock guarantees strictly descending depth on the
    // head (ObjectState.java:212): the new (innermost) lock's depth is
    // strictly greater than the previous head. Jeandle's vector runs
    // front=min(outermost)..back=max(innermost), so the invariant is that the
    // pushed depth strictly exceeds the current back. Mirrors Graal's
    // GraalError.guarantee; the lock cascade (ensureMaterialized /
    // materializeVirtualLocksBefore) relies on front()=min / back()=max.
    assert(Locks.empty() || M.BytecodeDepth > Locks.back().BytecodeDepth);
    Locks.push_back(M);
  }
  void removeLock() {
    assert(isVirtual() && !Locks.empty());
    Locks.pop_back();
  }
  // Clear the entire lock stack — used by materializeAt after un-eliding all
  // pending enters so commit()'s "non-empty Locks -> ineligible" gate doesn't
  // re-disqualify the VO.
  void clearLocks() {
    assert(isVirtual());
    Locks.clear();
  }
};

// ===========================================================================
// 1.4  PEABlockState
// ===========================================================================

class PEABlockState {
private:
  // ObjectStates is shared via std::shared_ptr and is copy-on-write: a mutator
  // calls getArrayForModification(), which deep-copies the array when it is
  // currently shared. Sharing is read directly from ObjectStates.use_count()
  // (== 1 ⇒ sole owner, mutate in place; > 1 ⇒ clone first). Because every
  // shared copy/move/destroy of a PEABlockState is just the shared_ptr doing
  // the same to its control block, use_count() is an exact sharer count and no
  // parallel manual refcount is needed. copy/move/assign/dtor are therefore
  // implicitly generated (rule of zero).
  std::shared_ptr<SmallVector<std::optional<ObjectState>, 8>> ObjectStates;

  bool Dead = false;

public:
  PEABlockState();

  void addObject(ObjectID ID, ObjectState State);

  bool hasObjectState(ObjectID ID) const;
  const ObjectState &getObjectState(ObjectID ID) const;
  ObjectState &getObjectStateForModification(ObjectID ID);
  const ObjectState *getObjectStateOptional(ObjectID ID) const;

  unsigned getStateCount() const {
    return ObjectStates ? static_cast<unsigned>(ObjectStates->size()) : 0u;
  }

  std::optional<ObjectID> resolveVirtualRef(Value *V,
                                            const AliasMap &Aliases) const;

  bool isDead() const { return Dead; }
  void markDead() { Dead = true; }

private:
  SmallVector<std::optional<ObjectState>, 8> *getArrayForModification();
};

// ===========================================================================
// 1.5  AliasMap
// ===========================================================================

class AliasMap {
  DenseMap<Value *, ObjectID> VirtualAliases;
  DenseMap<Value *, Value *> ScalarAliases;
  DenseSet<Instruction *> HasVirtualInputs;

public:
  void addVirtualAlias(Value *V, ObjectID ID);
  void addScalarAlias(Value *V, Value *Replacement);
  void resetAlias(Value *V);

  std::optional<ObjectID> getVirtualAlias(Value *V) const;
  Value *getScalarAlias(Value *V) const;

  bool hasVirtualInputs(Instruction *I) const {
    return HasVirtualInputs.count(I);
  }

  // Read-only view over the virtual alias map. Used to iterate every Value*
  // that represents a given ObjectID for the per-object eligibility cleanup
  // pass.
  const DenseMap<Value *, ObjectID> &virtualAliasesView() const {
    return VirtualAliases;
  }

  void clear();

  AliasMap snapshot() const;
  void restore(const AliasMap &S);
};

// ===========================================================================
// 1.6  Effects
//
// Graal's PEA models an IR mutation as an `Effect` (EffectList.java):
//   - abstract base `Effect` with `isVisible()`, `isCfgKill()`,
//     `apply(graph, obsoleteNodes)`, `format()`;
//   - concrete effects are anonymous subclasses, one per mutation kind,
//     categorized by `isCfgKill()` (drives the two-pass apply) and
//     `isVisible()` (logging only);
//   - stored per-block in a `GraphEffectList` (EffectList subclass) and
//     applied in two passes — non-cfgKill first, cfgKill second — driven
//     entirely by `isCfgKill()`.
//
// Jeandle mirrors this shape with two IR-form-induced adaptations:
//   1. LLVM's Analysis/Transform split forbids mutating IR during analysis,
//      so Jeandle's effects are serializable DATA RECORDS (named subclasses
//      with fields), not anonymous closures capturing graph nodes. They are
//      produced by the analysis pass, marshalled in `PEAResult`, and consumed
//      by a separate transform pass.
//   2. `apply(TransformContext&)` is the Jeandle form of Graal's
//      `apply(graph, obsoleteNodes)` — `TransformContext` bundles the
//      Function and the shared per-apply maps (defined in
//      PartialEscapeTransform.cpp).
// ===========================================================================

// Jeandle adaptation of Graal's `EffectList.Effect`. Abstract base for every
// recorded IR mutation.
class Effect {
public:
  enum class Kind : uint8_t {
    ReplaceLoad,
    ReplaceCall,
    EliminateStore,
    EliminateAllocation,
    Materialize,
    CreatePHI,
    RewritePhiIncoming,
  };

  // --- common fields (read by commit/dropEffectsFor/the transform) ---
  BasicBlock *Block = nullptr;
  // IR-FORM DIVERGENCE from Graal: Graal applies effects in pure list-order
  // (its per-block EffectList is append-only; loop headers use
  // insertAll(...,0)). Jeandle marshals effects into a per-block map and
  // re-sorts at apply time, so it needs an explicit ordering key. The
  // deferred-CreatePHI trick (emit at SeqNo=0, reassign a fresh nextSeqNo()
  // at drain) is load-bearing for the self-loop back-edge ordering invariant
  // (CreatePHI must sort strictly after per-pred Materialize in the same
  // block).
  // TODO(list-order): adopting Graal's pure list-order would require reworking
  // merge/loop emission; SeqNo + the deferred-CreatePHI trick covers ordering
  // meanwhile.
  uint32_t SeqNo = 0;
  ObjectID ObjID = InvalidObjectID;

  virtual ~Effect() = default;

  // Graal isCfgKill(): drives the two-pass apply (non-cfgKill first across all
  // blocks, cfgKill second). True ONLY for EliminateAllocation (maps to Graal's
  // deleteNode(WithExceptionNode) / killIfBranch / replaceWithSink). NOTE this
  // is NOT the same as
  // "structurally rewrites the CFG": Materialize (splitBasicBlock) and
  // CreatePHI (PHI insertion) DO rewrite control flow but run in the first
  // pass — they are not cfg-kill in Graal's apply-ordering sense.
  virtual bool isCfgKill() const { return false; }
  // Graal isVisible(): logging only. Jeandle has no deopt-internal effects
  // (Graal's addVirtualMapping/updateVirtualMapping/addLog), so all visible.
  virtual bool isVisible() const { return true; }

  virtual Kind getKind() const = 0;

  // The IR instruction this effect rewrites/erases, or null for effects that
  // have no single target (CreatePHI). Read generically by the analyzer-side
  // scans (processBlock tail, processBlockPhis identity check, commit).
  virtual Instruction *getTarget() const { return nullptr; }

  // Graal apply(graph, obsoleteNodes). Mutates Ctx.F's IR. Defined in
  // PartialEscapeTransform.cpp.
  virtual void apply(TransformContext &Ctx) = 0;

  // Graal format()/toString(). Drives the -jeandle-trace-pea per-effect line.
  // Non-pure: the kind name / ObjID / Block / Target are all reachable via the
  // base API, so one shared definition suffices. Defined in PartialEscape.cpp.
  virtual void dump(raw_ostream &OS) const;

  // Deep copy for the loop-fixpoint snapshot (takeLoopSnapshot/
  // restoreLoopSnapshot read the snapshot across multiple iterations, so they
  // need a copy; EffectList is move-only). IR-form divergence: Graal
  // backtracks by truncating its EffectList (size=0); Jeandle's per-block map
  // snapshot needs a real deep copy.
  virtual std::unique_ptr<Effect> clone() const = 0;
};

// Fold a load from a virtual object's field to the tracked scalar value (or a
// synthesized coercion / default). Non-cfgKill. Graal analog: replaceAtUsages.
class ReplaceLoadEffect : public Effect {
public:
  Instruction *Target = nullptr;
  Value *Replacement = nullptr;

  Kind getKind() const override { return Kind::ReplaceLoad; }
  static bool classof(const Effect *E) { return E->getKind() == Kind::ReplaceLoad; }
  Instruction *getTarget() const override { return Target; }
  void apply(TransformContext &Ctx) override;
  std::unique_ptr<Effect> clone() const override {
    return std::make_unique<ReplaceLoadEffect>(*this);
  }
};

// Fold a `jeandle.*` JavaOp against a virtual receiver to a constant / delete.
// Non-cfgKill. Graal analog: replaceAtUsages / deleteNode.
class ReplaceCallEffect : public Effect {
public:
  Instruction *Target = nullptr;
  Value *Replacement = nullptr;

  Kind getKind() const override { return Kind::ReplaceCall; }
  static bool classof(const Effect *E) { return E->getKind() == Kind::ReplaceCall; }
  Instruction *getTarget() const override { return Target; }
  void apply(TransformContext &Ctx) override;
  std::unique_ptr<Effect> clone() const override {
    return std::make_unique<ReplaceCallEffect>(*this);
  }
};

// Remove a store into a virtual object's field (value tracked in FieldStates).
// Non-cfgKill. Graal analog: deleteNode (FixedWithNextNode).
class EliminateStoreEffect : public Effect {
public:
  Instruction *Target = nullptr;

  Kind getKind() const override { return Kind::EliminateStore; }
  static bool classof(const Effect *E) { return E->getKind() == Kind::EliminateStore; }
  Instruction *getTarget() const override { return Target; }
  void apply(TransformContext &Ctx) override;
  std::unique_ptr<Effect> clone() const override {
    return std::make_unique<EliminateStoreEffect>(*this);
  }
};

// Rewrite the original allocation invoke into an unconditional branch (dropping
// the unwind edge) or erase a call alloc. cfgKill: applied in Pass 2.
// Graal analog: deleteNode (WithExceptionNode) / killIfBranch.
class EliminateAllocationEffect : public Effect {
public:
  Instruction *Target = nullptr;

  Kind getKind() const override { return Kind::EliminateAllocation; }
  static bool classof(const Effect *E) {
    return E->getKind() == Kind::EliminateAllocation;
  }
  bool isCfgKill() const override { return true; }
  Instruction *getTarget() const override { return Target; }
  void apply(TransformContext &Ctx) override;
  std::unique_ptr<Effect> clone() const override {
    return std::make_unique<EliminateAllocationEffect>(*this);
  }
};

// Emit a real allocation invoke before an escape point, replay tracked field
// stores, and re-emit surviving monitorenters. Non-cfgKill (Pass 1).
// Graal analog: the one `Effect("materializeBefore")` appended by
// PartialEscapeBlockState.materializeBefore.
class MaterializeEffect : public Effect {
public:
  // Per-offset snapshot of a virtual object's field values at a
  // materialization point.
  struct FieldEntry {
    int64_t Offset;
    FieldValue Value;
  };

  // WeakTrackingVH so erasing the insertion-point instruction auto-nulls the
  // handle; applyMaterialize recomputes a fresh IP in that case.
  WeakTrackingVH InsertBefore;
  Instruction *Target = nullptr; // the original allocation (OrigAlloc)
  SmallVector<FieldEntry, 8> FieldEntries;
  // Surviving (unbalanced) monitorenters to re-emit at the materialize point
  // (Graal: synthetic MonitorEnterNodes at the CommitAllocationNode), sorted
  // ascending by BytecodeDepth.
  SmallVector<MaterializedLock, 2> Locks;
  Instruction *DeoptBundleSource = nullptr;
  bool IsPerPred = false;
  Value *PerPredPlaceholder = nullptr;
  // The target merge block this per-pred materialize is destined for (the merge
  // whose `MergeProcessor::BB` was in scope when this effect was emitted), or
  // null for the Case-A / global path (mat placed at the pred's terminator end,
  // whose NewInv dominates all successors — Graal-aligned, benign shared flip).
  // IR-form divergence: Graal places the CommitAllocationNode before the pred's
  // control split so its materialized value dominates every successor; Jeandle
  // emits a real invoke with a single normal dest on one critical edge, so the
  // NewInv dominates only that edge. The transform's critical-edge pre-pass keys
  // `PHRename`/`BlockRename` by (PH, TargetMergeBB) so two per-pred mats from the
  // same PH to different merges do not collide, and `MatPerBlock`/`BlockRename`
  // chain walks route each merge's CreatePHI incoming through its own edge.
  BasicBlock *TargetMergeBB = nullptr;

  Kind getKind() const override { return Kind::Materialize; }
  static bool classof(const Effect *E) { return E->getKind() == Kind::Materialize; }
  Instruction *getTarget() const override { return Target; }
  void apply(TransformContext &Ctx) override;
  std::unique_ptr<Effect> clone() const override {
    return std::make_unique<MaterializeEffect>(*this);
  }
  // Critical-edge pre-pass re-aims a per-pred Materialize onto the split edge
  // block (IR-form divergence: Graal routes per-pred materialize to the pred's
  // block directly during analysis; Jeandle re-aims at transform time).
  void setBlock(BasicBlock *BB) { Block = BB; }
  // Defined out-of-line: assigning to the WeakTrackingVH needs Instruction to
  // be a complete type (Instruction* → Value* derived-to-base), which it is
  // not in this header.
  void setInsertBefore(Instruction *I);
};

// Insert an analyzer-built unparented PHINode at a merge block and wire its
// incomings. Non-cfgKill (Pass 1). Graal analog: addFloatingNode + setPhiInput
// (initializePhiInput).
class CreatePHIEffect : public Effect {
public:
  Type *PHIType = nullptr;
  SmallVector<Value *, 4> PHIIncomingValues;
  SmallVector<BasicBlock *, 4> PHIIncomingBlocks;
  PHINode *PhiInst = nullptr;
  bool RAUWOrigToPHI = false;

  Kind getKind() const override { return Kind::CreatePHI; }
  static bool classof(const Effect *E) { return E->getKind() == Kind::CreatePHI; }
  void apply(TransformContext &Ctx) override;
  std::unique_ptr<Effect> clone() const override {
    return std::make_unique<CreatePHIEffect>(*this);
  }
};

// Re-derive a loop/merge-carried DERIVED pointer (GEP/bitcast of a virtual
// object) at the per-predecessor materialization point and rewire the carrying
// PHI's incoming to it. This is Jeandle's extension of Graal
// getAliasAndResolve + setPhiInput (PartialEscapeClosure.java:1575-1584 /
// :1511) to LLVM derived pointers, which have no Graal analog: Graal only ever
// carries object-identity aliases, so its merge re-derivation hands back the
// materialized object directly; LLVM can carry a field address (a GEP with a
// byte offset), so the re-derivation must replay that offset over the freshly-
// materialized base. Non-cfgKill (Pass 1); must sort strictly after the per-pred
// Materialize in the same block bucket so NewInv is recorded before apply.
class RewritePhiIncomingEffect : public Effect {
public:
  PHINode *Phi = nullptr;              // the existing carrying PHI (already in IR)
  BasicBlock *Pred = nullptr;          // analyzer-recorded predecessor (e.g. latch)
  Value *PerPredPlaceholder = nullptr; // resolves to this pred's NewInv at apply
  int64_t ByteOffset = 0;   // constant byte offset of the carry from OrigAlloc
                            // (0 = bitcast/identity -> reuse NewInv directly)
  // Always null for this effect: RewritePhiIncoming is created only by the
  // Case-A path (processBlockPhis, SkipGlobalRAUW=false), whose mat is placed at
  // the pred's terminator end (NewInv dominates all successors, shared flip). The
  // transform's `BlockRename` chain walk keys by (LivePred, TargetMergeBB); null
  // here routes through the single shared MatCont (no critical-edge split).
  BasicBlock *TargetMergeBB = nullptr;

  Kind getKind() const override { return Kind::RewritePhiIncoming; }
  static bool classof(const Effect *E) {
    return E->getKind() == Kind::RewritePhiIncoming;
  }
  void apply(TransformContext &Ctx) override;
  std::unique_ptr<Effect> clone() const override {
    return std::make_unique<RewritePhiIncomingEffect>(*this);
  }
};

// Jeandle adaptation of Graal's `EffectList` / `GraphEffectList`. Append-only
// list of `unique_ptr<Effect>` (mirrors Graal's `Effect[]` of references).
// Move-only: the loop-fixpoint snapshot needs a copy, taken via `clone()`.
class EffectList {
  SmallVector<std::unique_ptr<Effect>, 16> Effects;

public:
  EffectList() = default;
  EffectList(EffectList &&) = default;
  EffectList &operator=(EffectList &&) = default;
  EffectList(const EffectList &) = delete;
  EffectList &operator=(const EffectList &) = delete;

  // Graal add(): append one effect at the tail.
  void add(std::unique_ptr<Effect> E) { Effects.push_back(std::move(E)); }
  // Graal addAll(): append all of Other at the tail (move-merge, clear Other).
  void addAll(EffectList &Other) {
    for (auto &E : Other.Effects)
      Effects.push_back(std::move(E));
    Other.Effects.clear();
  }
  // Graal insertAll(): insert all of Other at Pos, preserving order. Kept for
  // loop-header precedence should pure list-order be adopted (see TODO(list-
  // order) on Effect::SeqNo); currently unused.
  void insertAll(EffectList &Other, size_t Pos) {
    assert(Pos <= Effects.size());
    for (auto &E : Other.Effects)
      Effects.insert(Effects.begin() + Pos++, std::move(E));
    Other.Effects.clear();
  }
  // Graal clear(): size=0, retain capacity for backtracking reuse.
  void clear() { Effects.clear(); }
  bool empty() const { return Effects.empty(); }
  size_t size() const { return Effects.size(); }
  Effect &operator[](size_t I) { return *Effects[I]; }
  const Effect &operator[](size_t I) const { return *Effects[I]; }

  // IR-form extension: remove and return ownership of element I (critical-edge
  // bucket move in the transform pre-pass).
  std::unique_ptr<Effect> spliceOut(size_t I) {
    auto E = std::move(Effects[I]);
    Effects.erase(Effects.begin() + I);
    return E;
  }
  // IR-form extension: drop every effect matching Pred (dropEffectsFor).
  void eraseIf(function_ref<bool(const Effect &)> Pred) {
    SmallVector<std::unique_ptr<Effect>, 16> Kept;
    for (auto &E : Effects)
      if (!Pred(*E))
        Kept.push_back(std::move(E));
    Effects = std::move(Kept);
  }
  // Deep copy for the loop-fixpoint snapshot.
  EffectList clone() const {
    EffectList C;
    for (const auto &E : Effects)
      C.Effects.push_back(E->clone());
    return C;
  }

  // Apply every effect where `isCfgKill()==CfgKills`, in SeqNo order (Jeandle's
  // substitute for Graal list-order — see Effect::SeqNo). Defined in
  // PartialEscapeTransform.cpp.
  void apply(TransformContext &Ctx, bool CfgKills);

  // Range-for yielding Effect& / const Effect& (the analyzer-side scans iterate
  // by reference; ownership stays with the list).
  struct Iterator {
    SmallVectorImpl<std::unique_ptr<Effect>>::iterator It;
    Effect &operator*() const { return **It; }
    Iterator &operator++() { ++It; return *this; }
    bool operator==(const Iterator &O) const { return It == O.It; }
    bool operator!=(const Iterator &O) const { return It != O.It; }
  };
  struct ConstIterator {
    SmallVectorImpl<std::unique_ptr<Effect>>::const_iterator It;
    const Effect &operator*() const { return **It; }
    ConstIterator &operator++() { ++It; return *this; }
    bool operator==(const ConstIterator &O) const { return It == O.It; }
    bool operator!=(const ConstIterator &O) const { return It != O.It; }
  };
  Iterator begin() { return {Effects.begin()}; }
  Iterator end() { return {Effects.end()}; }
  ConstIterator begin() const { return {Effects.begin()}; }
  ConstIterator end() const { return {Effects.end()}; }
};

// ===========================================================================
// 1.7  PEAResult
// ===========================================================================

class PEAResult {
public:
  SmallVector<std::unique_ptr<VirtualObject>, 8> VirtualObjects;

  enum class EscapeKind : uint8_t {
    NeverEscapes,
    PartiallyEscapes,
    AlwaysEscapes
  };
  DenseMap<ObjectID, EscapeKind> EscapeClassification;

  DenseMap<BasicBlock *, EffectList> BlockEffects;

  // Locks to re-emit at each escape point (the shared InsertBefore of a
  // materialize / cascade group), globally sorted ascending by BytecodeDepth,
  // and the highest MaterializeEffect SeqNo at each escape point. Graal
  // flattens every lock materialized at one point into a single
  // CommitAllocationNode and lowers them globally depth-sorted with a strict-
  // increase guarantee (DefaultJavaLoweringProvider). Jeandle's per-VO
  // MaterializeEffect model otherwise re-emits per-effect, which mis-orders
  // re-entrant interleaved cascades (e.g. [a@0,b@1,a@2,c@3] re-emitting as
  // 0,2,1,3). The transform emits each escape point's merged list once, from
  // its highest-SeqNo effect (by then every sibling's NewInv exists), each
  // lock's receiver resolved via NewInvOf[SourceEffect] (with NewAllocFor
  // / NewInv fallbacks). Both live-path and per-pred cascades are merged:
  // per-pred effects sharing a (PH,S) edge are re-aimed onto one split-edge
  // block by the critical-edge pre-pass (or already share the single-succ
  // pred's terminator), so they group under one InsertBefore identically to a
  // live-path cascade. Populated by computeEscapePointLocks(), called once
  // before the transform applies Pass 1.
  DenseMap<Instruction *, SmallVector<MergedLock, 4>> EscapePointLocks;
  DenseMap<Instruction *, uint32_t> MaxSeqForEscapePoint;
  void computeEscapePointLocks();

  int VirtualizationDelta = 0;
  int AllocationDelta = 0;
  uint32_t NextSeqNo = 0;

  // PHI nodes created (unparented) by the analyzer for CreatePHI effects.
  // Stored as WeakTrackingVH so we can safely check for nullness in the
  // destructor — once a PHI is inserted into a BasicBlock and later RAUW'd
  // or erased by the transform's dead-code sweep, the handle auto-nulls and
  // the destructor skips it. Any PHI that's still alive AND still unparented
  // (i.e. the analyzer ran without a paired transform) is `delete`d.
  SmallVector<WeakTrackingVH, 4> OwnedPhis;

  // Non-PHI Instructions created (unparented) by the analyzer for
  // ReplaceLoad effects (e.g. a `bitcast` synthesized by `coerceToType` to
  // bridge a same-bit-width type mismatch between a stored Scalar and the
  // load's type). Same ownership rules as OwnedPhis: the transform splices
  // the instruction in before its Target; the destructor deletes any
  // handle still non-null and unparented at teardown. WeakTrackingVH guards
  // against a UAF if the inserted coercion is later erased by the dead-code
  // sweep.
  SmallVector<WeakTrackingVH, 4> OwnedInsts;

  // PHI nodes synthesized by mergeStates at ANY in-loop merge block (a loop
  // header OR a non-header block inside a loop), stored separately from
  // OwnedPhis because they must survive rollback within the loop-fixpoint
  // iteration. Each iteration re-runs mergeStates(Header) and would otherwise
  // allocate a fresh PHI per (BB, ID, offset), producing fresh Value* pointers
  // in FieldStates so the convergence check could never compare equal. The
  // analyzer's LoopFieldPhiCache (keyed on (BB, ID, offset)) returns the same
  // PHI across iterations; only the per-iteration CreatePHI Effect is rebuilt.
  // Non-header in-loop merges are included because restoreLoopSnapshot
  // preserves BlockExits[BB] for every loop block across iterations, so any
  // Value* reachable from a preserved BlockExits[BB] must outlive rollback —
  // were such a PHI to land in OwnedPhis (which IS truncated), the preserved
  // BlockExits[BB] would reference a deleted PHI. Lifecycle is identical to
  // OwnedPhis.
  SmallVector<WeakTrackingVH, 4> OwnedLoopFieldPhis;

  // Per-pred materialization placeholder Value*s synthesized by the analyzer
  // (one per (predecessor-block, ObjectID) materialization). Distinct
  // unparented instructions that stand in for the per-pred NewInv the
  // transform creates; resolved away (never inserted into IR) at apply time
  // via MatPerBlock / NewAllocFor. Same ownership rules as OwnedPhis: the
  // transform never inserts them, so the destructor deletes any handle still
  // non-null and unparented at teardown. WeakTrackingVH guards against a UAF
  // if a placeholder is ever erased. See Effect::PerPredPlaceholder.
  SmallVector<WeakTrackingVH, 4> OwnedMatPlaceholders;

  // Membership view of OwnedMatPlaceholders for O(1) "is this Value a per-pred
  // placeholder?" queries from the transform (CreatePHIEffect must distinguish
  // an unresolved per-pred placeholder — to be resolved/fallen-back — from a
  // legitimately-unparented loop field-PHI incoming, which must be left as-is).
  // Populated alongside OwnedMatPlaceholders in getOrCreatePerPredMatPlaceholder.
  DenseSet<Value *> PerPredMatPlaceholders;

  // Parented LLVM PHIs the transform should RAUW to poison + erase after
  // the main Pass-2 EliminateAllocation sweep. These are Case-B aliases on
  // a virtual that ended up NeverEscapes: the PHI was registered as an
  // alias so downstream loads/stores fold through the alias map, but the
  // PHI itself is now dead (every incoming was the VO's OrigAlloc, which
  // Pass-2 RAUW'd to poison). WeakTrackingVH so a parallel delete path
  // (e.g. unreachable-block pruning) leaves a null handle the transform
  // safely skips.
  SmallVector<WeakTrackingVH, 4> CaseBAliasedPhisToErase;

  PEAResult() = default;
  PEAResult(const PEAResult &) = delete;
  PEAResult &operator=(const PEAResult &) = delete;
  PEAResult(PEAResult &&) = default;
  PEAResult &operator=(PEAResult &&) = default;
  ~PEAResult();

  ObjectID createVirtualObject(std::unique_ptr<VirtualObject> VO);

  uint32_t nextSeqNo() { return NextSeqNo++; }

  void addBlockEffect(std::unique_ptr<Effect> E);

  bool hasOptimizationOpportunity() const;
};

} // namespace jeandle
} // namespace llvm

#endif // LLVM_ANALYSIS_JEANDLE_PARTIALESCAPE_H
