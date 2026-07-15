//===- PartialEscape.cpp - PEA shared data structure impls ----------------===//
//
// Copyright (c) 2026, the Jeandle-LLVM Authors. All Rights Reserved.
//
// Part of the Jeandle-LLVM project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implementation of non-trivial method bodies for the PEA data structures
// declared in llvm/include/llvm/Analysis/Jeandle/PartialEscape.h.

#include "llvm/Analysis/Jeandle/PartialEscape.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/Jeandle/PartialEscapeUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Jeandle/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>


using namespace llvm;
using namespace llvm::jeandle;

// Per-effect dbgs() trace. When enabled, every Effect routed through
// PEAResult::addBlockEffect emits a one-line summary on dbgs().
// Off by default; turn on with -jeandle-trace-pea. Defined here (rather
// than in PartialEscapeAnalysis.cpp) so the central emission site can
// read it without a separate accessor.
static llvm::cl::opt<bool> JeandleTracePEA(
    "jeandle-trace-pea", llvm::cl::init(false), llvm::cl::Hidden,
    llvm::cl::desc("PEA: emit a one-line dbgs() trace on every major "
                   "effect emission (EliminateAllocation / Materialize / "
                   "ReplaceCall / ReplaceLoad / EliminateStore / "
                   "CreatePHI)."));

// ===========================================================================
// VirtualObject
// ===========================================================================

int VirtualObject::getOrCreateFieldIndex(int64_t Offset, Type *Ty,
                                         const DataLayout &DL) {
  assert(Ty && "field type must be non-null");
  uint8_t ByteSize = 0;
  bool IsReference = false;
  if (Ty->isPointerTy()) {
    // Reference field: ptr addrspace(1). Its byte size is the target's Java
    // heap pointer size (DL.getPointerSize(JavaHeapAddrSpace)) — 8 on the
    // current 64-bit target, but derived from the DataLayout so a 32-bit or
    // compressed-oop heap model stays correct rather than hardcoding 8.
    assert(Ty->getPointerAddressSpace() ==
               jeandle::AddrSpace::JavaHeapAddrSpace &&
           "reference field must be in JavaHeapAddrSpace");
    ByteSize = static_cast<uint8_t>(
        DL.getPointerSize(jeandle::AddrSpace::JavaHeapAddrSpace));
    IsReference = true;
  } else {
    unsigned Bits = Ty->getPrimitiveSizeInBits();
    if (Bits == 0)
      return -1; // unknown-size type (e.g. vector/struct) — conservative escape
    ByteSize = static_cast<uint8_t>((Bits + 7) / 8);
  }

  auto It = std::lower_bound(
      Fields.begin(), Fields.end(), Offset,
      [](const FieldDesc &F, int64_t Off) { return F.Offset < Off; });

  // Exact match? Reuse if the existing FieldDesc agrees on size and
  // reference-ness.
  if (It != Fields.end() && It->Offset == Offset) {
    if (It->ByteSize != ByteSize || It->IsReference != IsReference)
      return -1;
    return static_cast<int>(It - Fields.begin());
  }

  // Overlap check against the candidate neighbor on the left.
  if (It != Fields.begin()) {
    auto Prev = std::prev(It);
    if (Prev->overlaps(Offset, ByteSize))
      return -1;
  }
  // Overlap check against the candidate neighbor on the right.
  if (It != Fields.end() && It->overlaps(Offset, ByteSize))
    return -1;

  FieldDesc New{Offset, Ty, ByteSize, IsReference};
  auto NewIt = Fields.insert(It, New);
  return static_cast<int>(NewIt - Fields.begin());
}

// Strip identity-preserving wrappers (freeze, bitcast, zext, sext) from an
// index Value. Used by matchArrayElementGEP to canonicalize the index so
// callers can pattern-match a ConstantInt or a Value identity.
static Value *peelIndexWrappers(Value *V) {
  for (unsigned Depth = 0; Depth < 8; ++Depth) {
    if (!V)
      return V;
    if (auto *FI = dyn_cast<FreezeInst>(V)) {
      V = FI->getOperand(0);
      continue;
    }
    if (auto *BC = dyn_cast<BitCastOperator>(V)) {
      // BitCast on an integer (the index slot) is identity-preserving.
      V = BC->getOperand(0);
      continue;
    }
    if (auto *ZI = dyn_cast<ZExtInst>(V)) {
      V = ZI->getOperand(0);
      continue;
    }
    if (auto *SI = dyn_cast<SExtInst>(V)) {
      V = SI->getOperand(0);
      continue;
    }
    // Handle the ConstantExpr forms in case they appear here.
    if (auto *CE = dyn_cast<ConstantExpr>(V)) {
      if (CE->getOpcode() == Instruction::ZExt ||
          CE->getOpcode() == Instruction::SExt ||
          CE->getOpcode() == Instruction::BitCast) {
        V = CE->getOperand(0);
        continue;
      }
    }
    break;
  }
  return V;
}

// Match the symbolic byte-offset pattern emitted by the abstract
// interpreter for indexed accesses on an i8-typed GEP: byteOff equals
// `ArrayBaseOffset + idx * scale`, where the scaling is expressed as a
// shift by log2(scale) or a multiply by scale. Returns the (peeled) index
// Value on success, or nullptr on no match.
static Value *matchAddBasePlusScaledIndex(Value *ByteOff,
                                          int64_t ExpectedBaseOffset,
                                          uint64_t Scale) {
  using namespace llvm::PatternMatch;
  if (!ByteOff || !ByteOff->getType()->isIntegerTy())
    return nullptr;
  if (Scale == 0)
    return nullptr;

  // Recognise the constant byte-offset operand of an `add C, X` pair.
  Value *ScaledIdx = nullptr;
  ConstantInt *BaseCI = nullptr;
  if (match(ByteOff, m_Add(m_ConstantInt(BaseCI), m_Value(ScaledIdx)))) {
    // matched
  } else if (match(ByteOff, m_Add(m_Value(ScaledIdx), m_ConstantInt(BaseCI)))) {
    // matched
  } else {
    return nullptr;
  }
  if (!BaseCI || BaseCI->getSExtValue() != ExpectedBaseOffset)
    return nullptr;

  // Now ScaledIdx must equal idx * Scale (or idx if Scale == 1).
  if (Scale == 1)
    return peelIndexWrappers(ScaledIdx);

  Value *Idx = nullptr;
  if (isPowerOf2_64(Scale)) {
    unsigned LogScale = Log2_64(Scale);
    ConstantInt *ShAmt = nullptr;
    if (match(ScaledIdx, m_Shl(m_Value(Idx), m_ConstantInt(ShAmt))) && ShAmt &&
        ShAmt->getZExtValue() == LogScale)
      return peelIndexWrappers(Idx);
  }
  ConstantInt *MulCI = nullptr;
  if (match(ScaledIdx, m_Mul(m_Value(Idx), m_ConstantInt(MulCI))) && MulCI &&
      (uint64_t)MulCI->getSExtValue() == Scale)
    return peelIndexWrappers(Idx);
  if (match(ScaledIdx, m_Mul(m_ConstantInt(MulCI), m_Value(Idx))) && MulCI &&
      (uint64_t)MulCI->getSExtValue() == Scale)
    return peelIndexWrappers(Idx);
  return nullptr;
}

std::optional<VirtualObject::ArrayElementGEPMatch>
VirtualObject::matchArrayElementGEP(GetElementPtrInst *GEP,
                                    const DataLayout &DL) const {
  if (!GEP)
    return std::nullopt;
  if (Kind != Array || !ArrayElementType || ArrayIndexScale == 0)
    return std::nullopt;

  auto *GEPOp = cast<GEPOperator>(GEP);
  const unsigned AS = GEPOp->getPointerAddressSpace();
  const unsigned PtrBits = DL.getPointerSizeInBits(AS);

  // Pattern A: typed GEP with sourceElementType == ArrayElementType. The
  // typed GEP's pointer operand must reach the alloc base with a constant
  // accumulated byte offset equal to ArrayBaseOffset (Jeandle emits this
  // as a preceding i8 + base-offset GEP).
  if (GEPOp->getSourceElementType() == ArrayElementType &&
      GEPOp->getNumIndices() == 1) {
    int64_t BaseOff = 0;
    bool NonConst = false;
    (void)jeandle::pea::stripPointerCastsAndOffsets(GEPOp->getPointerOperand(),
                                                    DL, &BaseOff, &NonConst);
    if (!NonConst && BaseOff == static_cast<int64_t>(ArrayBaseOffset))
      return ArrayElementGEPMatch{peelIndexWrappers(GEPOp->getOperand(1)),
                                  ArrayElementType};
  }

  // Pattern B: i8 GEP with a single byte-offset index. The byte offset may
  // be (a) a constant equal to ArrayBaseOffset + cidx * Scale, or (b) an
  // `add ArrayBaseOffset, (shl|mul) idx, Scale`. Pattern (a) is handled
  // naturally by resolveFieldOffset (constant byte offset), so the
  // matcher's main job here is pattern (b) for symbolic indices, but we
  // still recognise (a) and surface a ConstantInt-wrapped element index.
  if (GEPOp->getSourceElementType()->isIntegerTy(8) &&
      GEPOp->getNumIndices() == 1) {
    int64_t BaseOff = 0;
    bool NonConst = false;
    (void)jeandle::pea::stripPointerCastsAndOffsets(GEPOp->getPointerOperand(),
                                                    DL, &BaseOff, &NonConst);
    if (NonConst || BaseOff != 0)
      return std::nullopt;

    Value *ByteOff = GEPOp->getOperand(1);
    if (auto *CI = dyn_cast<ConstantInt>(ByteOff)) {
      // Constant byte offset. Recover the element index if (off - base) is
      // a multiple of Scale and lies in [0, ArrayLength).
      int64_t Off = CI->getValue().sextOrTrunc(PtrBits).getSExtValue();
      int64_t Adj = Off - static_cast<int64_t>(ArrayBaseOffset);
      if (Adj < 0 || (Adj % static_cast<int64_t>(ArrayIndexScale)) != 0)
        return std::nullopt;
      int64_t Cidx = Adj / static_cast<int64_t>(ArrayIndexScale);
      if (Cidx < 0 || static_cast<uint64_t>(Cidx) >= ArrayLength)
        return std::nullopt;
      Constant *CIdx = ConstantInt::get(CI->getType(), Cidx, /*isSigned=*/true);
      return ArrayElementGEPMatch{CIdx, ArrayElementType};
    }

    if (Value *Idx = matchAddBasePlusScaledIndex(
            ByteOff, static_cast<int64_t>(ArrayBaseOffset), ArrayIndexScale))
      return ArrayElementGEPMatch{Idx, ArrayElementType};
  }

  return std::nullopt;
}

void VirtualObject::copyStructuralFieldsFrom(const VirtualObject &O) {
  Klass = O.Klass;
  SizeInBytes = O.SizeInBytes;
  ArrayLength = O.ArrayLength;
  ArrayElementType = O.ArrayElementType;
  ArrayIndexScale = O.ArrayIndexScale;
  ArrayBaseOffset = O.ArrayBaseOffset;
  ArrayLengthLimit = O.ArrayLengthLimit;
  Fields = O.Fields;
}

std::unique_ptr<VirtualObject> VirtualObject::duplicate() const {
  // The clone is detached: ID is set to InvalidObjectID and the caller is
  // expected to register it via PEAResult::createVirtualObject to obtain a
  // fresh ID.
  auto Clone =
      std::make_unique<VirtualObject>(InvalidObjectID, Kind, AllocationCall);
  Clone->copyStructuralFieldsFrom(*this);
  // Synthetic-state fields are NOT copied — duplicate() is shared by the
  // generic VirtualObject clone path AND the Case C synthesis path; the
  // latter sets IsSynthetic/SyntheticSourceIDs/SyntheticPhi explicitly after
  // calling duplicate.
  Clone->IsSynthetic = false;
  Clone->SyntheticPhi = nullptr;
  return Clone;
}

// ===========================================================================
// FieldValue
// ===========================================================================

FieldValue FieldValue::scalar(Value *V) {
  assert(V && "scalar value must be non-null");
  FieldValue F;
  F.T = Scalar;
  F.U.V = V;
  F.DeclaredType = V->getType();
  return F;
}

FieldValue FieldValue::virtualRef(ObjectID ID, Type *RefTy) {
  assert(ID != InvalidObjectID);
  assert(RefTy && RefTy->isPointerTy() &&
         RefTy->getPointerAddressSpace() ==
             jeandle::AddrSpace::JavaHeapAddrSpace &&
         "virtualRef DeclaredType must be ptr addrspace(1)");
  FieldValue F;
  F.T = VirtualRef;
  F.U.Ref = ID;
  F.DeclaredType = RefTy;
  return F;
}

FieldValue FieldValue::materializedRef(Value *Ptr) {
  assert(Ptr && Ptr->getType()->isPointerTy());
  FieldValue F;
  F.T = MaterializedRef;
  F.U.V = Ptr;
  F.DeclaredType = Ptr->getType();
  return F;
}

Constant *FieldValue::defaultFor(Type *FieldType) {
  assert(FieldType);
  if (FieldType->isPointerTy()) {
    assert(FieldType->getPointerAddressSpace() ==
               jeandle::AddrSpace::JavaHeapAddrSpace &&
           "reference default must be in JavaHeapAddrSpace");
    return ConstantPointerNull::get(cast<PointerType>(FieldType));
  }
  return Constant::getNullValue(FieldType);
}

bool FieldValue::shallowEquals(const FieldValue &O) const {
  if (T != O.T)
    return false;
  switch (T) {
  case Unknown:
    return true;
  case Scalar:
  case MaterializedRef:
    return U.V == O.U.V;
  case VirtualRef:
    return U.Ref == O.U.Ref;
  }
  return false;
}

// ===========================================================================
// ObjectState
// ===========================================================================

// ===========================================================================
// PEABlockState
// ===========================================================================

// copy/move/assign/dtor are implicitly generated (rule of zero): they just
// copy/move/destroy the ObjectStates shared_ptr, which keeps use_count exact,
// and the plain `Dead` flag. No manual refcount is maintained.

PEABlockState::PEABlockState()
    : ObjectStates(
          std::make_shared<SmallVector<std::optional<ObjectState>, 8>>()) {}

SmallVector<std::optional<ObjectState>, 8> *
PEABlockState::getArrayForModification() {
  // Array-level copy-on-write. ObjectStates is a shared_ptr; when this
  // PEABlockState does not uniquely own it (a snapshot or peer shares it),
  // deep-copy the vector before mutating. ObjectState is stored by value in
  // the vector, so the deep copy already clones every slot — no per-slot
  // sharing annotation is needed.
  if (!ObjectStates) {
    // Defensive: a default-constructed PEABlockState always allocates an empty
    // vector, but a moved-from instance can leave us null.
    ObjectStates =
        std::make_shared<SmallVector<std::optional<ObjectState>, 8>>();
  } else if (ObjectStates.use_count() > 1) {
    // Defensive copy-on-write detach. NOTE: in the current architecture this
    // branch is unreachable — the analyzer's processBlock() resets CurrentState
    // to a fresh PEABlockState at every block header (resetPerBlockState), so
    // the one producer of shared state, the loop snapshot (takeLoopSnapshot),
    // never leaves CurrentState shared at a mutation point (use_count() is
    // always 1 here). It is retained as insurance: if that per-block-reset
    // invariant ever changes, mutating a shared vector in place would silently
    // corrupt the snapshot, and this detach is what prevents it.
    ObjectStates = std::make_shared<SmallVector<std::optional<ObjectState>, 8>>(
        *ObjectStates);
    assert(ObjectStates.use_count() == 1 &&
           "COW detach did not yield sole ownership");
  }
  return ObjectStates.get();
}

void PEABlockState::addObject(ObjectID ID, ObjectState State) {
  assert(ID != InvalidObjectID);
  // Route the write through the helper-returned pointer so we never touch
  // the shared_ptr's raw payload before COW has had a chance to detach us.
  auto *Arr = getArrayForModification();
  if (Arr->size() < ID + 1)
    Arr->resize(ID + 1);
  (*Arr)[ID] = std::move(State);
}

bool PEABlockState::hasObjectState(ObjectID ID) const {
  if (!ObjectStates || ID >= ObjectStates->size())
    return false;
  return (*ObjectStates)[ID].has_value();
}

const ObjectState &PEABlockState::getObjectState(ObjectID ID) const {
  assert(hasObjectState(ID));
  return *(*ObjectStates)[ID];
}

const ObjectState *PEABlockState::getObjectStateOptional(ObjectID ID) const {
  if (!ObjectStates || ID >= ObjectStates->size())
    return nullptr;
  const auto &Slot = (*ObjectStates)[ID];
  return Slot ? &*Slot : nullptr;
}

ObjectState &PEABlockState::getObjectStateForModification(ObjectID ID) {
  assert(ID != InvalidObjectID);
  auto *Arr = getArrayForModification();
  assert(ID < Arr->size() && (*Arr)[ID].has_value() &&
         "object not registered in this block state");
  // getArrayForModification already COW-detached the whole vector (deep-copying
  // every slot) if it was shared, so this slot is privately owned.
  return *(*Arr)[ID];
}

std::optional<ObjectID>
PEABlockState::resolveVirtualRef(Value *V, const AliasMap &Aliases) const {
  // We need a DataLayout for the GEP-walking inside pea::resolveVirtualRef.
  // Find it via V's parent Module if possible; otherwise we cannot resolve.
  if (!V)
    return std::nullopt;
  Module *M = nullptr;
  if (auto *I = dyn_cast<Instruction>(V)) {
    if (BasicBlock *BB = I->getParent())
      if (Function *F = BB->getParent())
        M = F->getParent();
  } else if (auto *A = dyn_cast<Argument>(V)) {
    if (Function *F = A->getParent())
      M = F->getParent();
  }
  if (!M)
    return std::nullopt;
  return pea::resolveVirtualRef(V, *this, Aliases, M->getDataLayout());
}

// ===========================================================================
// AliasMap
// ===========================================================================

void AliasMap::addVirtualAlias(Value *V, ObjectID ID) {
  assert(V && ID != InvalidObjectID);
  assert(!VirtualAliases.count(V) && "value already aliased");
  VirtualAliases[V] = ID;
  for (User *U : V->users()) {
    if (auto *I = dyn_cast<Instruction>(U))
      HasVirtualInputs.insert(I);
  }
}

void AliasMap::addScalarAlias(Value *V, Value *Replacement) {
  assert(V && Replacement);
  assert(V != Replacement && "scalar alias must not point to itself");
  assert(!getScalarAlias(Replacement) &&
         "scalar alias replacement must already be canonical");
  ScalarAliases[V] = Replacement;
}

void AliasMap::resetAlias(Value *V) {
  VirtualAliases.erase(V);
  ScalarAliases.erase(V);
}

std::optional<ObjectID> AliasMap::getVirtualAlias(Value *V) const {
  auto It = VirtualAliases.find(V);
  if (It == VirtualAliases.end())
    return std::nullopt;
  return It->second;
}

Value *AliasMap::getScalarAlias(Value *V) const {
  auto It = ScalarAliases.find(V);
  return It == ScalarAliases.end() ? nullptr : It->second;
}

Value *AliasMap::resolveScalarAlias(Value *V) const {
  assert(V && "cannot resolve a null scalar value");
  SmallPtrSet<Value *, 8> Visited;
  while (Value *Next = getScalarAlias(V)) {
    bool Inserted = Visited.insert(V).second;
    assert(Inserted && "cycle in scalar alias map");
    if (!Inserted)
      return V;
    V = Next;
  }
  return V;
}

void AliasMap::clear() {
  VirtualAliases.clear();
  ScalarAliases.clear();
  HasVirtualInputs.clear();
}

AliasMap AliasMap::snapshot() const { return *this; }

void AliasMap::restore(const AliasMap &S) { *this = S; }

// ===========================================================================
// PEAResult
// ===========================================================================

void PEAResult::computeEscapePointLocks() {
  // Globally merge + depth-sort the locks of every CASCADE group: a set of
  // MaterializeEffects (live-path OR per-pred) that share one escape point
  // (InsertBefore). Only such groups (>= 2 effects at a point) need a merged
  // re-emit; a single-effect escape point emits its own locks per-effect (its
  // NewInv trivially dominates its own emit point). Per-pred effects are
  // included: the critical-edge pre-pass re-aims every per-pred effect sharing
  // a (PH,S) edge onto the same split-edge block (or, on a single-succ pred,
  // they already share the pred's terminator), so a per-pred cascade groups
  // under one InsertBefore exactly like a live-path cascade. Also record the
  // highest SeqNo per escape point so the transform emits the merged list
  // once, from the last-applied effect (all sibling NewInvs then exist for
  // receiver lookup). See the comment on EscapePointLocks.
  DenseMap<Instruction *, unsigned> Count;
  for (auto &Kv : BlockEffects)
    for (jeandle::Effect &E : Kv.second) {
      auto *ME = dyn_cast<jeandle::MaterializeEffect>(&E);
      if (!ME)
        continue;
      if (auto *Key = dyn_cast_or_null<Instruction>(ME->InsertBefore))
        ++Count[Key];
    }
  for (auto &Kv : BlockEffects)
    for (jeandle::Effect &E : Kv.second) {
      auto *ME = dyn_cast<jeandle::MaterializeEffect>(&E);
      if (!ME)
        continue;
      Instruction *Key = dyn_cast_or_null<Instruction>(ME->InsertBefore);
      if (!Key)
        continue;
      auto CIt = Count.find(Key);
      if (CIt == Count.end() || CIt->second < 2)
        continue; // single-effect escape point — per-effect emit
      auto &Vec = EscapePointLocks[Key];
      for (const jeandle::MaterializedLock &ML : ME->Locks)
        Vec.push_back({ML.Callee, ML.NonReceiverArgs, ML.BytecodeDepth, ME});
      uint32_t &Max = MaxSeqForEscapePoint[Key];
      if (ME->SeqNo > Max)
        Max = ME->SeqNo;
    }
  for (auto &Kv : EscapePointLocks)
    llvm::sort(Kv.second, [](const jeandle::MergedLock &A,
                             const jeandle::MergedLock &B) {
      return A.BytecodeDepth < B.BytecodeDepth;
    });
}

PEAResult::~PEAResult() {
  // Any unparented PHI created by the analyzer (e.g., the analyzer ran
  // but the transform never consumed the result) must be freed. Once a PHI
  // has been inserted into a BasicBlock, that block's ilist owns it and we
  // must NOT delete here. WeakTrackingVH auto-nulls when the underlying
  // Value is deleted by an unrelated path (e.g. dead-code sweep), so the
  // null-check below also guards against stale references.
  for (WeakTrackingVH &VH : OwnedPhis) {
    if (Value *V = VH) {
      if (auto *Phi = dyn_cast<PHINode>(V))
        if (!Phi->getParent())
          delete Phi;
    }
  }
  // Same lifecycle for ReplaceLoad coercion instructions.
  for (WeakTrackingVH &VH : OwnedInsts) {
    if (Value *V = VH) {
      if (auto *I = dyn_cast<Instruction>(V))
        if (!I->getParent())
          I->deleteValue();
    }
  }
  // Loop-header field PHIs preserved across fixpoint iterations.
  for (WeakTrackingVH &VH : OwnedLoopFieldPhis) {
    if (Value *V = VH) {
      if (auto *Phi = dyn_cast<PHINode>(V))
        if (!Phi->getParent())
          delete Phi;
    }
  }
  // Per-pred materialization placeholders (never inserted by the transform).
  for (WeakTrackingVH &VH : OwnedMatPlaceholders) {
    if (Value *V = VH) {
      if (auto *I = dyn_cast<Instruction>(V))
        if (!I->getParent())
          I->deleteValue();
    }
  }
}

ObjectID PEAResult::createVirtualObject(std::unique_ptr<VirtualObject> VO) {
  assert(VO);
  ObjectID ID = static_cast<ObjectID>(VirtualObjects.size());
  // Re-stamp the ID. VirtualObject::ID is const, so use placement-new to
  // construct a fresh instance with the correct ID while keeping the rest of
  // the state intact.
  auto Stamped =
      std::make_unique<VirtualObject>(ID, VO->getKind(), VO->AllocationCall);
  Stamped->copyStructuralFieldsFrom(*VO);
  VirtualObjects.push_back(std::move(Stamped));
  return ID;
}

void PEAResult::addBlockEffect(std::unique_ptr<Effect> E) {
  assert(E->Block);
  // Per-effect trace (gated on -jeandle-trace-pea, off by default).
  // Centralised here so every emission site routes through a single trace
  // call; Effect::dump prints the kind, owning ObjectID (when set), block
  // label, and Target summary so a `2>&1 | grep PEA:` sweep is enough to
  // follow the analyser's decisions.
  if (JeandleTracePEA) {
    E->dump(llvm::dbgs());
    llvm::dbgs() << "\n";
  }
  BasicBlock *BB = E->Block;
  BlockEffects[BB].add(std::move(E));
}

Instruction *EliminateAllocationEffect::getTarget() const {
  return dyn_cast_or_null<Instruction>(static_cast<Value *>(Target));
}

void Effect::dump(raw_ostream &OS) const {
  OS << "PEA: ";
  switch (getKind()) {
  case Kind::ReplaceLoad:
    OS << "ReplaceLoad";
    break;
  case Kind::ReplaceCall:
    OS << "ReplaceCall";
    break;
  case Kind::EliminateStore:
    OS << "EliminateStore";
    break;
  case Kind::EliminateAllocation:
    OS << "EliminateAllocation";
    break;
  case Kind::Materialize:
    OS << "Materialize";
    break;
  case Kind::RecordDeoptState:
    OS << "RecordDeoptState";
    break;
  case Kind::CreatePHI:
    OS << "CreatePHI";
    break;
  case Kind::RewritePhiIncoming:
    OS << "RewritePhiIncoming";
    break;
  }
  if (ObjID != InvalidObjectID)
    OS << " [VO=" << static_cast<unsigned>(ObjID) << "]";
  if (Block && Block->hasName())
    OS << " block=%" << Block->getName();
  if (Instruction *T = getTarget())
    OS << " target=" << *T;
}

void MaterializeEffect::setInsertBefore(Instruction *I) { InsertBefore = I; }

bool PEAResult::hasOptimizationOpportunity() const {
  return VirtualizationDelta > 0 || AllocationDelta != 0 ||
         !BlockEffects.empty();
}
