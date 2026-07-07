//===- PartialEscapeUtils.cpp - PEA helper implementations ------*- C++ -*-===//
//
// Copyright (c) 2026, the Jeandle-LLVM Authors. All Rights Reserved.
//
// Part of the Jeandle-LLVM project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Pure helpers used by both the analysis and the transform pass.  No state.
// See PartialEscapeUtils.h for the interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Jeandle/PartialEscapeUtils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Jeandle/JavaType.h"
#include "llvm/IR/Jeandle/Metadata.h"
#include "llvm/IR/Jeandle/VMCallback.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"

using namespace llvm;

namespace llvm::jeandle::pea {

// ===========================================================================
// Callee-name predicates
// ===========================================================================

bool isJeandleCallNamed(const CallBase *CB, StringRef Name) {
  if (!CB)
    return false;
  // Reject inline asm and indirect calls.
  if (CB->isInlineAsm())
    return false;
  const Function *F = CB->getCalledFunction();
  if (!F)
    return false;
  return F->getName() == Name;
}

bool isJeandleNewInstance(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.new_instance");
}

bool isJeandleNewArray(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.new_array");
}

bool isJeandleAllocation(const CallBase *CB) {
  return isJeandleNewInstance(CB) || isJeandleNewArray(CB);
}

bool isJeandleArrayLength(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.arraylength");
}

bool isJeandleLoadKlass(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.load_klass");
}

bool isJeandleCheckCast(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.checkcast");
}

bool isJeandleInstanceOf(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.instanceof");
}

bool isJeandleCheckInstanceOf(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.check_instanceof");
}

bool isJeandleCheckIfValueBased(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.check_if_value_based");
}

bool isJeandleArrayStoreCheck(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.array_store_check");
}

bool isJeandlePostBarrier(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.post_barrier");
}

bool isJeandleMonitorEnter(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.monitorenter_with_monitor_lock") ||
         isJeandleCallNamed(CB, "jeandle.monitorenter_with_thin_lock") ||
         isJeandleCallNamed(CB, "jeandle.monitorenter_with_lightweight_lock");
}

bool isJeandleMonitorExit(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.monitorexit_with_monitor_lock") ||
         isJeandleCallNamed(CB, "jeandle.monitorexit_with_thin_lock") ||
         isJeandleCallNamed(CB, "jeandle.monitorexit_with_lightweight_lock");
}

bool isJeandleRegisterFinalizerIfNeeded(const CallBase *CB) {
  return isJeandleCallNamed(CB, "jeandle.register_finalizer_if_needed");
}

// ===========================================================================
// Type / klass helpers
// ===========================================================================

std::optional<JBasicType> elementTypeForArrayKlass(uintptr_t ArrayKlass) {
  if (ArrayKlass == 0)
    return std::nullopt;
  const jeandle::VMCallbacks *CB = jeandle::getVMCallbacks();
  if (!CB || !CB->ElementBasicTypeOfArrayKlass)
    return std::nullopt;
  int Raw = CB->ElementBasicTypeOfArrayKlass(ArrayKlass);
  if (Raw < 0 || Raw >= static_cast<int>(JBasicType::Count))
    return std::nullopt;
  return static_cast<JBasicType>(Raw);
}

Type *llvmElementTypeFor(JBasicType Kind, LLVMContext &Ctx) {
  switch (Kind) {
  case JBasicType::Boolean:
    return Type::getInt1Ty(Ctx);
  case JBasicType::Byte:
    return Type::getInt8Ty(Ctx);
  case JBasicType::Char:
    return Type::getInt16Ty(Ctx);
  case JBasicType::Short:
    return Type::getInt16Ty(Ctx);
  case JBasicType::Int:
    return Type::getInt32Ty(Ctx);
  case JBasicType::Long:
    return Type::getInt64Ty(Ctx);
  case JBasicType::Float:
    return Type::getFloatTy(Ctx);
  case JBasicType::Double:
    return Type::getDoubleTy(Ctx);
  case JBasicType::Object:
    return PointerType::get(Ctx, jeandle::AddrSpace::JavaHeapAddrSpace);
  case JBasicType::Count:
    return nullptr;
  }
  return nullptr;
}

// ===========================================================================
// Structural pointer chain walker
// ===========================================================================

// If V is `inttoptr(ptrtoint(x))` with matching pointer/int bit widths — a
// legal pointer-laundering round-trip — return x (the ptrtoint's pointer
// operand, after stripping offset-neutral `add/or X,0` chains the frontend
// may emit between the two casts). Tagged/masked encodings (any non-zero
// addend) are NOT round-trips and return nullptr.
//
// Shared by resolveVirtualRefImpl (identity resolution, case 7) and
// stripPointerCastsAndOffsets (offset resolution) so the two agree on exactly
// which laundered pointers are transparent: an `inttoptr` that is NOT a clean
// round-trip is opaque to both (identity -> nullopt, offset -> stop walking).
// Recognize an IntToPtr over both Instruction and ConstantExpr forms. LLVM
// has `PtrToIntOperator` but no `IntToPtrOperator`, so match via the generic
// Operator + opcode. Used at both call sites so identity and offset
// wrapper-stripping stay symmetric for the round-trip (Trap 4).
static bool isIntToPtrOp(const Value *V) {
  if (auto *Op = dyn_cast<Operator>(V))
    return Op->getOpcode() == Instruction::IntToPtr;
  return false;
}

static Value *getIntToPtrRoundTripInner(Value *V, const DataLayout &DL) {
  if (!isIntToPtrOp(V))
    return nullptr;
  auto *I2P = cast<Operator>(V);
  Value *Inner = I2P->getOperand(0);
  // Peel `add X, 0` / `or X, 0` chains between PtrToInt and IntToPtr; some
  // frontends emit these as part of constant-folded address arithmetic and
  // they are transparent (offset-neutral) to the round-trip. Use Operator +
  // opcode so ConstantExpr `add`/`or` is peeled too.
  for (unsigned StripDepth = 0; StripDepth < 8; ++StripDepth) {
    auto *BO = dyn_cast<Operator>(Inner);
    if (BO && (BO->getOpcode() == Instruction::Add ||
               BO->getOpcode() == Instruction::Or)) {
      if (auto *RHS = dyn_cast<ConstantInt>(BO->getOperand(1));
          RHS && RHS->isZero()) {
        Inner = BO->getOperand(0);
        continue;
      }
      if (auto *LHS = dyn_cast<ConstantInt>(BO->getOperand(0));
          LHS && LHS->isZero() && BO->getOpcode() == Instruction::Add) {
        Inner = BO->getOperand(1);
        continue;
      }
    }
    break;
  }
  // PtrToIntOperator covers both PtrToIntInst and ConstantExprPtrToInt.
  auto *P2I = dyn_cast<PtrToIntOperator>(Inner);
  if (!P2I)
    return nullptr;
  Type *PtrTy = P2I->getPointerOperand()->getType();
  if (auto *PT = dyn_cast<PointerType>(PtrTy)) {
    unsigned AS = PT->getAddressSpace();
    unsigned PtrBits = DL.getPointerSizeInBits(AS);
    unsigned IntBits = P2I->getType()->getIntegerBitWidth();
    if (PtrBits == IntBits)
      return P2I->getPointerOperand();
  }
  return nullptr;
}

Value *stripPointerCastsAndOffsets(Value *Ptr, const DataLayout &DL,
                                   int64_t *OutOffset, bool *NonConstant) {
  if (OutOffset)
    *OutOffset = 0;
  if (NonConstant)
    *NonConstant = false;
  if (!Ptr)
    return nullptr;

  Value *V = Ptr;
  // Bound the walk defensively; Jeandle IR typically has < 5 layers.
  for (unsigned Depth = 0; Depth < 32; ++Depth) {
    if (auto *GEP = dyn_cast<GEPOperator>(V)) {
      const unsigned AS = GEP->getPointerAddressSpace();
      const unsigned PtrBits = DL.getPointerSizeInBits(AS);
      APInt Acc(PtrBits, 0, /*isSigned=*/true);
      if (!GEP->accumulateConstantOffset(DL, Acc)) {
        if (NonConstant)
          *NonConstant = true;
        return V;
      }
      if (OutOffset)
        *OutOffset += Acc.getSExtValue();
      V = GEP->getPointerOperand();
      continue;
    }
    if (auto *BC = dyn_cast<BitCastOperator>(V)) {
      // Cross-address-space bitcasts are illegal IR; addrspacecast is required.
      // We don't assert here (this is a util used over arbitrary IR); just
      // continue on the inner operand.
      V = BC->getOperand(0);
      continue;
    }
    if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V)) {
      // Only chase through casts that stay in JavaHeapAddrSpace.
      auto *DstPT = dyn_cast<PointerType>(ASC->getType());
      auto *SrcPT = dyn_cast<PointerType>(ASC->getOperand(0)->getType());
      if (!DstPT || !SrcPT)
        return V;
      if (DstPT->getAddressSpace() != jeandle::AddrSpace::JavaHeapAddrSpace ||
          SrcPT->getAddressSpace() != jeandle::AddrSpace::JavaHeapAddrSpace)
        return V;
      V = ASC->getOperand(0);
      continue;
    }
    if (auto *FI = dyn_cast<FreezeInst>(V)) {
      V = FI->getOperand(0);
      continue;
    }
    // llvm.launder/strip.invariant.group are pointer-identity-preserving.
    // resolveVirtualRef sees them through the alias map (processIntrinsic
    // installs it); offset resolution must peel them too, so a launder-wrapped
    // GEP keeps its accumulated byte offset.
    if (auto *II = dyn_cast<IntrinsicInst>(V)) {
      Intrinsic::ID ID = II->getIntrinsicID();
      if (ID == Intrinsic::launder_invariant_group ||
          ID == Intrinsic::strip_invariant_group) {
        V = II->getArgOperand(0);
        continue;
      }
    }
    // inttoptr(ptrtoint(x)) same-width round-trip (legal laundering): peel to
    // x so any GEP inside the round-trip contributes its offset. A non-round-
    // trip inttoptr is opaque — stop here. (resolveVirtualRef would not have
    // resolved such a pointer to a virtual, so the offset is unused there.)
    // getIntToPtrRoundTripInner uses Operator-form casts so a ConstantExpr
    // round-trip is peeled the same as an Instruction-form one — keeping this
    // offset path symmetric with the identity path (Trap 4).
    if (Value *Inner = getIntToPtrRoundTripInner(V, DL)) {
      V = Inner;
      continue;
    }
    break;
  }
  return V;
}

// ===========================================================================
// resolveVirtualRef
// ===========================================================================

// Bound on structural recursion through PHI/Select. With Visited-set cycle
// detection the recursion is already bounded by the IR structure, but a hard
// depth cap prevents pathological diamond DAGs from causing exponential
// expansion through repeated PHI/Select operands.
static constexpr unsigned ResolveVirtualRefMaxDepth = 8;

static std::optional<ObjectID>
resolveVirtualRefImpl(Value *V, const PEABlockState &State,
                      const AliasMap &Aliases, const DataLayout &DL,
                      DenseSet<Value *> &Visited, unsigned Depth);

static std::optional<ObjectID>
checkAliasMap(Value *V, const PEABlockState &State, const AliasMap &Aliases) {
  // Virtual alias: a Value* registered as standing for some ObjectID.
  if (auto ID = Aliases.getVirtualAlias(V)) {
    if (const ObjectState *OS = State.getObjectStateOptional(*ID)) {
      if (OS->isVirtual())
        return *ID;
    }
    // Materialized or missing: V no longer denotes a virtual object.
    return std::nullopt;
  }
  // Scalar replacement aliases never resolve to a virtual object.
  if (Aliases.getScalarAlias(V) != nullptr)
    return std::nullopt;
  return std::nullopt;
}

// RAII helper: insert `V` into `Visited` on entry and erase on scope exit so
// that the set tracks "values currently on the DFS stack", not "values ever
// seen". The on-stack semantics correctly detect cycles (re-entering a value
// already being resolved) without rejecting legitimate diamond patterns where
// two sibling subtrees share a value (e.g. `select i1 %c, %o, %o`).
namespace {
struct StackGuard {
  DenseSet<Value *> &Set;
  Value *V;
  bool Inserted;
  StackGuard(DenseSet<Value *> &S, Value *Val) : Set(S), V(Val) {
    Inserted = Set.insert(V).second;
  }
  ~StackGuard() {
    if (Inserted)
      Set.erase(V);
  }
};
} // namespace

static std::optional<ObjectID>
resolveVirtualRefImpl(Value *V, const PEABlockState &State,
                      const AliasMap &Aliases, const DataLayout &DL,
                      DenseSet<Value *> &Visited, unsigned Depth) {
  if (!V)
    return std::nullopt;
  if (Depth > ResolveVirtualRefMaxDepth)
    return std::nullopt;
  // Cycle detection: if V is already on the resolution stack we're in a
  // self-reference (e.g. phi referencing itself) — bail.
  if (Visited.count(V))
    return std::nullopt;
  StackGuard Guard(Visited, V);

  // (1) Alias map lookup takes precedence over structural peeling so that
  // alias-registered Values (loads, PHIs, ...) resolve correctly even though
  // they have no structural relationship with their allocation site.
  if (Aliases.getVirtualAlias(V).has_value() ||
      Aliases.getScalarAlias(V) != nullptr)
    return checkAliasMap(V, State, Aliases);

  // (2) Constants and special values.
  if (auto *C = dyn_cast<Constant>(V)) {
    if (C->isNullValue() || isa<UndefValue>(C) || isa<PoisonValue>(C) ||
        isa<GlobalValue>(C) || isa<ConstantInt>(C) || isa<ConstantFP>(C) ||
        isa<ConstantPointerNull>(C))
      return std::nullopt;
    // ConstantExpr GEP/cast falls through to the structural cases below.
  }

  // (3) GEP — chase the base pointer (offset is resolveFieldOffset's job).
  if (auto *GEP = dyn_cast<GEPOperator>(V))
    return resolveVirtualRefImpl(GEP->getPointerOperand(), State, Aliases, DL,
                                 Visited, Depth + 1);

  // (4) AddrSpaceCast — only chase within JavaHeapAddrSpace.
  if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V)) {
    if (auto *DstPT = dyn_cast<PointerType>(ASC->getType()))
      if (DstPT->getAddressSpace() != jeandle::AddrSpace::JavaHeapAddrSpace)
        return std::nullopt;
    if (auto *SrcPT = dyn_cast<PointerType>(ASC->getOperand(0)->getType()))
      if (SrcPT->getAddressSpace() != jeandle::AddrSpace::JavaHeapAddrSpace)
        return std::nullopt;
    return resolveVirtualRefImpl(ASC->getOperand(0), State, Aliases, DL,
                                 Visited, Depth + 1);
  }

  // (5) BitCast.
  if (auto *BC = dyn_cast<BitCastOperator>(V))
    return resolveVirtualRefImpl(BC->getOperand(0), State, Aliases, DL, Visited,
                                 Depth + 1);

  // (6) Freeze.
  if (auto *FI = dyn_cast<FreezeInst>(V))
    return resolveVirtualRefImpl(FI->getOperand(0), State, Aliases, DL, Visited,
                                 Depth + 1);

  // (7) IntToPtr(PtrToInt(x)) round-trip with matching widths is a legal
  // laundering pattern (see getIntToPtrRoundTripInner); tagged-pointer
  // encodings (with masking/shifting) must escape. A non-round-trip inttoptr
  // is opaque — return nullopt so the caller materializes. isIntToPtrOp
  // covers both Instruction and ConstantExpr forms (Trap 4 symmetry).
  if (isIntToPtrOp(V)) {
    if (Value *Inner = getIntToPtrRoundTripInner(V, DL))
      return resolveVirtualRefImpl(Inner, State, Aliases, DL, Visited,
                                   Depth + 1);
    return std::nullopt;
  }

  // (8) PHINode / (9) SelectInst — recursion through Case-B-style merges that
  // the AliasMap didn't pre-install (e.g. PHIs outside the current BlockState
  // domain, chained PHIs not yet processed, or Selects, which have no
  // Case-B-style pre-installation since Select isn't a PHI). If every operand
  // resolves to the same ObjectID, the merge denotes that virtual object on
  // every execution path. StackGuard guarantees that Visited reverts to the
  // caller's state on every return path, so a sibling subtree that shares a
  // value with an earlier sibling can still descend through it.
  if (auto *Phi = dyn_cast<PHINode>(V)) {
    std::optional<ObjectID> Common;
    bool First = true;
    for (Value *In : Phi->incoming_values()) {
      // Treat a poison incoming as no-contribution. The pred path that
      // supplies poison is by definition unreachable at runtime (or produces
      // UB if reached); the incoming carries no information about virtual
      // identity. Skip it entirely so a single live virtual incoming makes
      // the whole PHI denote that virtual.
      if (isa<PoisonValue>(In))
        continue;
      auto Sub =
          resolveVirtualRefImpl(In, State, Aliases, DL, Visited, Depth + 1);
      if (!Sub)
        return std::nullopt;
      if (First) {
        Common = Sub;
        First = false;
      } else if (*Sub != *Common) {
        return std::nullopt;
      }
    }
    return Common;
  }

  if (auto *Sel = dyn_cast<SelectInst>(V)) {
    Value *TV = Sel->getTrueValue();
    Value *FV = Sel->getFalseValue();
    // A poison arm carries no virtual identity information; if the OTHER arm
    // resolves to a virtual, that virtual is the only possible runtime value
    // of the Select (the poison-arm path would be UB if executed).
    bool TPoison = isa<PoisonValue>(TV);
    bool FPoison = isa<PoisonValue>(FV);
    if (TPoison && FPoison)
      return std::nullopt;
    if (TPoison)
      return resolveVirtualRefImpl(FV, State, Aliases, DL, Visited, Depth + 1);
    if (FPoison)
      return resolveVirtualRefImpl(TV, State, Aliases, DL, Visited, Depth + 1);
    auto T = resolveVirtualRefImpl(TV, State, Aliases, DL, Visited, Depth + 1);
    if (!T)
      return std::nullopt;
    auto F = resolveVirtualRefImpl(FV, State, Aliases, DL, Visited, Depth + 1);
    if (!F || *T != *F)
      return std::nullopt;
    return T;
  }

  // (10) Call, Argument, ... — opaque to alias analysis. Fall through.
  return std::nullopt;
}

std::optional<ObjectID> resolveVirtualRef(Value *V, const PEABlockState &State,
                                          const AliasMap &Aliases,
                                          const DataLayout &DL) {
  DenseSet<Value *> Visited;
  return resolveVirtualRefImpl(V, State, Aliases, DL, Visited, /*Depth=*/0);
}

// ===========================================================================
// resolveFieldOffset
// ===========================================================================

std::optional<int64_t> resolveFieldOffset(Value *Ptr, const DataLayout &DL) {
  if (!Ptr)
    return std::nullopt;
  // Delegate to the shared offset accumulator so identity- and
  // offset-resolution peel the same set of wrappers (GEP constant offsets,
  // bitcast, JavaHeap addrspacecast, freeze, launder/strip.invariant.group,
  // inttoptr(ptrtoint(x)) round-trip): a pointer that resolves to a virtual
  // base yields its true byte offset here. A non-constant GEP -> nullopt
  // (caller materializes); a non-GEP base (the object itself, a whole-object
  // Case-B PHI/Select) -> 0.
  int64_t Off = 0;
  bool NonConst = false;
  stripPointerCastsAndOffsets(Ptr, DL, &Off, &NonConst);
  if (NonConst)
    return std::nullopt;
  return Off;
}

// ===========================================================================
// Misc
// ===========================================================================

uintptr_t extractAllocationKlass(const CallBase *AllocCB) {
  if (!AllocCB || AllocCB->arg_size() == 0)
    return 0;
  // Both jeandle.new_instance and jeandle.new_array take the klass pointer as
  // their first operand.
  return jeandle::extractKlassConstant(
      const_cast<Value *>(AllocCB->getArgOperand(0)));
}

std::optional<uint32_t> extractInstanceSize(const CallBase *NewInstance) {
  if (!isJeandleNewInstance(NewInstance) || NewInstance->arg_size() < 2)
    return std::nullopt;
  auto *CI = dyn_cast<ConstantInt>(NewInstance->getArgOperand(1));
  if (!CI)
    return std::nullopt;
  return static_cast<uint32_t>(CI->getZExtValue());
}

std::optional<uint32_t> extractArrayLength(const CallBase *NewArray) {
  if (!isJeandleNewArray(NewArray) || NewArray->arg_size() != 5)
    return std::nullopt;
  auto *CI = dyn_cast<ConstantInt>(NewArray->getArgOperand(1));
  if (!CI)
    return std::nullopt;
  return static_cast<uint32_t>(CI->getZExtValue());
}

} // namespace llvm::jeandle::pea
