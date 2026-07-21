//===- PartialEscapeUtils.h - PEA helpers ------------------------*- C++
//-*-===//
//
// Copyright (c) 2026, the Jeandle-LLVM Authors. All Rights Reserved.
//
// Part of the Jeandle-LLVM project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pure helpers used by both the analysis and the transform pass.  No state.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_JEANDLE_PARTIALESCAPEUTILS_H
#define LLVM_ANALYSIS_JEANDLE_PARTIALESCAPEUTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/Jeandle/PartialEscape.h"
#include "llvm/IR/Jeandle/VMConstants.h"
#include "llvm/IR/Value.h"
#include <cstdint>
#include <optional>

namespace llvm {
class CallBase;
class DataLayout;
class Function;
class GetElementPtrInst;
class Instruction;
class LLVMContext;
class Type;
} // namespace llvm

namespace llvm::jeandle::pea {

// Callee-name predicates. Match a CallBase by callee Function name. Tolerates
// CallInst, InvokeInst, and any subclass of CallBase. Returns false on inline
// asm, indirect calls, or NULL Function.
bool isJeandleCallNamed(const CallBase *CB, StringRef Name);
bool isJeandleNewInstance(const CallBase *CB);
bool isJeandleNewArray(const CallBase *CB);
bool isJeandleAllocation(const CallBase *CB); // either of the above
bool isJeandleArrayLength(const CallBase *CB);
bool isJeandleLoadKlass(const CallBase *CB);
bool isJeandleGetClass(const CallBase *CB);
bool isJeandleCheckCast(const CallBase *CB);
bool isJeandleInstanceOf(const CallBase *CB);
bool isJeandleCheckInstanceOf(const CallBase *CB);
bool isJeandleCheckIfValueBased(const CallBase *CB);
bool isJeandleArrayStoreCheck(const CallBase *CB);
bool isJeandlePostBarrier(const CallBase *CB);
bool isJeandleMonitorEnter(const CallBase *CB);
bool isJeandleMonitorExit(const CallBase *CB);
bool isJeandleRegisterFinalizerIfNeeded(const CallBase *CB);
bool isJeandleEncodeHeapOop(const CallBase *CB);
bool isJeandleDecodeHeapOop(const CallBase *CB);

// Extract the Java element basic type from an array klass pointer.
// TODO: returns std::nullopt when VMCallbacks are unavailable; pending
// full VMCallbacks integration.
std::optional<JBasicType> elementTypeForArrayKlass(uintptr_t ArrayKlass);

// Map a JBasicType to its LLVM IR storage type for one element (e.g., Byte→i8,
// Object→ptr addrspace(1)).  Returns nullptr if Kind == Count.
Type *llvmElementTypeFor(JBasicType Kind, LLVMContext &Ctx);

// Strip pointer-identity-preserving operations (bitcast, addrspacecast
// within addrspace(1), freeze) and constant-offset GEPs, accumulating the
// constant offset into *OutOffset. Returns the root pointer. Sets
// *NonConstant = true if a non-constant GEP index was encountered (then the
// accumulated offset is invalid). Shared structural helper used by
// resolveVirtualRef and resolveFieldOffset.
Value *stripPointerCastsAndOffsets(Value *Ptr, const DataLayout &DL,
                                   int64_t *OutOffset, bool *NonConstant);

// Resolve V to the ObjectID of the virtual object it names in the current
// State. Returns std::nullopt if V does not resolve to a virtual object.
// Cycle-safe via a small visited set. Canonical implementation; the
// PEABlockState::resolveVirtualRef overload delegates here.
std::optional<ObjectID> resolveVirtualRef(Value *V, const PEABlockState &State,
                                          const AliasMap &Aliases,
                                          const DataLayout &DL);

// Resolve V to the byte offset from its (presumed virtual) base pointer.
// Returns std::nullopt if V is not a constant-offset access (e.g., GEP with
// non-constant index, inttoptr, or unrecognised pattern).
std::optional<int64_t> resolveFieldOffset(Value *Ptr, const DataLayout &DL);

// Whether V denotes the object it resolves to WHOLE — at byte offset 0 on
// every execution path. resolveVirtualRef returns object identity and
// discards byte offsets, and resolveFieldOffset has no Select/PHI case (it
// returns 0 for them), so a consumer that records a whole-object reference
// (a VirtualRef field entry, a whole-object alias-forward) must check this:
// a Select/PHI whose arms may carry different non-zero offsets resolves to
// the object by identity without being address-equal to it. Recurses
// through Select arms and PHI incomings (cycle-safe via depth cap); leaves
// go through resolveFieldOffset.
bool isWholeObjectReference(Value *V, const DataLayout &DL);

// Find the start index (into the "deopt" bundle's Inputs) of the CURRENT
// (innermost) deopt scope's duplicated-BCI pair. A Jeandle deopt bundle is
// [root scope][inlinee scope]... with the innermost (current-method) scope
// LAST, each scope opening with an adjacent pair of equal-valued i32
// constants (the duplicated BCI marker; the root scope's pair is preceded by
// an i64 should_reexecute slot, inlinee scopes by a MethodType marker pair).
// The scan runs BACKWARD over the bundle inputs so inlinee scopes (appended
// last) are found first. Returns std::nullopt for a missing bundle, a
// malformed pair (mismatched adjacent values), or no pair at all — callers
// must treat those as "scope boundaries unknown" and bail conservatively
// (NEVER report_fatal_error on arbitrary IR: PEA may run on any legal LLVM
// IR, not only frontend-produced bundles).
std::optional<unsigned> findInnermostDeoptScopeBCIPairStart(const CallBase &CB);

/// Returns the start index of the FIRST (root) scope's duplicated-BCI pair in
/// the "deopt" operand bundle of CB, or std::nullopt if the bundle is missing
/// or carries no adjacent equal-i32 pair. A bundle is laid out as
/// [root scope][inlinee scope]... and every scope begins with a duplicated
/// BCI marker (two adjacent equal i32 constants). The marker is always
/// immediately preceded by at least one i64 (the should_reexecute slot; an
/// inlinee scope additionally carries its MethodType pair), and every i32
/// slot VALUE is always followed by an i64 encoding — so an adjacent
/// equal-i32 pair can only ever be a scope's BCI marker, and the FIRST one
/// anchors the root scope. PEA places ALL VO descriptors into the root
/// scope's VO section (the deopt-point-level object pool — C2's
/// dump_object_pool-before-scope-values analog), so this finder (not the
/// innermost one) anchors the descriptor insert position. Graceful: returns
/// std::nullopt on malformed bundles so callers can bail conservatively.
std::optional<unsigned> findFirstDeoptScopeBCIPairStart(const CallBase &CB);

// Returns the klass pointer (as uintptr_t) attached to the allocation call's
// klass argument, or 0 if not statically known.  Wraps
// jeandle::extractKlassConstant for the allocation-specific operand index.
uintptr_t extractAllocationKlass(const CallBase *AllocCB);

// Returns the constant size-in-bytes operand of jeandle.new_instance.
// std::nullopt if non-constant.
std::optional<uint32_t> extractInstanceSize(const CallBase *NewInstance);

// Returns the constant length operand of jeandle.new_array.
// std::nullopt if non-constant.
std::optional<uint32_t> extractArrayLength(const CallBase *NewArray);

} // namespace llvm::jeandle::pea

#endif
