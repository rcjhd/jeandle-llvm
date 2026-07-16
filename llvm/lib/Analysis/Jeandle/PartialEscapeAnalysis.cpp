//===- PartialEscapeAnalysis.cpp - PEA (analysis pass)
//---------------------===//
//
// Copyright (c) 2026, the Jeandle-LLVM Authors. All Rights Reserved.
//
// Part of the Jeandle-LLVM project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Partial Escape Analysis. Tracks Java objects as addrspace(1) pointers that
// are allocated at InvokeInst sites to jeandle.new_instance / jeandle.new_array
// and have not yet escaped. On an escape point (generic call, ret, store into
// non-virtual memory, ...) a Materialize effect is recorded; the transform
// re-emits the allocation immediately before the escape, replays tracked field
// stores, and re-emits surviving monitorenters. Effects are Graal-style
// polymorphic records (jeandle::Effect subclasses). isCfgKill() orders the
// transform's two apply passes and is true ONLY for EliminateAllocation;
// "control-flow-rewriting" (Materialize's splitBasicBlock, CreatePHI's PHI
// insertion) is a separate notion and runs in Pass 1, matching Graal's
// EffectList.isCfgKill().
//
// Per-block exit state (virtual set, field values, lock counts) is snapshotted
// into BlockExits; at each block header we inherit a single pred's snapshot or
// run mergeStates() over all preds. An object stays virtual iff every pred is
// Virtual and they agree on field values and lock counts; any disagreement
// marks the object ineligible and commit() drops its effects.
//
// At multi-pred merges, explicit LLVM PHIs of heap pointers are classified:
// Case A (a non-virtual incoming or a Case-C bail) materializes each virtual
// incoming at its pred; Case B (uniform ObjectID, still virtual) registers the
// PHI as an alias; Case C (distinct but compatible IDs) synthesizes one VO
// cloned from the first per-pred VO. A synthetic VO that later needs
// materialization is dropped together with its per-pred sources (see
// TODO(cascade-materialize) in materializeAt).
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Jeandle/PartialEscapeAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Jeandle/PartialEscape.h"
#include "llvm/Analysis/Jeandle/PartialEscapeUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Jeandle/JavaType.h"
#include "llvm/IR/Jeandle/Deoptimization.h"
#include "llvm/IR/Jeandle/Metadata.h"
#include "llvm/IR/Jeandle/VMCallback.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <functional>
#include <unordered_map>

#define DEBUG_TYPE "partial-escape-analysis"

using namespace llvm;

// Per-statistic counters surfaced via LLVM's standard
// `-stats` flag (Statistic.h). Counts are bumped at the analyzer's effect-
// emission sites; a small drift vs. the final committed effect set is
// possible when a late dropEffectsFor() strips a Materialize that was
// already counted (Eligible flipped after emission). Treated as a
// diagnostic, not an audit.
STATISTIC(JeandlePEAVirtualized, "Number of virtual objects PEA created");
STATISTIC(JeandlePEAEliminated, "Number of allocations eliminated by PEA");
STATISTIC(JeandlePEAMaterialized, "Number of materializations emitted by PEA");
STATISTIC(JeandlePEAMaterializedPHI,
          "Materializations triggered by PHI / Case-A merge");
STATISTIC(JeandlePEAMaterializedMerge,
          "Materializations triggered by state merge");
STATISTIC(JeandlePEAMaterializedLoopExit,
          "Materializations at loop preheader (force-drain)");
STATISTIC(JeandlePEAMaterializedUnhandled,
          "Materializations for unhandled instruction (escape point)");
// Extended counters. The first three measure how aggressively
// the analyzer drives its loop fixpoint, and so help calibrate
// MaxLoopFixpointIters / JeandlePEALoopCutoff. The last two split the
// "Unhandled" bucket so cascade and nested materializations can be
// audited separately from the user-visible escape-point reason.
STATISTIC(JeandlePEALoopFixpointRetries,
          "Sum of inner-fixpoint iterations across every processLoop call");
STATISTIC(JeandlePEAOuterFixpointIterations,
          "Top-level processLoop entries (one per top-level loop traversal)");
STATISTIC(JeandlePEAModeEscalations,
          "Regular -> MaterializeAll mode flips (escalations)");
STATISTIC(JeandlePEAMaterializedCascade,
          "Materializations triggered by strict-lock cascade");
STATISTIC(JeandlePEAMaterializedNested,
          "Materializations triggered by nested-virtual prerequisite");

// When set, dump per-function PEA classification stats
// on errs() after commit(). The line is prefixed with `;; PEA stats` so it
// is greppable both as IR-comment-shaped output and as a regular stderr
// line. Counts are derived from PEAResult::EscapeClassification (populated
// by commit()).
static llvm::cl::opt<bool> JeandleDumpPEAStats(
    "jeandle-dump-pea-stats", llvm::cl::init(false), llvm::cl::Hidden,
    llvm::cl::desc("PEA: dump per-function NeverEscapes / PartiallyEscapes "
                   "/ AlwaysEscapes counts on errs() after analysis."));

// Allocation sinking requires a deopt state for the replacement allocation.
// By default only the escaping call or invoke itself may provide that state.
// The option additionally permits the nearest safe prior deopt point.
static llvm::cl::opt<bool> JeandlePEAEnableAggressiveAllocationSinking(
    "jeandle-pea-enable-allocation-sinking", llvm::cl::init(false),
    llvm::cl::Hidden,
    llvm::cl::desc("PEA: also allow allocation sinking at the nearest safe "
                   "prior call or invoke carrying deopt state."));

// Cascade other still-locked virtuals at materialization when the target uses
// HotSpot's lightweight locking (LM_LIGHTWEIGHT requires strict lock nesting).
// Resolved per-run by resolveStrictLockOrder(): explicit cl override wins,
// otherwise the RequiresStrictLockOrder VMCallback is consulted; final
// fallback is true (JDK 21+ x86_64 default).
static llvm::cl::opt<bool> JeandleAssumeStrictLockOrder(
    "jeandle-assume-strict-lock-order", llvm::cl::init(true), llvm::cl::Hidden,
    llvm::cl::desc("PEA: testing override for whether the target VM requires "
                   "strict lock nesting (cascades still-locked virtuals on "
                   "materialization). When unset, the value is queried from "
                   "the RequiresStrictLockOrder VM callback."));

// Cap on array length for virtualization candidates (default 128). Hidden
// cl::opt so tests can override without rebuilding.
static llvm::cl::opt<unsigned> MaximumEscapeAnalysisArrayLength(
    "jeandle-pea-max-array-length", llvm::cl::init(128), llvm::cl::Hidden,
    llvm::cl::desc("PEA: cap on array length eligible for virtualization. "
                   "Larger arrays bypass PEA. Default 128."));

// Testing-only knob: when true, every top-level processLoop entry switches to
// Mode::MaterializeAll for the whole fixpoint, exercising the deferred
// end-of-block Materialize emission path without constructing a pathological
// non-converging fixpoint. Default off; not surfaced to the VM.
static llvm::cl::opt<bool> JeandlePEAForceMaterializeAll(
    "jeandle-pea-force-materialize-all", llvm::cl::init(false),
    llvm::cl::Hidden,
    llvm::cl::desc("PEA: force every top-level processLoop into "
                   "Mode::MaterializeAll on entry. Testing aid for "
                   "virtualize-then-materialise coverage."));

// Loop nesting DEPTH threshold. When a top-level processLoop encounters a
// nest whose maximum depth exceeds this value, the analyzer transiently
// enters Mode::StopNewInLoopNest: processAllocation refuses NEW virtual
// allocations inside the nest, but all other operations continue. Bounds the
// worst-case cost of a deep nest while preserving virtualisation for objects
// allocated outside it. Default 20. Distinct from MaxLoopFixpointIters (10),
// which caps BODY iterations within a single fixpoint.
static llvm::cl::opt<unsigned> JeandlePEALoopCutoff(
    "jeandle-pea-loop-cutoff", llvm::cl::init(20), llvm::cl::Hidden,
    llvm::cl::desc("PEA: loop nesting depth threshold. When a nest's maximum "
                   "depth exceeds this value, the analyzer enters the "
                   "Mode::StopNewInLoopNest state for the duration of the "
                   "nest (refuse new virtualisations but keep tracking "
                   "existing virtuals). Default 20."));

// Filter PEA to functions whose name contains a given substring.
// When the option is non-empty, Analyzer::run() returns an empty PEAResult
// for any function whose name does not contain the filter; the transform
// pass then sees nothing to commit. Empty (the default) means "analyze
// every Java method". Used to focus PEA on a specific compilation, and by
// lit tests that want to exercise gating.
static llvm::cl::opt<std::string> JeandleEscapeAnalyzeOnly(
    "jeandle-pea-analyze-only", llvm::cl::init(""), llvm::cl::Hidden,
    llvm::cl::desc("PEA: only analyze functions whose name contains the "
                   "supplied substring. Empty (the default) analyzes every "
                   "Java method gated by jeandle.java_method_compilation."));

// Per-effect dbgs() trace. The cl::opt itself lives in
// PartialEscape.cpp (the centralised effect-emission site at
// PEAResult::addBlockEffect is the only consumer), so no declaration is
// needed in this translation unit. Turn on with -jeandle-trace-pea for
// lit-time diagnostics.

AnalysisKey PartialEscapeAnalysis::Key;

namespace {

// A live (still-unbalanced) monitorenter on a virtual object.
//   * Call is the original jeandle.monitorenter call site (used by
//     materializeAt to undo the ReplaceCall elision).
//   * BytecodeDepth is the lock-nesting ordering key used by the strict-lock
//     cascade and the merge-time stack-identity comparison. It is sourced from
//     the analyzer's monotonic NextLockEnterOrder, cached per call site in
//     LockDepthCache (stable across loop-fixpoint iterations). Order is a
//     separate per-push tag that DOES advance on every re-push; the loop
//     fixpoint convergence check therefore compares Call identity only.
// Cascade decisions and merge-time stack-identity compare BytecodeDepth.
struct LockEnter {
  llvm::CallBase *Call;
  uint32_t Order;
  uint32_t BytecodeDepth;
};

// Per-block-exit snapshot of the analyzer's per-object state. We record
// this AFTER processing a block so successors can either inherit directly
// (single live pred) or merge across multiple preds. Keeping the state
// per-block — rather than one global accumulating state — is what makes
// multi-pred merge correct in the presence of branches that mutate the field
// values of a virtual independently.
//
// BlockExitData carries the raw per-object data fields. BlockExitInfo
// inherits from it and adds the exception-edge state-splitting
// book-keeping that lives alongside the normal exit data.
struct BlockExitData {
  DenseSet<jeandle::ObjectID> Virtuals;
  DenseSet<jeandle::ObjectID> Materialized;
  DenseMap<jeandle::ObjectID, DenseMap<int64_t, jeandle::FieldValue>>
      FieldStates;
  DenseMap<jeandle::ObjectID, unsigned> LockCounts;
  // Per-object live monitorenter stack at block exit. Each entry is an
  // unbalanced monitorenter call site (i.e. its matching monitorexit hasn't
  // been seen yet on this path) PLUS a per-run-monotonic order tag (see
  // LockEnter). Sized identically to LockCounts[ID]. Used by materializeAt
  // to undo only the path-relevant elisions; the Order field powers the
  // narrow cascade rule.
  DenseMap<jeandle::ObjectID, SmallVector<LockEnter, 4>> LiveLockEnters;
  // Per-object materialized Value* at block exit. For an object in
  // Materialized, this is the LLVM pointer that downstream merges should use
  // as the PHI input on this predecessor edge. Initially the placeholder is
  // the original allocation (VObj.AllocationCall); the transform redirects
  // through MatPerBlock at apply time.
  DenseMap<jeandle::ObjectID, Value *> MaterializedValues;
};

struct BlockExitInfo : BlockExitData {
  // Exception-edge state-splitting. Only meaningful when the block's
  // terminator is an InvokeInst (which has both a normal and an unwind
  // successor).
  //
  // Jeandle's per-block analyzer processes the terminator-invoke last; the
  // materialize Effects emitted by the invoke are physically inserted
  // immediately before the invoke and survive on BOTH IR successors
  // (definitional dominance). The state-split here is purely about which
  // VirtualMap / FieldStates each successor INHERITS from the analyzer's
  // book-keeping: the normal successor sees the post-call state (the base
  // data of this struct), while the unwind successor sees the pre-call
  // snapshot recorded in UnwindData (the materialize logically happened
  // during the call, so on unwind any partially-materialized state is
  // unobservable to the handler).
  //
  // TerminatorInvoke / UnwindDest are stashed so the analyzer's pred-state
  // lookup (exitDataFor) can detect "this pred's terminator is an invoke
  // whose unwind dest equals the successor block I'm processing".
  llvm::InvokeInst *TerminatorInvoke = nullptr;
  BasicBlock *UnwindDest = nullptr;
  // The invoke was fully virtualized (e.g. processJavaOp emitted a
  // ReplaceCall effect on it). The transform replaces such invokes with
  // an unconditional branch to the normal dest, so from the analyzer's
  // perspective the unwind edge is dead — the handler is unreachable for
  // analysis purposes. exitDataFor returns nullptr when a successor asks
  // for a killed unwind edge, which makes the merge consumer treat the
  // pred as contributing nothing on that path (mark-as-dead).
  bool UnwindEdgeKilled = false;
  // Pre-invoke snapshot of the per-object data. Captured by processBlock
  // immediately before applying the terminator-invoke, then stashed here
  // only if the post-invoke base data differs (i.e. the invoke actually
  // changed some per-object state, e.g. by materializing an operand).
  // When unset (and UnwindEdgeKilled is false), the unwind successor
  // inherits the same base data as the normal successor (no state-split
  // needed because the invoke was a no-op for PEA).
  std::optional<BlockExitData> UnwindData;
};

// Resolve the effective "strict lock order" value for one analyzer run.
// Precedence:
//   1. If the JeandleAssumeStrictLockOrder cl::opt was explicitly set on the
//      command line (getNumOccurrences() > 0): honor it (testing override).
//   2. Otherwise, if the RequiresStrictLockOrder VMCallback is registered:
//      call it once and use its value (1 == strict, 0 == relaxed).
//   3. Otherwise (no callback, no override): fall back to the cl::opt's
//      default (true). This keeps lit-test behavior unchanged when no JVM
//      is wired up.
//
// The result is cached for the lifetime of an Analyzer instance, so the
// VMCallback fires at most once per compilation — which keeps the callback
// log's sequential-consistency invariant trivially satisfied (it appears
// exactly once at the start of each compilation's recording, and replay
// matches that ordering).
static bool resolveStrictLockOrder() {
  if (JeandleAssumeStrictLockOrder.getNumOccurrences() > 0)
    return JeandleAssumeStrictLockOrder;
  if (const jeandle::VMCallbacks *CB = jeandle::getVMCallbacks()) {
    if (CB->RequiresStrictLockOrder)
      return CB->RequiresStrictLockOrder() != 0;
  }
  return JeandleAssumeStrictLockOrder;
}

class Analyzer {
public:
  Analyzer(Function &F, DominatorTree &DT, LoopInfo &LI,
           jeandle::PartialEscapeOptions Options)
      : F(F), DT(DT), LI(LI), DL(F.getParent()->getDataLayout()),
        StrictLockOrder(resolveStrictLockOrder()),
        LockEliminationEnabled(Options.EliminateLocks) {}

  jeandle::PEAResult run();

  // Cap on iterations for the loop-fixpoint.
  static constexpr unsigned MaxLoopFixpointIters = 10;

  // Loop-nest execution mode. Mirrors Graal's EffectsClosureMode
  // (EffectsClosure.java:331-345), a single closure-global field:
  //   Regular            (REGULAR_VIRTUALIZATION) — processAllocation registers
  //                        virtuals normally.
  //   StopNewInLoopNest  (STOP_NEW_VIRTUALIZATIONS_LOOP_NEST) —
  //   processAllocation
  //                        refuses NEW virtualisations inside the active loop
  //                        nest, but already-virtual objects, loads/stores,
  //                        merges, locks, and exits all continue to be tracked
  //                        exactly as in Regular. Graal flips this per-loop in
  //                        stripKilledLoopLocations when loop.depth > cutoff
  //                        (PClosure:675); Jeandle flips it nest-wide at the
  //                        outermost processLoop when the nest's max depth
  //                        exceeds JeandlePEALoopCutoff (conservative analog).
  //                        In this mode, ensureMaterialized on a (necessarily
  //                        outer-scope) virtual object polls OverflowFlag
  //                        (Graal throws EffecsClosureOverflowException).
  //   MaterializeAll     (MATERIALIZE_ALL) — processAllocation registers AND
  //                        immediately schedules an end-of-block materialise
  //                        for the new VO (virtualize-then-materialise), so
  //                        intra-block folds survive. Reached on the first
  //                        non-convergence or overflow (Graal
  //                        EClosure:550-561); the mode persists through the
  //                        rest of the nest and is reset to Regular only at
  //                        depth==1 convergence (EClosure:482-488).
  enum class Mode : uint8_t { Regular, StopNewInLoopNest, MaterializeAll };

private:
  Function &F;
  DominatorTree &DT;
  LoopInfo &LI;
  const DataLayout &DL;
  // Cached "strict lock order" decision for this run; see
  // resolveStrictLockOrder() for the precedence rules.
  const bool StrictLockOrder;
  const bool LockEliminationEnabled;
  jeandle::PEAResult Result;

  // ---------------------------------------------------------------------
  // STATE MODEL — intentional divergence from Graal (documented so each
  // map below maps to its Graal counterpart).
  //
  // Graal carries a block's entire PEA state in ONE PartialEscapeBlockState
  // whose core is an ObjectState[] array indexed by VO id (each ObjectState
  // holding entries/locks/materializedValue). Jeandle instead fragments the
  // equivalent information across the analyzer-wide maps declared below, each
  // keyed by ObjectID:
  //   CurrentState (PEABlockState)   <-> the live PartialEscapeBlockState
  //   FieldStates[ID][offset]        <-> Graal ObjectState.entries (Jeandle's
  //                                    ObjectState carries NO field state)
  //   LockCounts[ID] / LiveLockEnters[ID]  <-> ObjectState.locks (lock stack)
  //   Materialized (DenseSet) / MaterializedValues[ID]  <->
  //                                    ObjectState's materialized flag + value
  //   Aliases (AliasMap)             <-> Graal's global `aliases` map
  //
  // This is deliberate: Jeandle is an RPO single-pass walk over LLVM IR (not
  // Graal's CFG-block-effect closure), and per-block snapshots for the merge
  // fixpoint are encoded in BlockExitData/BlockExitInfo rather than cloned
  // PartialEscapeBlockState arrays.
  // TODO(graal-unify): unifying to Graal's single-container model would touch
  // ~every state read/write, snapshot/restore, the merge fixpoint, and
  // LoopSnapshot, with no observable IR benefit.
  // ---------------------------------------------------------------------
  jeandle::AliasMap Aliases;
  // Per-block accumulating object state. Reset at the top of every block from
  // the predecessor snapshots (single inherit / multi-pred merge); rebuilt as
  // the instructions in the block are processed; snapshotted to BlockExits at
  // the end of the block.
  jeandle::PEABlockState CurrentState;

  // Per-object field state: ObjectID -> (offset -> FieldValue). This — not the
  // (field-less) ObjectState — is Jeandle's counterpart to Graal's
  // ObjectState.entries. Field discovery is lazy, so this map is deliberately
  // kept decoupled from VirtualObject::Fields (the two are not kept in lock
  // step).
  // Both VirtualObject::Fields and this map are path-dependent: they record
  // only offsets that some store/load actually touched, never the declared
  // field layout. An offset absent here means "Java default" (zero/null), not
  // "field does not exist" — never treat the size of either as a structural
  // invariant of the object's type (see synthesizeCaseC).
  DenseMap<jeandle::ObjectID, DenseMap<int64_t, jeandle::FieldValue>>
      FieldStates;

  // Per-object eligibility flag. Function-wide: starts true at allocation;
  // flipped to false on any escape (non-constant offset, type mismatch,
  // nested-virtual store/load, opaque consumer, incompatible multi-pred
  // merge). Once false, commit() drops every recorded effect for the object.
  DenseMap<jeandle::ObjectID, bool> Eligible;

  // Per-object monitor lock counter. Incremented on a folded monitorenter,
  // decremented on a folded monitorexit. Any object with LockCount != 0 at
  // commit time is marked ineligible (unbalanced locking).
  DenseMap<jeandle::ObjectID, unsigned> LockCounts;

  // Per-block live monitorenter stack per ObjectID. Pushed by a folded
  // monitorenter on the receiver, popped by the matching monitorexit. At any
  // point the stack contains exactly the unbalanced enter call sites whose
  // matching exits haven't been seen yet on this path; size == LockCounts[ID].
  // Reset+inherited per block (same lifecycle as LockCounts) so siblings in
  // a diamond CFG don't share stacks. materializeAt walks this stack to
  // decide which ReplaceCall elisions to undo. Each entry also carries a
  // per-run-monotonic Order tag (see LockEnter) serving as the lock-depth
  // stand-in for the narrow cascade rule. back().Order == this VO's max
  // lock depth, front().Order == this VO's min lock depth.
  DenseMap<jeandle::ObjectID, SmallVector<LockEnter, 4>> LiveLockEnters;

  // Monotonically increasing counter, bumped on every push to any
  // LiveLockEnters[id]. Never reset within an Analyzer run — relative
  // ordering within any one snapshot is all that matters for the cascade
  // rule, and across loop fixpoint iterations the inherited snapshots
  // already carry their own (older) Order tags, so a fresh push on
  // re-processing gets a strictly larger value without disturbing the
  // ordering of stale entries.
  uint32_t NextLockEnterOrder = 0;

  // Stable-across-iterations BytecodeDepth source. We need a per-call-site
  // depth value that (a) is a meaningful proxy for cascade decisions within
  // a single processBlock walk, AND (b) does NOT change across loop-fixpoint
  // re-pushes of the same call. The Analyzer-run-monotonic NextLockEnterOrder
  // satisfies (a) but fails (b) — every re-push at iteration N+1 advances
  // the counter, which would mutate ObjectState::Locks[i].BytecodeDepth and
  // break the structural ObjectState::equivalentTo check used by the loop
  // fixpoint convergence path (via PEABlockState::equivalentTo).
  //
  // Solution: cache the FIRST NextLockEnterOrder value ever assigned to each
  // call site and reuse it on subsequent visits. The cache survives loop
  // snapshot/restore (it's an Analyzer-run-wide property of the call site,
  // not per-block state). This RPO-order proxy is sound for every load-bearing
  // use of BytecodeDepth (cascade `<`, re-emit sort, locksEqual, strict-
  // increasing assert): under the structured-locking assumption PEA already
  // relies on, "lock X acquired before Y while both live" <=> "X's enter
  // dominates Y's" <=> "RPO visits X before Y" <=> proxy(X) < proxy(Y).
  DenseMap<CallBase *, uint32_t> LockDepthCache;

  // Per-path "this object has already been materialized somewhere upstream"
  // set. Materialization is recorded at most once per ObjectID per pred path
  // — the first escape site wins; multi-pred merges that see the object
  // materialized on every incoming carry the Materialized state forward.
  DenseSet<jeandle::ObjectID> Materialized;

  // Per-block exit snapshots, keyed by the block that produced them.
  DenseMap<BasicBlock *, BlockExitInfo> BlockExits;

  // Function-wide dedup of (Pred, TargetMerge, ObjectID) materializations.
  // Multiple merge-time Materialize-at-pred emissions for the same (Pred, M,
  // ID) would otherwise produce duplicate invokes; this nested map ensures we
  // emit exactly one Materialize effect per (Pred, M, ID) across the entire
  // run. Per-pred mats for distinct target merges M1, M2 at the same PH are NOT
  // deduped (they are distinct edge materializations). The Case-A / global path
  // passes M=null (mat at the pred's terminator end, shared across merges) so
  // its (PH, null, ID) entry dedups across all merges sharing PH. Nested as
  // PH -> M -> ID set so `MaterializedAtPred[PH][M]` is a `DenseSet<ID>&`
  // (bindable to MaterializeContext::MaterializedSet) and the loop rollback's
  // `MaterializedAtPred.erase(BB)` still erases per-PH (all M, all ID).
  DenseMap<BasicBlock *, DenseMap<BasicBlock *, DenseSet<jeandle::ObjectID>>>
      MaterializedAtPred;

  // Per-merge-block deferred CreatePHI effects. mergeStates pushes
  // every CreatePHI it would have committed directly onto this list (keyed
  // by merge block); processBlock drains the list at the END of the block
  // walk, assigning each effect a FRESH nextSeqNo() at drain time. Drain-time
  // SeqNo assignment guarantees a deferred CreatePHI sorts strictly AFTER any
  // per-pred Materialize emitted in the same processBlock walk. This matters
  // when the merge block is its own back-edge predecessor — both the CreatePHI
  // and a per-pred Materialize land in BlockEffects[BB], and the per-pred
  // Materialize must populate MatPerBlock first so the CreatePHI wires its
  // back-edge incoming to the per-pred NewInv, not the OrigAlloc placeholder
  // (which EliminateAllocation later poisons).
  DenseMap<BasicBlock *, jeandle::EffectList> PendingMergePhis;

  // Current analyzer mode. Flipped to MaterializeAll by
  // processLoop's overflow retry, or to StopNewInLoopNest at top-level
  // processLoop entry on a nest deeper than JeandlePEALoopCutoff, then
  // reverted before processLoop returns.
  Mode CurrentMode = Mode::Regular;

  // Cross-recursion overflow signal — the polled (-fno-exceptions) equivalent
  // of Graal's EffecsClosureOverflowException. Latched in ensureMaterialized
  // when CurrentMode == StopNewInLoopNest (a deep nest) and a virtual object
  // is about to be materialized: in STOP_NEW no new virtualizations occur, so
  // such an object must be an outer-scope (pre-loop / outer-loop) allocation,
  // and materializing it would force re-iteration of the whole nest
  // (exponential in nest depth). Nested processLoop calls (depth>1) return
  // immediately when it is already set (propagation, matching Graal's
  // re-throw), and the outermost (depth==1) processLoop catches it, restores
  // the snapshot, drains the preheader, and redoes the nest in MATERIALIZE_ALL
  // (EffectsClosure.java:533-551). Cleared on every top-level processLoop entry
  // and consumed (cleared) immediately before the MATERIALIZE_ALL retry.
  bool OverflowFlag = false;

  // Per-block list of VOs registered while CurrentMode was
  // MaterializeAll. processBlock drains this list at end-of-block (after
  // the instruction walk but before snapshotExitState) and emits a
  // Materialize effect for each VO at the block's terminator. The
  // deferred emission lets all intra-block loads/stores fold against
  // FieldStates first; the materialise then captures the final field
  // values. Cleared per block by resetPerBlockState semantics — but the
  // bucket is keyed by BasicBlock so siblings don't collide.
  DenseMap<BasicBlock *, SmallVector<jeandle::ObjectID, 4>>
      PendingMaterializeAllVOs;

  // Stable allocation-site -> ObjectID cache. processAllocation consults this
  // before creating a fresh VO so re-processing the same alloc inside a loop
  // fixpoint iteration reuses the original ID (otherwise the convergence
  // check, which compares FieldStates and Virtuals sets across iterations,
  // would diverge forever). Monotonically grown; never cleared.
  DenseMap<CallBase *, jeandle::ObjectID> AllocSiteToVO;

  // Every Loop* on which processLoop was invoked (including recursive
  // sub-loop calls). The safety-net
  // materializePreheaderVirtualsForUnvisitedLoops() drains preheader
  // virtuals ONLY for loops absent from this set, i.e. truly unvisited
  // loops (an unreachable top-level loop the RPO walk skipped, or a
  // sub-loop whose containing top-level loop's processLoop bailed before
  // recursing into it). processLoop handles every loop it visits —
  // whether the body fixpoint converged or fell into the MATERIALIZE_ALL
  // fallback — by draining preheader virtuals itself, so the only loops
  // that need this safety-net drain are those processLoop never ran on.
  DenseSet<Loop *> VisitedLoops;

  // Per-in-loop-block field-PHI cache. Keyed on (BB, ID, Offset) where BB is
  // any merge block inside a loop (loop header OR non-header in-loop merge).
  // The cache returns a STABLE PHINode* across fixpoint iterations so the
  // convergence check on BlockExitInfo.FieldStates can compare FieldValues
  // by Value pointer (otherwise every iteration would synthesize a fresh
  // PHI and the fixpoint would never close), AND so that the preserved
  // BlockExits[BB] (restoreLoopSnapshot does not roll back loop-block
  // BlockExits) never references a PHI that rollback deletes. Offset == -1
  // (i.e. the magic VirtualObject::ArrayLengthSlotOffset value, also used as
  // a sentinel here for the "merged materialized pointer" PHI in the
  // all-materialized merge branch) is overloaded as the cache key for the
  // materialized-ptr PHI; offsets >= 0 are field PHIs. The cached PHI lives
  // in Result.OwnedLoopFieldPhis, which is preserved across rollback (unlike
  // OwnedPhis, which is truncated). The CreatePHI Effect referencing the
  // cached PHI is re-emitted in BlockEffects[BB] on every iteration —
  // BlockEffects[BB] is wiped on rollback, but the PHI itself is not. The
  // `Header` field is the merge block BB passed to getOrCreateLoopFieldPhi.
  struct LoopPhiKey {
    BasicBlock *Header;
    jeandle::ObjectID ID;
    int64_t Offset; // -1 for the merged-materialized-ptr PHI
    bool operator==(const LoopPhiKey &O) const {
      return Header == O.Header && ID == O.ID && Offset == O.Offset;
    }
  };
  struct LoopPhiKeyHash {
    size_t operator()(const LoopPhiKey &K) const {
      return static_cast<size_t>(
          hash_combine(hash_value(K.Header), K.ID, K.Offset));
    }
  };
  std::unordered_map<LoopPhiKey, PHINode *, LoopPhiKeyHash> LoopFieldPhiCache;

  // Per-pred materialization placeholder cache (Graal analog: a distinct
  // AllocatedObjectNode created per materialize). The merge-driven
  // materialize path (materializeAtPredFromExitInfo) gives each predecessor
  // a DISTINCT placeholder Value* for a materialized object, so the
  // MergeProcessor fast-path's deep-value equivalence check is sound
  // post-cascade (distinct values => not equivalent). Keyed by
  // {predecessor-block, target-merge, ObjectID}: a per-pred materialize is
  // destined for ONE target merge (the MergeProcessor::BB in scope), and the
  // same PH may carry distinct per-pred mats for different target merges (one
  // per critical edge), so the placeholder must be distinct per (PH, M, ID).
  // The Case-A / global path passes M=null (mat at the pred's terminator end,
  // NewInv dominates all successors, shared flip) — its placeholder is shared
  // across merges (one mat per (PH, ID)). The cache is loop-stable: the same
  // (PH, M, ID) returns the same placeholder across loop-fixpoint iterations,
  // keeping the header's B-vs-B' convergence stable. Placeholders live in
  // Result.OwnedMatPlaceholders; the transform resolves them away (never
  // inserts them). See Effect::PerPredPlaceholder / Effect::TargetMergeBB.
  struct PerPredPlaceholderKey {
    BasicBlock *PH;
    BasicBlock *TargetMerge; // null for Case-A / global
    jeandle::ObjectID ID;
    bool operator==(const PerPredPlaceholderKey &O) const {
      return PH == O.PH && TargetMerge == O.TargetMerge && ID == O.ID;
    }
  };
  struct PerPredPlaceholderKeyHash {
    size_t operator()(const PerPredPlaceholderKey &K) const {
      return static_cast<size_t>(
          hash_combine(hash_value(K.PH), hash_value(K.TargetMerge), K.ID));
    }
  };
  std::unordered_map<PerPredPlaceholderKey, Value *, PerPredPlaceholderKeyHash>
      PerPredMatPlaceholderCache;

  // Per-VO record of LLVM pointer-PHIs that processBlockPhis
  // aliased via Case-B (every incoming agrees on the same ObjectID).
  // commit() consults this to schedule explicit PHI erasures for VOs
  // that end up NeverEscapes, so EliminateAllocation isn't left with
  // a dead-but-still-parented `phi [poison, ..., poison]` survivor
  // in the IR.
  DenseMap<jeandle::ObjectID, SmallVector<llvm::PHINode *, 2>> CaseBPhiAliases;

  // Graal-aligned MergeProcessor (mirrors PartialEscapeClosure's inner
  // MergeProcessor class). Holds the per-merge context (merge block,
  // predecessor exit snapshots, the intersected ID set) and drives the
  // do/while merge fixpoint. The Graal-named steps correspond as:
  //   run()                    ~= merge() + doMergeWithoutDead()
  //   intersectVirtualObjects  ~= Graal intersectVirtualObjects
  //   mergeObjectState(ID)     ~= the per-object loop body in merge()
  //   mergeFieldStates(ID)     ~= Graal mergeObjectStates (per-entry phi)
  // processBlockPhis / synthesizeCaseC (~= Graal processPhi) remain Analyzer
  // methods and are invoked from the fixpoint's phi loop. Reference members
  // alias the Analyzer's per-block state so the ported merge code reads
  // verbatim (only method calls are qualified with A.).
  class MergeProcessor {
  public:
    MergeProcessor(Analyzer &A, BasicBlock *BB);
    void run();

  private:
    Analyzer &A;
    BasicBlock *BB;
    // Per-merge context (the Graal MergeProcessor's mergeBlock + caches).
    // Preds[i] points at the shared `BlockExits[Pred]` (or its UnwindData
    // variant) via exitDataFor. Per-pred materialize does NOT flip this shared
    // state (its NewInv dominates one critical edge — see
    // materializeAtPredFromExitInfo); the merge reads the per-pred placeholder
    // directly via getOrCreatePerPredMatPlaceholder. Case-A / global
    // materialize DOES flip the shared state (NewInv at PH end dominates all
    // successors — Graal-aligned, benign).
    SmallVector<BlockExitData *, 4> Preds;
    SmallVector<BasicBlock *, 4> PredBBs;
    SmallVector<jeandle::ObjectID, 8> IDs;

    // References to the Analyzer's per-block state. They alias the member
    // SLOTS, so reassigning e.g. CurrentState (the reset/clear retry) writes
    // through to the Analyzer member; clear()/insert() likewise mutate it.
    jeandle::PEABlockState &CurrentState;
    DenseMap<jeandle::ObjectID, DenseMap<int64_t, jeandle::FieldValue>>
        &FieldStates;
    DenseMap<jeandle::ObjectID, bool> &Eligible;
    DenseMap<jeandle::ObjectID, unsigned> &LockCounts;
    DenseMap<jeandle::ObjectID, SmallVector<LockEnter, 4>> &LiveLockEnters;
    DenseSet<jeandle::ObjectID> &Materialized;
    jeandle::AliasMap &Aliases;
    jeandle::PEAResult &Result;
    DenseMap<BasicBlock *, jeandle::EffectList> &PendingMergePhis;
    // Graal mergeEffects buffer: this merge's deferred CreatePHI effects,
    // retry-cleared, committed to PendingMergePhis[BB] after convergence.
    jeandle::EffectList MergeEffects;

    void intersectVirtualObjects();
    bool mergeObjectState(jeandle::ObjectID ID);
    bool mergeFieldStates(jeandle::ObjectID ID);
    bool materializeAndBuildPhi(jeandle::ObjectID ID);
  };

  // Returns a stable PHI for the given (in-loop merge block, ID, offset)
  // tuple, creating one (and registering it in OwnedLoopFieldPhis) on first
  // use. Falls back to createUnparentedPhi (i.e. the legacy OwnedPhis path)
  // when BB is not inside any loop (LI.getLoopFor(BB) == nullptr). Inside a
  // loop - including non-header in-loop merge blocks - the PHI must be cached
  // so its Value* survives restoreLoopSnapshot (which preserves BlockExits[BB]
  // for loop blocks but truncates OwnedPhis).
  PHINode *getOrCreateLoopFieldPhi(BasicBlock *BB, jeandle::ObjectID ID,
                                   int64_t Offset, Type *Ty, unsigned N,
                                   const Twine &Name);

  // Returns the per-pred materialization placeholder for (PH, TargetMerge, ID):
  // a stable, distinct, analysis-owned Value* that stands in for the per-pred
  // NewInv the transform creates. Cached by {PH, M, ID} (M=null for Case-A /
  // global) so it is loop-stable. Created as an unparented instruction
  // registered in Result.OwnedMatPlaceholders.
  Value *getOrCreatePerPredMatPlaceholder(BasicBlock *PH,
                                          BasicBlock *TargetMerge,
                                          jeandle::ObjectID ID);

  void processBlock(BasicBlock *BB);
  void processInstruction(Instruction *I);

  // Per-block state helpers.
  void resetPerBlockState();
  void inheritFromExit(const BlockExitData &Exit);
  void mergeStates(BasicBlock *BB);
  void snapshotExitState(BasicBlock *BB);
  // Mirror of snapshotExitState that writes the per-object snapshot into
  // an arbitrary BlockExitData (rather than into BlockExits[BB]). Used by
  // processBlock to capture the pre-invoke state for the unwind variant.
  void snapshotExitStateInto(BlockExitData &Data);
  // Out receives this block's deferred CreatePHI effects. For the entry /
  // single-pred paths it is PendingMergePhis[BB] (drained at end of
  // processBlock); for a merge it is the MergeProcessor's retry-cleared
  // MergeEffects buffer (committed to PendingMergePhis[BB] after the fixpoint
  // converges) — matching Graal's separation of mergeEffects from the
  // blockEffects committed only once the merge stabilizes.
  void processBlockPhis(BasicBlock *BB, jeandle::EffectList &Out);
  // Truncate Result.OwnedPhis / OwnedInsts to the given marks, deleting any
  // unparented PHIs/insts added since (the MergeProcessor retry discards a
  // failed iteration's fresh non-loop-header PHIs). OwnedLoopFieldPhis
  // (loop-header-cached) is intentionally untouched.
  void deleteOwnedSince(size_t PhiMark, size_t InstMark);

  // Resolve the per-pred BlockExitData a successor block should
  // inherit. Returns nullptr when the predecessor's contribution is dead
  // for this successor (either the pred has not been processed yet OR the
  // pred's terminator is an InvokeInst whose unwind edge was killed
  // because the invoke was virtualized away). Otherwise returns either
  // the pred's normal exit data (base data) or — when the successor is
  // the pred's invoke's unwind dest AND a pre-invoke snapshot was
  // recorded — the pre-invoke unwind variant.
  BlockExitData *exitDataFor(BasicBlock *Pred, BasicBlock *Succ);

  // Has a per-pred / Case-A materialize already been emitted for (Pred, M, ID)?
  // Used by processBlockPhis Case-A to skip a duplicate mat when the VO was
  // already per-pred-mat'd for THIS merge (per-pred does not flip the shared
  // state, so the Virtuals check alone would miss it). M=null queries the
  // Case-A / global slot.
  bool isMaterializedAtPred(BasicBlock *Pred, BasicBlock *M,
                            jeandle::ObjectID ID);

  // PHI Case C: synthesize a merged VirtualObject when every incoming of a
  // pointer-PHI resolves to a DIFFERENT but COMPATIBLE virtual ID. Returns
  // true on success (the new VO is registered, the PHI is aliased to it, and
  // per-entry CreatePHI effects are emitted); false if compatibility,
  // identity, or per-entry type checks fail — caller falls through to the
  // Case A per-pred materialize path. Restricted to non-self-reference,
  // non-byte-array, non-two-slot entry cases.
  bool synthesizeCaseC(BasicBlock *BB, PHINode *Phi,
                       ArrayRef<std::optional<jeandle::ObjectID>> InIDs,
                       jeandle::EffectList &Out);

  // Loop-header cache for Case C — keyed on (mergeBB, sorted-source-IDs).
  // The cache exists so an iterative merge stabilization re-visiting a
  // loop header doesn't synthesize a fresh VO every iteration (otherwise
  // VirtualObjects grows unboundedly and the fixpoint never closes). Cache
  // value is the synthesized VO id; the caller looks it up and reuses the
  // existing VO + alias rather than calling createVirtualObject.
  //
  // Under processLoop's body fixpoint the cache hits on iter >= 1: it
  // survives across iterations (not snapshotted by take/restoreLoopSnapshot)
  // so the same synthetic VO ID is reused at the header. Combined with
  // LoopFieldPhiCache (stable per-offset PHI shells), this keeps FieldStates
  // structurally equal across iterations, which the single-state B-vs-B'
  // convergence check requires.
  struct CaseCKey {
    BasicBlock *Block;
    SmallVector<jeandle::ObjectID, 4> SourceIDs;
    bool operator==(const CaseCKey &O) const {
      return Block == O.Block && SourceIDs == O.SourceIDs;
    }
  };
  struct CaseCKeyHash {
    size_t operator()(const CaseCKey &K) const {
      hash_code H = hash_value(K.Block);
      for (jeandle::ObjectID ID : K.SourceIDs)
        H = hash_combine(H, ID);
      return static_cast<size_t>(H);
    }
  };
  std::unordered_map<CaseCKey, jeandle::ObjectID, CaseCKeyHash> CaseCVOCache;

  PHINode *createUnparentedPhi(Type *Ty, unsigned N, const Twine &Name);

  // Produce a Value* of type LoadTy semantically equal to V, possibly
  // synthesizing an unparented coercion instruction (registered with
  // Result.OwnedInsts) that the transform's ReplaceLoad handler will splice
  // in immediately before the load. Returns V unchanged if no coercion is
  // needed, or nullptr if no safe coercion exists; callers should bail to
  // ineligible in the latter case. InsertContext is the load whose DebugLoc
  // (if any) is propagated onto the synthesized cast.
  //
  // Precondition: the load reads a WHOLE stored slot. The caller (processLoad)
  // bails to materialization for any sub-slot / partial-field read before
  // calling, so this routine only handles same-slot type reinterprets.
  Value *coerceToType(Value *V, Type *LoadTy, Instruction *InsertContext);

  // Why a materialization was emitted. Used to bump the
  // per-reason Statistic counter at the emission site. Cascade / Nested /
  // Unknown are not surfaced as standalone counters — they roll into the
  // total `JeandlePEAMaterialized` only.
  enum class MatReason : uint8_t {
    Unhandled, // unhandled escape-point instruction (generic
               // "virtualize returned false" path).
    Merge,     // state merge: mixed virtual/materialized at a multi-pred
               // entry, or lock-count cascade at merge.
    LoopExit,  // force-drain at a loop preheader.
    Phi,       // Case-A LLVM PHI fallback or Case-C synthetic-VO cascade.
    Cascade,   // sibling-virtual cascade triggered by strict lock order.
    Nested,    // recursive prerequisite materialization for an inner VO.
  };

  // Loop-soundness helpers.
  // Runs the preheader-drain ONLY on loops absent from
  // VisitedLoops (those never reached by processLoop). All other loops
  // are handled by processLoop directly: convergence (no drain needed),
  // pessimistic fallback (already drained inline), or
  // the overflow-recovery path which calls processStateBeforeLoopOnOverflow.
  void materializePreheaderVirtualsForUnvisitedLoops();

  // Drain every still-virtual VO at the loop preheader's
  // terminator. Used by the overflow-recovery path to forcibly demote any
  // outer VO that an inner MATERIALIZE_ALL iteration tried to touch, so
  // the outer fixpoint can re-run cleanly. Idempotent; safe to call
  // multiple times.
  void processStateBeforeLoopOnOverflow(Loop *L);

  // At every exit block of L, force-materialise still-virtual VOs
  // when the exit successor is an EH-pad block (landingpad / catchpad /
  // cleanuppad), to keep exception handlers from observing a
  // partially-materialised state. Deopt bundles are modeled separately at
  // their exact call sites; this loop-exit drain is only for EH visibility.
  // Called from processLoop after convergence.
  void processLoopExit(Loop *L);
  // SkipGlobalRAUW=true marks the emitted Materialize as IsPerPred, attaching a
  // per-pred placeholder so its OrigAlloc->NewInv mapping is recorded
  // distinctly per predecessor (MatPerBlock) — used by the lock-cascade path,
  // where multiple per-pred materializations of the same OrigAlloc coexist and
  // each CreatePHI incoming must resolve to its own pred's NewInv, not a single
  // shared def.
  void materializeAtPredFromExitInfo(jeandle::ObjectID ID, BasicBlock *PH,
                                     BlockExitData &ExitInfo,
                                     bool SkipGlobalRAUW = false,
                                     MatReason Reason = MatReason::Merge,
                                     BasicBlock *TargetMerge = nullptr);

  // Real loop fixpoint. processLoop runs the fixpoint over L (which
  // includes its sub-loops; nested loops are dispatched recursively when
  // their header is encountered in the RPO walk). On convergence, all blocks
  // of L have their BlockExits populated and the outer RPO walk continues.
  // On failure, restores to the pre-loop snapshot, then runs the legacy
  // pessimistic path: force-materialize at preheader (drains every virtual)
  // and process the body once in MATERIALIZE_ALL mode (no new VOs created
  // inside).
  void processLoop(Loop *L, ArrayRef<BasicBlock *> FunctionRPO);

  // Helpers used exclusively by processLoop.
  // Loop blocks of L (including its sub-loops) in function-RPO order,
  // computed once per processLoop and reused across the inner fixpoint body
  // passes (the loop's CFG is stable across the fixpoint).
  SmallVector<BasicBlock *, 32>
  loopBlocksInRPO(Loop *L, ArrayRef<BasicBlock *> FunctionRPO);
  void processLoopBodyOnePass(Loop *L, ArrayRef<BasicBlock *> LoopRPO,
                              ArrayRef<BasicBlock *> FunctionRPO);

  // The per-iteration snapshot. All members are independently restorable.
  //
  // Eligible MUST be rolled back across loop-fixpoint iterations.
  // Numerous in-body paths (mergeStates' incompatible-merge bail, store
  // overlap, atomic / cmpxchg / memcpy fallbacks, etc.) mutate
  // Eligible[ID] = false unconditionally. Without snapshotting, a transient
  // escape pattern in iter N (e.g. an as-yet-unpopulated empty back-edge
  // exit) would permanently wedge the VO ineligible for iter N+1, even
  // though iter N+1's fully-populated back-edge would otherwise allow
  // re-virtualisation. Each iteration's per-VO ineligibility flips are
  // local to that iteration and only persist if the converged exit state
  // requires them.
  struct LoopSnapshot {
    jeandle::PEABlockState CurrentState;
    jeandle::AliasMap Aliases;
    DenseMap<jeandle::ObjectID, DenseMap<int64_t, jeandle::FieldValue>>
        FieldStates;
    DenseMap<jeandle::ObjectID, unsigned> LockCounts;
    DenseMap<jeandle::ObjectID, SmallVector<LockEnter, 4>> LiveLockEnters;
    DenseSet<jeandle::ObjectID> Materialized;
    DenseMap<jeandle::ObjectID, bool> EligibleSnapshot;
    // Number of VOs that existed BEFORE this iteration. IDs greater
    // than or equal to this index were created within the rolled-back iter;
    // restoreLoopSnapshot re-marks them Eligible[ID]=true so the next iter's
    // AllocSiteToVO cache-hit path (which bails on !Eligible.lookup) sees a
    // re-eligible VO and proceeds. Without this, body-local allocations
    // created inside the loop would be wedged ineligible across every
    // iteration after the first.
    size_t PreIterVOCount = 0;
    uint32_t NextSeqNo = 0;
    size_t OwnedPhisSize = 0;
    size_t OwnedInstsSize = 0;

    // For each block in L, the prior BlockExits[BB] (if any) and
    // BlockEffects[BB] (if any), captured *before* the loop iteration began.
    DenseMap<BasicBlock *, BlockExitInfo> SavedBlockExits;
    DenseMap<BasicBlock *, jeandle::EffectList> SavedBlockEffects;
    DenseMap<BasicBlock *, DenseMap<BasicBlock *, DenseSet<jeandle::ObjectID>>>
        SavedMaterializedAtPred;
    DenseSet<BasicBlock *> HadBlockExits;
    DenseSet<BasicBlock *> HadBlockEffects;
    DenseSet<BasicBlock *> HadMaterializedAtPred;
  };

  void takeLoopSnapshot(Loop *L,
                        const llvm::SmallPtrSetImpl<BasicBlock *> &LoopBlocks,
                        LoopSnapshot &S);
  void
  restoreLoopSnapshot(const llvm::SmallPtrSetImpl<BasicBlock *> &LoopBlocks,
                      const LoopSnapshot &S);

  // Structural equivalence of the BlockExitData base (the per-object
  // book-keeping). The loop fixpoint's single-state B-vs-B' convergence test
  // (paper §5.2.5 / Graal EClosure:472 equivalentTo).
  static bool exitDataEquivalent(const BlockExitData &A,
                                 const BlockExitData &B);

  void processAllocation(CallBase *CB);
  // Returns true iff the store was consumed as a virtualised store
  // (pointer side resolved to a virtual base, regardless of whether the
  // value side made the VO ineligible). Returns false when the pointer
  // operand is NOT a virtual ref; the caller then falls through to the
  // generic hasVirtualInputs / materializeAllVirtualOperands path so a
  // VALUE-side virtual is not silently leaked to a global / non-virtual
  // pointer (e.g. `store ptr %virt, ptr @G`).
  bool processStore(StoreInst *SI);
  void processLoad(LoadInst *LI);
  // Resolve a load/store pointer to a byte offset within the virtual object
  // BaseID, applying the array-element GEP fast path, the general
  // constant-offset resolver, and the header-offset guard (instance AND
  // array). Returns nullopt (caller marks BaseID ineligible) when the access
  // cannot be virtualized: symbolic array index, non-constant GEP offset,
  // out-of-bounds index, or a header (mark/klass) field access. Graal
  // correspondence: the constant-offset + entryIndex resolution shared by
  // LoadFieldNode/StoreFieldNode/LoadIndexedNode/StoreIndexedNode/RawLoad/
  // RawStoreNode.virtualize.
  std::optional<int64_t> resolveAccess(Value *Ptr, jeandle::ObjectID BaseID);
  // TODO(unsafe-inliner): processAtomicRMW / processCmpXchg (re-add with the
  // jeandle-jdk frontend inliner for Unsafe atomic intrinsics),
  // processArrayCopy (System.arraycopy → llvm.memcpy/memmove), processMemSet
  // (Arrays.fill → llvm.memset). Until then these shapes fall through to
  // conservative materialization.
  bool processJavaOp(CallBase *CB);
  void recordDeoptBundleState(CallBase *CB);
  // Known non-escaping LLVM intrinsics (assume, lifetime/invariant
  // markers, debug, annotations, branch hints, ...) are no-ops for PEA;
  // launder/strip.invariant.group forward the argument's virtual alias.
  // Returns true if the intrinsic was handled (no-op or alias-forwarded),
  // false to fall through to the ICmp/JavaOp/generic-escape path.
  bool processIntrinsic(llvm::IntrinsicInst *II);
  // Fold an equality icmp against a virtual pointer to a constant (virtuals
  // are non-null by construction; identity comparison). Returns true if
  // folded, false to fall through to materialization.
  bool foldICmpEquality(llvm::ICmpInst *ICmp);
  bool foldArrayLength(CallBase *CB);
  bool foldLoadKlass(CallBase *CB);
  bool foldCheckCast(CallBase *CB);
  bool foldInstanceOf(CallBase *CB);
  bool foldMonitorEnter(CallBase *CB);
  bool foldMonitorExit(CallBase *CB);
  // Resolve the bytecode lock depth of a monitorenter call site: the FIRST
  // NextLockEnterOrder value cached in LockDepthCache (stable across
  // loop-fixpoint iterations). Shared by foldMonitorEnter and
  // materializeVirtualLocksBefore.
  uint32_t getOrCreateLockDepth(CallBase *CB);
  // Graal materializeVirtualLocksBefore (PartialEscapeClosure.java:641-652):
  // before a REAL (non-virtualized) monitorenter whose depth is D, materialize
  // every still-virtual VO holding an elided lock with a strictly shallower
  // min depth. Fired from processInstruction on the not-deleted monitorenter
  // branch (Graal PClosure:263-264), distinct from foldMonitorEnter's
  // elide-path pre-cascade.
  void materializeVirtualLocksBefore(CallBase *MonEnter);
  bool foldArrayStoreCheck(CallBase *CB);
  bool foldPostBarrier(CallBase *CB);
  bool foldCheckIfValueBased(CallBase *CB);
  bool foldRegisterFinalizerIfNeeded(CallBase *CB);
  // Common helper for checkcast/check_instanceof: returns the folded constant
  // bool (true/false) if the relationship is statically known, or
  // std::nullopt otherwise.
  std::optional<bool> evalSubtypeRelation(uintptr_t SubKlass,
                                          uintptr_t SuperKlass);
  void emitReplaceCall(CallBase *CB, Value *Replacement, jeandle::ObjectID ID);
  void materializeAllVirtualOperands(Instruction *I);
  // Sound analog of materializeAllVirtualOperands: materialize only when no
  // operand is a DERIVED pointer of a still-virtual object (a derived operand
  // computed before I would resolve to poison, since pea.mat at I cannot
  // dominate its def); otherwise mark every virtual operand's object ineligible
  // so derived pointers stay valid. Graal's processNodeInputs
  // (PartialEscapeClosure.java:433-451) always materializes — Jeandle diverges
  // to markIneligible only because LLVM represents field access as derived
  // address computations, which Graal's identity-based alias model does not.
  void materializeVirtualOperandsSafely(Instruction *I);
  // Mark every distinct virtual ObjectID among I's operands ineligible (each
  // keeps its ORIGINAL allocation real). The bail primitive backing
  // materializeVirtualOperandsSafely; same principle as the processStore/
  // processLoad bail (markIneligible rather than materialize, to keep a
  // pre-computed derived address valid).
  void markVirtualOperandsIneligible(Instruction *I);
  struct DeoptMaterializeAnchor {
    Instruction *InsertBefore = nullptr;
    CallBase *DeoptSource = nullptr;
  };
  bool instructionUsesAnyVirtualObject(Instruction *I);
  bool canHoistMaterializeToDeoptAnchor(Instruction *Anchor,
                                        Instruction *EscapePoint);
  std::optional<DeoptMaterializeAnchor>
  findDeoptMaterializeAnchor(Instruction *EscapePoint);
  void materializeAt(jeandle::ObjectID ID, Instruction *InsertBefore,
                     MatReason Reason = MatReason::Unhandled);

  // Graal analog: PartialEscapeClosure.ensureMaterialized(state, object, fixed,
  // effects, counter) -> materializeBefore -> materializeWithCommit. Jeandle's
  // per-block state is fragmented across several maps (vs Graal's single
  // PartialEscapeBlockState container), so the unified algorithm takes this
  // view: the operative state maps (the live analyzer maps for the escape-point
  // path, or a predecessor's BlockExitData maps for the merge-driven path), the
  // idempotency set, and callbacks for the per-path I/O. The two wrappers
  // (materializeAt / materializeAtPredFromExitInfo) each build a context and
  // delegate the shared cascade/lock-capture/prereq/dominance/emit/flip
  // algorithm here, localizing the ~8 genuine per-path differences in the
  // callbacks
  // instead of scattering `if (isPredPath)` through the body.
  struct MaterializeContext {
    DenseMap<jeandle::ObjectID, DenseMap<int64_t, jeandle::FieldValue>>
        &FieldStates;
    DenseMap<jeandle::ObjectID, unsigned> &LockCounts;
    DenseMap<jeandle::ObjectID, SmallVector<LockEnter, 4>> &LiveLockEnters;
    // Idempotency set: has this ObjectID already been materialized in this call
    // chain (cascade cycle-prevention) AND function-wide for this path? Bound
    // to the analyzer-wide `Materialized` (escape-point path) or to
    // `MaterializedAtPred[PH][TargetMerge]` (merge-driven path — a DenseSet<ID>
    // per (PH, target-merge) so distinct target merges each get their own
    // per-pred mat at the same PH, while Case-A/global M=null dedups across
    // merges sharing PH).
    DenseSet<jeandle::ObjectID> &MaterializedSet;
    MatReason Reason;
    // Recurse on a nested/cascade object (materializeAt vs
    // materializeAtPredFromExitInfo — different signatures, hence a callback).
    function_ref<void(jeandle::ObjectID, MatReason)> Recurse;
    // Clear the operative lock state after the surviving unbalanced enters are
    // captured for re-emit. Live also clears the ObjectState::Locks mirror;
    // pred clears the ExitInfo maps. Needed so commit()'s LockCounts!=0 gate
    // doesn't disqualify the VO.
    function_ref<void(jeandle::ObjectID)> ClearLockState;
    // Capture the surviving unbalanced enters into the Materialize effect's
    // Locks list (sorted ascending by BytecodeDepth), for re-emit at the
    // materialize point by applyMaterialize (Graal: synthetic MonitorEnterNodes
    // at the CommitAllocationNode).
    function_ref<void(ArrayRef<LockEnter>, jeandle::ObjectID,
                      jeandle::MaterializeEffect &E)>
        CaptureLocksIntoEffect;
    // Drop alias-map entries resolving to a just-materialized inner (live
    // only).
    function_ref<void(jeandle::ObjectID)> DropInnerAliases;
    // Compute the safe materialization insertion point.
    function_ref<Instruction *()> ComputeSafeIP;
    // Set the path-specific flags on an already-built Materialize effect (the
    // unified function builds the common fields + field snapshot, then calls
    // this to add IsPerPred / PerPredPlaceholder / DeoptBundleSource).
    function_ref<void(jeandle::MaterializeEffect &E, Instruction *SafeIP)>
        SetEffectFlags;
    // Flip the per-object state to materialized (live CurrentState vs
    // ExitInfo).
    function_ref<void(jeandle::ObjectID)> FlipState;
    // Resolve the materialized value to record when a field references a just-
    // materialized object Oid (field-replay). Live returns OrigAlloc (single
    // global-RAUW materialize); pred returns the per-pred placeholder (Graal's
    // distinct AllocatedObjectNode) so the field resolves to this pred's OWN
    // NewInv. Used by the recursive-prereq rewrite and the sibling sweeps.
    function_ref<Value *(jeandle::ObjectID)> MaterializedValue;
  };
  void ensureMaterialized(jeandle::ObjectID ID, MaterializeContext &C);
  void propagatePointerAlias(Instruction *I);

  // Walk every other VO's FieldStates in the supplied map and rewrite
  // any FieldValue::virtualRef(FlippedID) entry to MaterializedRef(NewPtr).
  // Called after a VO is flipped to Materialized so sibling VOs that hold a
  // reference to it learn about the materialized pointer immediately. Without
  // this sweep, a later store/load through a sibling's field that still
  // carries VirtualRef(FlippedID) would fire the transform-side assertion
  // ("VirtualRef field entries must have been rewritten to MaterializedRef
  // during analysis") in debug, or silently drop the field in release.
  void updateOtherStatesForMaterialized(
      jeandle::ObjectID FlippedID, Value *NewPtr,
      DenseMap<jeandle::ObjectID, DenseMap<int64_t, jeandle::FieldValue>> &FS);

  // Bump the appropriate per-reason Statistic. Always bumps the
  // total JeandlePEAMaterialized; the per-reason counter is bumped only for
  // the four reasons (Unhandled / Merge / LoopExit / PHI).
  static void bumpMaterializeStat(MatReason R);

  void validateAllocationEliminationPlans();
  void commit();
  void keepOriginalAllocation(jeandle::ObjectID ID);
  void dropEffectsFor(jeandle::ObjectID ID);

  // Mark a VO ineligible, taking the TRANSITIVE closure over synthetic Case-C
  // sources. A synthetic VO has no real backing allocation (AllocationCall is
  // borrowed), so dropping it alone would let its per-pred sources still be
  // eliminated, leaving the merge PHI's incomings as poison. A source can
  // itself be synthetic (Case C can nest), so the cascade walks the whole
  // synthetic-source DAG so every leaf real allocation survives in IR. The
  // relation is acyclic (a source is created at an earlier merge than the
  // synthetic referencing it), so the worklist terminates; the visited set is
  // pure cycle defense. The SyntheticPhi alias is reset for every synthetic so
  // downstream resolveVirtualRef stops folding through it. Safe on
  // non-synthetic VOs (degenerates to Eligible[ID] = false). Single source of
  // truth for the synthetic cascade; the mixed-merge bail and the two
  // materialization paths call through here.
  void markIneligible(jeandle::ObjectID ID);
};

void Analyzer::markIneligible(jeandle::ObjectID ID) {
  SmallVector<jeandle::ObjectID, 8> Worklist;
  DenseSet<jeandle::ObjectID> Visited;
  Worklist.push_back(ID);
  while (!Worklist.empty()) {
    jeandle::ObjectID Cur = Worklist.pop_back_val();
    if (!Visited.insert(Cur).second)
      continue;
    Eligible[Cur] = false;
    jeandle::VirtualObject &VObj = *Result.VirtualObjects[Cur];
    if (VObj.IsSynthetic) {
      if (VObj.SyntheticPhi)
        Aliases.resetAlias(VObj.SyntheticPhi);
      for (jeandle::ObjectID PID : VObj.SyntheticSourceIDs)
        Worklist.push_back(PID);
    }
  }
}

void Analyzer::processBlock(BasicBlock *BB) {
  // Rebuild per-block state from the predecessor snapshots before we walk
  // instructions. Entry block starts empty. Single-pred blocks inherit.
  // Multi-pred blocks run the merge (mergeStates also handles the degenerate
  // "no processed preds" case, which can happen e.g. on irreducible loop
  // headers where the back-edge pred hasn't been visited yet — we treat
  // those conservatively as starting empty).
  // Statically-unreachable blocks (constant-condition dead arms, blocks
  // unreachable from entry) are pruned by the pre-PEA SimplifyCFG pass in
  // buildJeandlePipeline, so we never see them here.

  resetPerBlockState();
  if (BB == &F.getEntryBlock()) {
    // Entry: nothing to inherit; per-block state is empty.
    processBlockPhis(BB, PendingMergePhis[BB]);
  } else if (BB->hasNPredecessors(1)) {
    BasicBlock *P = *predecessors(BB).begin();
    if (BlockExitData *ED = exitDataFor(P, BB))
      inheritFromExit(*ED);
    processBlockPhis(BB, PendingMergePhis[BB]);
  } else {
    // mergeStates wraps BOTH the per-VO loop AND the PHI loop in a
    // single do-while so a Case-C synthesis or Case-A fallback in
    // processBlockPhis that materialises an inner VO at a pred can
    // reawaken the per-VO decisions (which depended on the now-stale
    // pred-side virtuality). mergeStates calls processBlockPhis itself.
    mergeStates(BB);
  }

  // Exception edge state splitting. If the block ends in an InvokeInst,
  // snapshot the per-object state immediately BEFORE applying the invoke.
  // The post-invoke state (the regular snapshotExitState below) is what
  // the normal successor inherits; the snapshot we take here becomes the
  // unwind successor's inheritance (the materialize logically happens
  // during the call, so the handler should not see partially-materialized
  // state).
  //
  // We only bother when the function has a personality (no personality =>
  // no real exception handlers; the work would be observably inert) and
  // the terminator actually IS an InvokeInst. Both conditions are required
  // for state-splitting to matter.
  InvokeInst *TermII = dyn_cast<InvokeInst>(BB->getTerminator());
  bool MaybeSplit = TermII && F.hasPersonalityFn();
  std::optional<BlockExitData> PreInvokeSnapshot;
  // Drain any VOs registered under MATERIALIZE_ALL in this
  // block, materialising each at the terminator IP. Called immediately
  // before the terminator instruction is processed so the materialise
  // takes effect BEFORE PreInvokeSnapshot (so both normal and unwind
  // successors inherit a Materialized-VO view, never a stale Virtual
  // view that would dangling-ref an OrigAlloc that the transform RAUW'd
  // away). Idempotent (clears the per-block list).
  auto drainMaterializeAll = [&]() {
    auto PMIt = PendingMaterializeAllVOs.find(BB);
    if (PMIt == PendingMaterializeAllVOs.end())
      return;
    SmallVector<jeandle::ObjectID, 4> ToMat = std::move(PMIt->second);
    PendingMaterializeAllVOs.erase(PMIt);
    llvm::sort(ToMat); // deterministic order
    Instruction *IP = BB->getTerminator();
    for (jeandle::ObjectID ID : ToMat)
      materializeAt(ID, IP, MatReason::Unhandled);
  };

  for (Instruction &I : *BB) {
    if (&I == BB->getTerminator())
      drainMaterializeAll();
    if (MaybeSplit && &I == TermII) {
      PreInvokeSnapshot.emplace();
      snapshotExitStateInto(*PreInvokeSnapshot);
    }
    processInstruction(&I);
  }
  // Defensive: empty blocks or blocks whose terminator was never enqueued
  // through the loop above still need a drain (the foreach loop above does
  // the right thing for any non-empty block, but a future change might
  // skip-over the terminator).
  drainMaterializeAll();

  snapshotExitState(BB);

  if (TermII) {
    BlockExitInfo &Info = BlockExits[BB];
    Info.TerminatorInvoke = TermII;
    Info.UnwindDest = TermII->getUnwindDest();

    // Detect whether the terminator-invoke was virtualized away by the
    // analyzer (e.g. processJavaOp emitted a ReplaceCall effect on it).
    // The transform rewrites such invokes as an unconditional branch to
    // the normal successor, so from PEA's perspective the unwind edge no
    // longer exists. Mark the edge killed so exitDataFor reports no
    // contribution to the unwind successor.
    bool Virtualized = false;
    auto EIt = Result.BlockEffects.find(BB);
    if (EIt != Result.BlockEffects.end()) {
      for (const auto &E : EIt->second) {
        if (E.getTarget() == TermII &&
            E.getKind() == jeandle::Effect::Kind::ReplaceCall) {
          Virtualized = true;
          break;
        }
      }
    }
    Info.UnwindEdgeKilled = Virtualized;

    // Only stash the pre-invoke snapshot if (a) we actually took one
    // (function has a personality), (b) the invoke wasn't virtualized
    // (the kill flag already handles that case), AND (c) the snapshot
    // actually differs from the post-invoke base data. The last gate
    // avoids paying the DenseMap-copy cost (and the convergence-check
    // cost downstream) for invokes that triggered no PEA state change.
    if (!Virtualized && PreInvokeSnapshot &&
        !exitDataEquivalent(Info, *PreInvokeSnapshot))
      Info.UnwindData = std::move(PreInvokeSnapshot);
  }

  // Drain deferred CreatePHI effects emitted during this block's
  // mergeStates / processBlockPhis / synthesizeCaseC, assigning each a
  // FRESH SeqNo at drain time so it sorts strictly AFTER any per-pred
  // Materialize emitted during the instruction walk above. Cross-block
  // ordering is unchanged because the per-pred Materialize lives in a
  // different BlockEffects bucket from the CreatePHI; the SeqNo assigned
  // here is irrelevant to cross-block ordering (RPO drives that), but
  // consistent across runs because nextSeqNo is monotonic within a single
  // Analyzer::run.
  auto It = PendingMergePhis.find(BB);
  if (It != PendingMergePhis.end()) {
    jeandle::EffectList &Phis = It->second;
    for (auto &PE : Phis)
      PE.SeqNo = Result.nextSeqNo();
    while (!Phis.empty())
      Result.addBlockEffect(Phis.spliceOut(0));
    PendingMergePhis.erase(It);
  }
}

void Analyzer::resetPerBlockState() {
  CurrentState = jeandle::PEABlockState();
  FieldStates.clear();
  LockCounts.clear();
  LiveLockEnters.clear();
  Materialized.clear();
}

BlockExitData *Analyzer::exitDataFor(BasicBlock *Pred, BasicBlock *Succ) {
  auto It = BlockExits.find(Pred);
  if (It == BlockExits.end())
    return nullptr;
  BlockExitInfo &Info = It->second;
  // When the pred ends in an InvokeInst whose unwind dest is `Succ`,
  // the unwind edge participates in state-splitting.
  if (Info.TerminatorInvoke && Info.UnwindDest == Succ) {
    if (Info.UnwindEdgeKilled) {
      // Invoke was virtualized; the handler is unreachable for analysis.
      // Return nullptr so the merge consumer drops this pred entirely.
      return nullptr;
    }
    if (Info.UnwindData)
      return &*Info.UnwindData;
  }
  // Normal successor (or unwind successor with no state-split needed).
  return &Info;
}

bool Analyzer::isMaterializedAtPred(BasicBlock *Pred, BasicBlock *M,
                                    jeandle::ObjectID ID) {
  auto It = MaterializedAtPred.find(Pred);
  if (It == MaterializedAtPred.end())
    return false;
  auto MIt = It->second.find(M);
  if (MIt == It->second.end())
    return false;
  return MIt->second.count(ID);
}

void Analyzer::inheritFromExit(const BlockExitData &Exit) {
  for (jeandle::ObjectID ID : Exit.Virtuals) {
    if (!Eligible.lookup(ID))
      continue;
    CurrentState.addObject(ID, jeandle::ObjectState());
  }
  for (jeandle::ObjectID ID : Exit.Materialized) {
    if (!Eligible.lookup(ID))
      continue;
    jeandle::VirtualObject &VObj = *Result.VirtualObjects[ID];
    jeandle::ObjectState OS;
    // Prefer the snapshot's MaterializedValues entry (e.g. a merge-block
    // PHI synthesized by an earlier mergeStates) over the OrigAlloc fallback.
    Value *MV = nullptr;
    auto MIt = Exit.MaterializedValues.find(ID);
    if (MIt != Exit.MaterializedValues.end())
      MV = MIt->second;
    if (!MV)
      MV = VObj.AllocationCall;
    OS.escape(MV);
    CurrentState.addObject(ID, std::move(OS));
    Materialized.insert(ID);
  }
  for (auto &Kv : Exit.FieldStates) {
    if (!Eligible.lookup(Kv.first))
      continue;
    FieldStates[Kv.first] = Kv.second;
  }
  for (auto &Kv : Exit.LockCounts) {
    if (!Eligible.lookup(Kv.first))
      continue;
    if (Kv.second != 0)
      LockCounts[Kv.first] = Kv.second;
  }
  // Inherit the live monitorenter stack alongside LockCounts. The
  // CallBase* identities are still valid (we never erase IR during analysis).
  for (auto &Kv : Exit.LiveLockEnters) {
    if (!Eligible.lookup(Kv.first))
      continue;
    if (!Kv.second.empty())
      LiveLockEnters[Kv.first] = Kv.second;
  }
  // Mirror the inherited live stack onto each per-VO ObjectState's
  // Locks so the on-VO view matches the analyzer-side DenseMap from the
  // first instruction of this block. Without this, ObjectState::Locks would
  // be empty for an inherited VO until the first foldMonitorEnter in this
  // block, and a downstream equivalentTo comparison (e.g. merge fast path)
  // would over-collapse two VOs with structurally different lock states.
  for (auto &Kv : Exit.LiveLockEnters) {
    if (!Eligible.lookup(Kv.first))
      continue;
    if (!CurrentState.hasObjectState(Kv.first))
      continue;
    jeandle::ObjectState &OS =
        CurrentState.getObjectStateForModification(Kv.first);
    if (!OS.isVirtual())
      continue;
    for (const LockEnter &LE : Kv.second)
      OS.addLock({LE.Call, LE.BytecodeDepth});
  }
}

PHINode *Analyzer::createUnparentedPhi(Type *Ty, unsigned N,
                                       const Twine &Name) {
  PHINode *Phi = PHINode::Create(Ty, N, Name);
  Result.OwnedPhis.emplace_back(Phi);
  return Phi;
}

Value *Analyzer::getOrCreatePerPredMatPlaceholder(BasicBlock *PH,
                                                  BasicBlock *TargetMerge,
                                                  jeandle::ObjectID ID) {
  PerPredPlaceholderKey Key{PH, TargetMerge, ID};
  auto It = PerPredMatPlaceholderCache.find(Key);
  if (It != PerPredMatPlaceholderCache.end())
    if (Value *V = It->second)
      return V;
  // Create a fresh unparented placeholder of the Java-heap pointer type. It
  // is never inserted into IR; the transform resolves it away via
  // MatPerBlock[{Pred, placeholder}] / NewAllocFor[placeholder]. A PHINode is
  // used purely as a concrete distinct Value* (resolution keys on pointer
  // identity) — it is NOT a real merge PHI, and lives in OwnedMatPlaceholders
  // (separate from OwnedPhis/OwnedLoopFieldPhis) so it is never confused with
  // one. Keyed by {PH, TargetMerge, ID} the same pointer is returned across
  // loop-fixpoint iterations (per-merge), preserving the B-vs-B' convergence.
  // TargetMerge is null for the Case-A / global path (one shared placeholder
  // per (PH, ID)).
  Type *PtrTy =
      PointerType::get(F.getContext(), jeandle::AddrSpace::JavaHeapAddrSpace);
  PHINode *Placeholder = PHINode::Create(PtrTy, 0, "pea.perpred");
  Result.OwnedMatPlaceholders.emplace_back(Placeholder);
  PerPredMatPlaceholderCache[Key] = Placeholder;
  Result.PerPredMatPlaceholders.insert(Placeholder);
  return Placeholder;
}

PHINode *Analyzer::getOrCreateLoopFieldPhi(BasicBlock *BB, jeandle::ObjectID ID,
                                           int64_t Offset, Type *Ty, unsigned N,
                                           const Twine &Name) {
  // Outside any loop (LI.getLoopFor(BB) == nullptr), fall back to the legacy
  // single-shot OwnedPhis path. Inside a loop - including NON-HEADER in-loop
  // merge blocks - the PHI is cached so its Value* stays stable across
  // fixpoint iterations: restoreLoopSnapshot preserves BlockExits[BB] for
  // every loop block (so the next iteration's in-pass mergeStates(Header)
  // can read the back-edge pred's exit state), and any Value* reachable from
  // a preserved BlockExits[BB] must therefore survive rollback. Cached PHIs
  // live in OwnedLoopFieldPhis, which restoreLoopSnapshot does NOT pop
  // (unlike OwnedPhis); were a non-header in-loop merge to bypass the cache,
  // its PHI would land in OwnedPhis and be deleted on rollback while the
  // preserved BlockExits[BB] still references it -> dangling Value*.
  if (!LI.getLoopFor(BB))
    return createUnparentedPhi(Ty, N, Name);

  // At an in-loop BB, size the PHI shell for the block's FULL
  // fan-in (every predecessor — forward edge AND back edge), regardless of
  // how many incomings the caller has visited on this particular iteration.
  // Without this, iter 1 (which only sees forward-edge preds before the
  // body has been walked) reserves capacity 1; iter 2's caller hands N=2
  // and PHINode::reserveOperand has to grow the operand list — wasted work
  // and (defensively) avoids a stale-capacity surprise if the IR has more
  // preds than the analyzer thinks. BB->getNumPredecessors() observes ALL
  // preds even on first-pass analysis.
  unsigned FullN =
      static_cast<unsigned>(std::distance(pred_begin(BB), pred_end(BB)));
  if (FullN < N)
    FullN = N; // caller knows of more than IR shows; should not occur on
               // well-formed IR but cheap to honour.

  LoopPhiKey K{BB, ID, Offset};
  auto It = LoopFieldPhiCache.find(K);
  if (It != LoopFieldPhiCache.end()) {
    PHINode *Cached = It->second;
    // Defensive: WeakTrackingVH would auto-null if the PHI was deleted via
    // some other code path. The cache entry is plain PHINode*, so revalidate.
    bool Valid = false;
    for (auto &VH : Result.OwnedLoopFieldPhis) {
      if (VH == Cached) {
        Valid = true;
        break;
      }
    }
    // Validity check. During analysis the PHI is an unparented shell — the
    // transform pass is responsible for inserting it into the merge block and
    // calling addIncoming. Therefore a healthy cached PHI has
    // getNumIncomingValues() == 0 at every analysis-time touch. Any non-zero
    // count indicates a leak (e.g. an earlier code path mistakenly called
    // addIncoming on the shell) and we drop the cache entry to force
    // re-creation.
    if (Valid && Cached->getType() == Ty &&
        Cached->getNumIncomingValues() == 0) {
      // On cache hit, defensively clear any operands the shell
      // might have accumulated. Today's caller-driven incoming-list
      // construction passes Value*/BasicBlock* through the CreatePHI
      // Effect's PHIIncomingValues / PHIIncomingBlocks lists — never via
      // direct Phi->addIncoming() during analysis — so the loop below is
      // a no-op in practice. It's spelled out anyway to keep the
      // post-condition explicit ("cache hit returns an empty shell").
      while (Cached->getNumIncomingValues() > 0)
        Cached->removeIncomingValue(Cached->getNumIncomingValues() - 1u,
                                    /*DeletePHIIfEmpty=*/false);
      return Cached;
    }
    // Cache stale (rare): drop entry and fall through to recreate.
    LoopFieldPhiCache.erase(It);
  }

  PHINode *Phi = PHINode::Create(Ty, FullN, Name);
  Result.OwnedLoopFieldPhis.emplace_back(Phi);
  LoopFieldPhiCache[K] = Phi;
  return Phi;
}

// Type-coercion for processLoad.
//
// Handles loads that read a WHOLE stored slot, possibly reinterpreting its
// type: the trivial same-type return, a same-bit-width primitive↔primitive
// BitCast (Float↔Int, Half↔i16, etc.), and pointer↔pointer passthrough.
//
// The caller, processLoad, rejects sub-slot ("incomplete field") reads — a load
// whose within-slot byte offset is nonzero — before reaching this routine,
// forcing the object to materialize instead. Widening loads (EntryWidth <
// LoadWidth), narrower whole-slot loads (EntryWidth > LoadWidth), and
// pointer↔primitive mismatches (stable-slot-kind invariant) bail here.
// TODO(unsafe-inliner): see the access dispatch (processStore/processLoad).
//
// The same-bit-width bitcast is registered (unparented) in Result.OwnedInsts;
// the transform's ReplaceLoad handler splices it before the target load.
Value *Analyzer::coerceToType(Value *V, Type *LoadTy,
                              Instruction *InsertContext) {
  Type *VTy = V->getType();
  if (VTy == LoadTy)
    return V;
  if (!VTy->isSized() || !LoadTy->isSized())
    return nullptr;
  uint64_t VBits = DL.getTypeSizeInBits(VTy);
  uint64_t LBits = DL.getTypeSizeInBits(LoadTy);

  // Stable-slot-kind invariant: ref↔primitive at the same slot must
  // materialize.
  if (VTy->isPointerTy() != LoadTy->isPointerTy())
    return nullptr;

  // Pointer↔pointer: pointers don't truncate. Require matching bit width and
  // address spaces. Same-AS same-bitwidth pointers are already type-identical
  // under opaque pointers, so a true pointer coercion is rare; defend against
  // the cross-AS case.
  if (VTy->isPointerTy() && LoadTy->isPointerTy()) {
    if (VBits != LBits)
      return nullptr;
    if (VTy->getPointerAddressSpace() != LoadTy->getPointerAddressSpace())
      return nullptr;
    return V;
  }

  // Both primitives from here. Sub-byte loads (e.g. i1) are bailed; supporting
  // them would need bit-granular shift/mask which we don't need today.
  if (LBits == 0 || (LBits % 8) != 0)
    return nullptr;
  if (VBits == 0 || (VBits % 8) != 0)
    return nullptr;

  // Same bit width → BitCast (Float↔Int, Half↔i16, etc.).
  if (VBits == LBits) {
    if (!CastInst::isBitCastable(VTy, LoadTy))
      return nullptr;
    Instruction *Cast =
        CastInst::Create(Instruction::BitCast, V, LoadTy, "pea.coerce",
                         /*InsertBefore=*/nullptr);
    if (InsertContext)
      Cast->setDebugLoc(InsertContext->getDebugLoc());
    Result.OwnedInsts.emplace_back(Cast);
    return Cast;
  }

  // Anything else bails: a narrower whole-slot load (EntryWidth > LoadWidth),
  // a widening load (EntryWidth < LoadWidth), and any other cross-width
  // mismatch. The caller marks the VO ineligible so the original
  // alloc/store/load survive in IR.
  return nullptr;
}

Analyzer::MergeProcessor::MergeProcessor(Analyzer &A, BasicBlock *BB)
    : A(A), BB(BB), CurrentState(A.CurrentState), FieldStates(A.FieldStates),
      Eligible(A.Eligible), LockCounts(A.LockCounts),
      LiveLockEnters(A.LiveLockEnters), Materialized(A.Materialized),
      Aliases(A.Aliases), Result(A.Result),
      PendingMergePhis(A.PendingMergePhis) {}

void Analyzer::mergeStates(BasicBlock *BB) {
  MergeProcessor MP(*this, BB);
  MP.run();
}

// Graal merge() + doMergeWithoutDead(): collect predecessor snapshots, apply
// the degenerate fast paths, intersect the tracked IDs, then drive the do/while
// merge fixpoint.
void Analyzer::MergeProcessor::run() {
  // Collect the snapshots of every predecessor we've already processed. RPO
  // guarantees forward-edge preds are visited first; back-edge preds are not
  // yet available and are silently skipped (the loop-preheader force-
  // materialization sweep handles loop soundness). Preds[i] points at the
  // shared `BlockExits[P]` (or its UnwindData variant) via exitDataFor;
  // per-pred materialize does not flip this state (see
  // materializeAtPredFromExitInfo).
  for (BasicBlock *P : predecessors(BB)) {
    // When P ends in an InvokeInst whose unwind dest is BB,
    // exitDataFor returns either the pre-invoke unwind variant (if
    // recorded) or nullptr (if the invoke was virtualized and the
    // unwind edge is dead — that pred contributes nothing).
    BlockExitData *ED = A.exitDataFor(P, BB);
    if (!ED)
      continue;
    PredBBs.push_back(P);
    Preds.push_back(ED);
  }
  if (Preds.empty()) {
    // Nothing to inherit; start empty. PHIs (if any) at this BB still need
    // walking — none of their incomings have a live pred contribution, so
    // processBlockPhis simply records nothing here, but we run it for
    // consistency with the non-degenerate paths.
    A.processBlockPhis(BB, PendingMergePhis[BB]);
    return;
  }
  if (Preds.size() == 1) {
    A.inheritFromExit(*Preds[0]);
    // With a single live pred this is effectively a single-pred
    // block (all other preds were dead/killed). processBlockPhis still
    // has to alias / materialize the explicit LLVM PHIs of pointer type.
    A.processBlockPhis(BB, PendingMergePhis[BB]);
    return;
  }

  // identicalExitData fast path (Graal's identicalObjectStates,
  // PartialEscapeClosure.java:935-936) runs INSIDE the do/while below — see
  // the comment there. It is sound there because per-pred materialization
  // gives each pred a DISTINCT materialized value, so the deep-value
  // equivalence check does not false-positive post-cascade.

  // Only IDs in the intersection of every predecessor's tracked set may
  // remain unified at BB's entry; any ID present on some preds but not all is
  // dropped here (processBlockPhis Case-A fallback handles it by materialising
  // the virtual incomings at the per-pred terminator).
  intersectVirtualObjects();

  // Iterative merge stabilization (Graal's do { ... } while (materialized)
  // merge fixpoint). A nested materialize triggered inside per-field PHI
  // synthesis can invalidate earlier per-VO decisions, so the per-VO loop is
  // re-run whenever any materializeAtPredFromExitInfo call emits an Effect.
  //
  // The retry discards this merge's partial OUTPUT and re-derives it from a
  // clean slate — mirroring Graal's newState.resetObjectStates() +
  // mergeEffects.clear(). Because processBlock calls resetPerBlockState()
  // before invoking the MergeProcessor, the output state (CurrentState /
  // FieldStates / LockCounts / LiveLockEnters / Materialized) is always EMPTY
  // at run() entry, so resetting to empty is equivalent to a snapshot/restore.
  // Merge-block-local CreatePHI effects are buffered in MergeEffects (not
  // committed to PendingMergePhis[BB] until the fixpoint converges), so a
  // retry just clears the buffer. Materialize side-effects (per-pred ExitInfo
  // flips, MaterializedAtPred, per-pred BlockEffects, Eligible flags, and
  // Result.NextSeqNo) survive across retries by design — they are monotone,
  // real effects. Graal's per-phi aliases[phi]=null is mirrored by clearing
  // every BB-phi alias before each iteration's processBlockPhis, so aliases
  // are re-derived idempotently and need no snapshot. Progress is monotone, so
  // the cap of 10 is a defensive safety net only.
  constexpr unsigned MaxRetries = 10;
  size_t OwnedPhisMark = Result.OwnedPhis.size();
  size_t OwnedInstsMark = Result.OwnedInsts.size();
  unsigned Iter = 0;
  bool Changed = false;

  do {
    if (Iter > 0) {
      // Discard this merge's partial output and start the iteration clean.
      // (ExitInfo / MaterializedAtPred / per-pred BlockEffects / Eligible /
      // NextSeqNo all carry over by design.)
      CurrentState = jeandle::PEABlockState();
      FieldStates.clear();
      LockCounts.clear();
      LiveLockEnters.clear();
      Materialized.clear();
      MergeEffects.clear();
      A.deleteOwnedSince(OwnedPhisMark, OwnedInstsMark);
    }
    if (Iter >= MaxRetries) {
      // Safety net. Cap reached — bail every VO in the working set so the
      // original IR survives at this merge. This indicates a pathology in the
      // input.
      LLVM_DEBUG(dbgs() << "PEA: mergeStates retry cap (" << MaxRetries
                        << ") reached at BB '" << BB->getName()
                        << "'; bailing all VOs at merge.\n");
      for (jeandle::ObjectID ID : IDs)
        Eligible[ID] = false;
      return;
    }
    ++Iter;
    Changed = false;

    // identicalExitData fast path (Graal's identicalObjectStates,
    // PartialEscapeClosure.java:935-936), re-evaluated each iteration. If every
    // predecessor's BlockExitData is byte-equivalent, the merge is degenerate:
    // inherit preds[0] directly and skip the O(|preds|*|virtuals|*|offsets|)
    // per-VO merge. processBlockPhis still runs (Graal runs getPhis() at
    // :1041-1046) to alias any pointer PHIs.
    //
    // SOUNDNESS (why this is safe inside the loop, matching Graal): Graal uses
    // a reference-identity compare on the ObjectState[] array — a materialize
    // cascade COW-clones each pred's array, so the check returns false
    // post-cascade. Jeandle's per-block snapshot model has no shared array to
    // compare, so it carries DISTINCT per-pred placeholder Value*s in
    // MaterializedValues instead (materializeAtPredFromExitInfo): after a
    // cascade each pred's value differs, so exitDataEquivalent returns false
    // and the fast path correctly does NOT fire. It therefore effectively only
    // fires on the first iteration, before any materialization — exactly when
    // an all-equivalent result is genuinely identical.
    {
      // Under escape-point placement, two preds can both report an object
      // materialized with the SAME OrigAlloc placeholder yet have DIFFERENT
      // real per-arm NewInvs (each arm escapes independently). byte-equivalent
      // exits are then not genuinely identical, so the fast path must NOT fire
      // when any object is materialized — the per-VO merge (and its
      // materializedValuePhi) is required to reconcile the per-arm values.
      bool AnyMaterialized = !Preds[0]->Materialized.empty();
      bool AllSame = !AnyMaterialized;
      for (unsigned i = 1; AllSame && i < Preds.size(); ++i) {
        if (!A.exitDataEquivalent(*Preds[0], *Preds[i])) {
          AllSame = false;
          break;
        }
      }
      if (AllSame) {
        A.inheritFromExit(*Preds[0]);
        A.processBlockPhis(BB, MergeEffects);
        break; // exit the do/while; fall through to the MergeEffects commit.
      }
    }

    for (jeandle::ObjectID ID : IDs) {
      if (!Eligible.lookup(ID))
        continue;
      Changed |= mergeObjectState(ID);
    }

    // Run the PHI loop INSIDE the merge do-while (Graal resets aliases[phi]
    // before each iteration's processPhi). Case-A fallback and Case-C
    // synthesis both call materializeAtPredFromExitInfo on inner / per-pred
    // VOs; any such call mutates a pred's ExitInfo (Virtuals->Materialized),
    // which can invalidate the per-VO decisions just made. We detect the work
    // via Result.NextSeqNo delta and set Changed=true so the next iter re-runs
    // the per-VO loop against the updated pred ExitInfos. processBlockPhis
    // routes its CreatePHI effects into MergeEffects (retry-cleared), matching
    // Graal's mergeEffects buffer.
    for (PHINode &Phi : BB->phis())
      Aliases.resetAlias(&Phi);
    {
      uint32_t PrePhiSeqNo = Result.NextSeqNo;
      A.processBlockPhis(BB, MergeEffects);
      if (Result.NextSeqNo != PrePhiSeqNo)
        Changed = true;
    }
  } while (Changed);

  // Commit this merge's deferred CreatePHI effects to PendingMergePhis[BB];
  // processBlock drains the list at the end of the block walk, assigning each
  // a fresh SeqNo strictly after any per-pred Materialize emitted during the
  // same block walk (the self-loop ordering invariant).
  PendingMergePhis[BB].addAll(MergeEffects);
}

// Graal intersectVirtualObjects: IDs tracked (virtual or materialized) by
// EVERY predecessor. Only these may remain unified at the merge entry.
void Analyzer::MergeProcessor::intersectVirtualObjects() {
  DenseSet<jeandle::ObjectID> Intersect;
  for (jeandle::ObjectID ID : Preds[0]->Virtuals)
    Intersect.insert(ID);
  for (jeandle::ObjectID ID : Preds[0]->Materialized)
    Intersect.insert(ID);
  for (unsigned i = 1; i < Preds.size(); ++i) {
    // Test membership directly against each pred's existing tracked sets
    // (Preds[i]->Virtuals/Materialized are already DenseSets) rather than
    // materializing a per-predecessor copy.
    SmallVector<jeandle::ObjectID, 8> ToRemove;
    for (jeandle::ObjectID ID : Intersect)
      if (!Preds[i]->Virtuals.count(ID) && !Preds[i]->Materialized.count(ID))
        ToRemove.push_back(ID);
    for (jeandle::ObjectID ID : ToRemove)
      Intersect.erase(ID);
  }
  IDs.assign(Intersect.begin(), Intersect.end());
  llvm::sort(IDs); // deterministic order for ineligibility/marking effects.
}

// Graal per-object loop body in merge(): decide virtual/materialized/phi for
// one object across all predecessors. Returns true if a
// materializeAtPredFromExitInfo call emitted an Effect this iteration (the
// run() do/while re-runs on true).
bool Analyzer::MergeProcessor::mergeObjectState(jeandle::ObjectID ID) {
  // Per-pred disposition.
  bool AllVirtual = true;
  bool AllMaterialized = true;
  for (const auto *P : Preds) {
    bool V = P->Virtuals.count(ID);
    bool M = P->Materialized.count(ID);
    if (!V)
      AllVirtual = false;
    if (!M)
      AllMaterialized = false;
  }

  if (AllMaterialized) {
    // Every incoming path already materialized the object. If every pred
    // shares one real (non-placeholder) materialized pointer, install it
    // directly (Graal uniqueMaterializedValue,
    // PartialEscapeClosure.java:982). Otherwise (differing pointers, or the
    // per-pred OrigAlloc placeholder) build a ptr addrspace(1)
    // materializedValuePhi. No pred is virtual here, so materializeAndBuildPhi
    // materializes nothing and just builds the PHI.
    jeandle::VirtualObject &VObj = *Result.VirtualObjects[ID];
    Value *Unique = nullptr;
    bool AllSame = true;
    for (auto *P : Preds) {
      auto It = P->MaterializedValues.find(ID);
      Value *V = (It != P->MaterializedValues.end()) ? It->second
                                                     : VObj.AllocationCall;
      if (!Unique)
        Unique = V;
      else if (Unique != V) {
        AllSame = false;
        break;
      }
    }
    bool IsPerPredPlaceholder =
        AllSame && Preds.size() > 1 && Unique == VObj.AllocationCall;
    if (AllSame && !IsPerPredPlaceholder) {
      jeandle::ObjectState OS;
      OS.escape(Unique);
      CurrentState.addObject(ID, std::move(OS));
      Materialized.insert(ID);
      return false;
    }
    return materializeAndBuildPhi(ID);
  }

  if (!AllVirtual) {
    // Mixed (virtual on some paths, materialized or missing on others).
    // Short-circuit when the object isn't tracked on every pred (e.g., an
    // LLVM PHI mixing a virtual incoming from one branch and an unrelated
    // pointer from another). The merge entry simply doesn't carry this
    // ObjectID; processBlockPhis below picks up any LLVM PHI that references
    // it and materializes the virtual incomings.
    bool MissingOnSomePred = false;
    for (auto *P : Preds) {
      if (!P->Materialized.count(ID) && !P->Virtuals.count(ID)) {
        MissingOnSomePred = true;
        break;
      }
    }
    if (MissingOnSomePred)
      return false;

    // Mixed virtual+materialized merge — Graal's per-pred+PHI else-branch
    // (PartialEscapeClosure.java:984-1002): materialize each still-virtual pred
    // at its predecessor-end and build a materializedValuePhi over the per-pred
    // materialized pointers. This is sound because (a) each pred's materialize
    // is placed at its own predecessor-end (self-contained, not a dominating
    // hoist covering all arms), and (b) the lock model deletes+re-emits enters
    // at the materialize point (no in-place retarget that would need the
    // materialize to dominate the original enter).
    //
    // Synthetic VOs (Case-C, borrowed AllocationCall) still bail: they have no
    // per-pred allocation to materialize from. TODO(cascade-materialize):
    // per-pred source materialization + PHI reuse (see ensureMaterialized).
    if (Result.VirtualObjects[ID]->IsSynthetic) {
      A.markIneligible(ID); // cascades transitively over nested synthetics.
      return false;
    }
    return materializeAndBuildPhi(ID);
  }

  // All preds report Virtual: check lock counts and live enter-stacks.
  unsigned RefLC = Preds[0]->LockCounts.lookup(ID);
  bool LocksMatch = true;
  for (const auto *P : Preds) {
    if (P->LockCounts.lookup(ID) != RefLC) {
      LocksMatch = false;
      break;
    }
  }
  bool StacksMatch = true;
  if (LocksMatch && RefLC != 0) {
    // Compare (Call, BytecodeDepth) so two paths that re-entered the SAME
    // call site but at different bytecode-level depths are distinct stacks:
    // lock capture/re-emit at a downstream escape would re-emit the same lock
    // set either way, so bytecode depth is what distinguishes the two paths.
    const auto &RefStack = Preds[0]->LiveLockEnters.lookup(ID);
    for (const auto *P : Preds) {
      const auto &S = P->LiveLockEnters.lookup(ID);
      if (S.size() != RefStack.size()) {
        StacksMatch = false;
        break;
      }
      for (unsigned i = 0; i < S.size(); ++i) {
        if (S[i].Call != RefStack[i].Call ||
            S[i].BytecodeDepth != RefStack[i].BytecodeDepth) {
          StacksMatch = false;
          break;
        }
      }
      if (!StacksMatch)
        break;
    }
  }

  if (AllVirtual && LocksMatch && StacksMatch)
    return mergeFieldStates(ID);

  // else (Graal else-branch, PartialEscapeClosure.java:984-1002): a lock-count
  // or live-enter-stack mismatch forces per-pred materialization. Do it in a
  // SINGLE pass and build the materializedValuePhi here (each pred carries its
  // OWN lock list, so the alloc-commit emits exactly that pred's monitorenter
  // set — no synthesized enters are added on the lower-count side). On the
  // retry every pred has flipped to Materialized; the loop-header-cached PHI
  // shell stays stable, so this converges in at most one retry.
  return materializeAndBuildPhi(ID);
}

// Graal else-branch (PartialEscapeClosure.java:984-1002) / the incompatible
// tail of mergeObjectStates (PartialEscapeBlockState.java:1312-1331):
// materialize every still-virtual pred and build a ptr addrspace(1)
// materializedValuePhi over the per-pred materialized pointers, installing a
// materialized ObjectState. Reached for the mixed `!AllVirtual` case (Graal's
// else-branch, PartialEscapeClosure.java:984-1002), for
// AllMaterialized-differing-pointers (no virtuals to materialize — just builds
// the PHI), and for lock/stack mismatch (every virtual pred materialized).
// Returns true if a per-pred materialize emitted an Effect this iteration (the
// run() do/while re-runs on true).
// TODO(ensure-virtualized): when an EnsureVirtualized bit lands on ObjectState,
// downgrade it here per-pred (Graal setEnsureVirtualized(false) where not all
// preds agree) — this entry covers the AllMaterialized-divergence arm of
// mergeObjectState.
bool Analyzer::MergeProcessor::materializeAndBuildPhi(jeandle::ObjectID ID) {
  jeandle::VirtualObject &VObj = *Result.VirtualObjects[ID];
  Type *PtrTy =
      PointerType::get(A.F.getContext(), jeandle::AddrSpace::JavaHeapAddrSpace);

  // Loop-PHI cache: stable PHI across fixpoint iterations.
  PHINode *Phi = A.getOrCreateLoopFieldPhi(
      BB, ID, /*Offset=*/-1, PtrTy, Preds.size(), "pea.materialized.phi");
  auto PE = std::make_unique<jeandle::CreatePHIEffect>();
  PE->Block = BB;
  // SeqNo is assigned at drain time (end of processBlock(BB)) so the CreatePHI
  // sorts strictly after any per-pred Materialize emitted during the same block
  // walk. The placeholder 0 here is overwritten when PendingMergePhis[BB]
  // drains.
  PE->SeqNo = 0;
  PE->ObjID = ID;
  PE->PhiInst = Phi;
  PE->PHIType = PtrTy;
  // RAUWOrigToPHI is set unconditionally (always true, see below) so the merge
  // PHI is registered as a definition point of OrigAlloc. The point-sensitive
  // resolution sub-pass then rewrites only the post-merge OrigAlloc uses the
  // PHI actually dominates onto it (arm uses that a per-pred NewInv dominates
  // still resolve to that NewInv).

  bool Mat = false;
  for (unsigned i = 0; i < Preds.size(); ++i) {
    if (!Eligible.lookup(ID))
      break;
    // Materialize any still-virtual pred at its terminator (each carrying its
    // OWN lock list). AllMaterialized-differing-pointers has no virtual preds,
    // so this is a no-op there; lock/stack mismatch materializes each one.
    if (Preds[i]->Virtuals.count(ID)) {
      uint32_t PreSeqNo = Result.NextSeqNo;
      // Per-pred mat destined for THIS merge (BB): TargetMerge=BB so the
      // placeholder is keyed (PH, BB, ID) and the transform's critical-edge
      // pre-pass splits the PH→BB edge. Per-pred does NOT flip the shared
      // BlockExits[PH] (NewInv dominates one edge, must not leak) — the
      // placeholder is read back below via getOrCreatePerPredMatPlaceholder.
      A.materializeAtPredFromExitInfo(ID, PredBBs[i], *Preds[i],
                                      /*SkipGlobalRAUW=*/true, MatReason::Merge,
                                      /*TargetMerge=*/BB);
      if (Result.NextSeqNo != PreSeqNo)
        Mat = true;
    }
    // Per-pred does NOT flip BlockExits[PH] (it must not leak to non-target
    // successors), so the placeholder is not in Preds[i]->MaterializedValues.
    // For a pred that was (or just became) per-pred-mat'd for THIS merge, read
    // the placeholder directly from the cache (stable across retries); for a
    // pred already materialized via escape-point/Case-A (real flip), use that
    // recorded value; else fall back to OrigAlloc (the resolution sub-pass
    // rewrites it to the dominating def).
    Value *V;
    if (Preds[i]->Virtuals.count(ID))
      V = A.getOrCreatePerPredMatPlaceholder(PredBBs[i], BB, ID);
    else {
      auto It = Preds[i]->MaterializedValues.find(ID);
      V = (It != Preds[i]->MaterializedValues.end()) ? It->second
                                                     : VObj.AllocationCall;
    }
    PE->PHIIncomingValues.push_back(V);
    PE->PHIIncomingBlocks.push_back(PredBBs[i]);
  }
  // Always register this materialized-object PHI as a definition point of
  // OrigAlloc. Post-merge OrigAlloc uses (and uses on arms whose NewInv does
  // not dominate them) are rewritten to it by the resolution sub-pass, which
  // guards each rewrite on dominance — so pre-merge arm uses still resolve to
  // that arm's own NewInv, and only uses the PHI actually dominates go through
  // it. This covers AllMaterialized-differing (both arms live-path
  // materialized, no per-pred materialize in this call) as well as the per-pred
  // / mixed cases.
  PE->RAUWOrigToPHI = true;
  MergeEffects.add(std::move(PE));

  jeandle::ObjectState OS;
  OS.escape(Phi);
  CurrentState.addObject(ID, std::move(OS));
  Materialized.insert(ID);
  return Mat;
}

// Graal mergeObjectStates (PartialEscapeClosure.java:1102-1331), COMPATIBLE
// branch: per-offset field-PHI synthesis for a virtual object whose locks agree
// across all predecessors. Identical entries flow straight into the merged
// state; disagreements synthesize a per-offset PHI. (The INCOMPATIBLE tail of
// Graal's mergeObjectStates — materialize every pred + materializedValuePhi —
// corresponds to materializeAndBuildPhi here.) Returns true if a nested
// inner-VO materialize emitted an Effect (retry).
//
// TODO(mergeObjectStates-two-slot-and-bytearray): mirror Graal's
// virtualByteCount / twoSlotKinds compatibility pre-scan
// (PartialEscapeClosure.java:1110-1253) and widened-PHI synthesis
// (:1256-1283). The integer-widening zext below is guarded so a narrow pred
// with a conflicting non-default scalar in the wide type's byte span bails
// instead of discarding adjacent-byte contributions. Today that widening is
// unreachable — processStore's getOrCreateFieldIndex bails on a width
// mismatch at the same offset, so two preds never present different integer
// widths for one offset here — but the guard keeps the merge sound if that
// ever changes (e.g. byte-array write decomposition under unsafe support).
bool Analyzer::MergeProcessor::mergeFieldStates(jeandle::ObjectID ID) {
  unsigned RefLC = Preds[0]->LockCounts.lookup(ID); // all preds agree here.
  bool Changed = false;

  // Check field states at every tracked offset. Identical entries flow
  // straight into Merged; disagreements trigger field-PHI synthesis.
  DenseSet<int64_t> Offsets;
  for (const auto *P : Preds) {
    auto FIt = P->FieldStates.find(ID);
    if (FIt == P->FieldStates.end())
      continue;
    for (auto &Kv : FIt->second)
      Offsets.insert(Kv.first);
  }
  SmallVector<int64_t, 8> SortedOffsets(Offsets.begin(), Offsets.end());
  llvm::sort(SortedOffsets); // determinism
  bool BailObject = false;
  DenseMap<int64_t, jeandle::FieldValue> Merged;
  // Snapshot of newly-emitted CreatePHI effects for this object's fields;
  // committed to Result only if every offset succeeds.
  jeandle::EffectList PendingPhiEffects;
  for (int64_t Off : SortedOffsets) {
    // Per-offset bail. When a single offset cannot be merged (incompatible
    // non-integer types, no safe coercion), DROP just that offset from the
    // merged FieldStates rather than marking the entire VO ineligible.
    // Downstream loads from a missing offset see Unknown — processLoad forces
    // materialization at the load site, which is conservative-but-sound. The
    // whole-VO BailObject path remains for unrecoverable conditions.
    bool BailOffset = false;
    jeandle::FieldValue Ref = jeandle::FieldValue::unknown();
    bool HaveRef = false;
    bool Disagrees = false;
    for (const auto *P : Preds) {
      jeandle::FieldValue FV = jeandle::FieldValue::unknown();
      auto FIt = P->FieldStates.find(ID);
      if (FIt != P->FieldStates.end()) {
        auto OIt = FIt->second.find(Off);
        if (OIt != FIt->second.end())
          FV = OIt->second;
      }
      if (!HaveRef) {
        Ref = FV;
        HaveRef = true;
      } else if (!Ref.shallowEquals(FV)) {
        Disagrees = true;
        break;
      }
    }
    if (!Disagrees) {
      if (!Ref.isUnknown())
        Merged[Off] = Ref;
      continue;
    }

    // Field disagreement at Off — attempt field-PHI synthesis. Decide the
    // merged PHI type. Compatibility: every non-unknown entry's declared type
    // must be identical, OR every non-unknown entry must be a pointer in the
    // Java heap addrspace (PHI is ptr addrspace(1), ref/scalar
    // interchangeable). Additionally allows integer-width promotion: promote
    // the PHI type to the widest integer and zext narrower entries.
    Type *PhiType = nullptr;
    bool AllPointer = true;
    bool AllInteger = true;
    unsigned WidestIntBits = 0;
    for (const auto *P : Preds) {
      auto FIt = P->FieldStates.find(ID);
      if (FIt == P->FieldStates.end())
        continue;
      auto OIt = FIt->second.find(Off);
      if (OIt == FIt->second.end())
        continue;
      const jeandle::FieldValue &FV = OIt->second;
      if (FV.isUnknown())
        continue;
      Type *T = nullptr;
      if (FV.isScalar())
        T = FV.getScalar()->getType();
      else
        T = FV.getDeclaredType();
      if (!T) {
        BailObject = true;
        break;
      }
      if (!T->isPointerTy() ||
          T->getPointerAddressSpace() != jeandle::AddrSpace::JavaHeapAddrSpace)
        AllPointer = false;
      if (T->isIntegerTy()) {
        unsigned B = T->getIntegerBitWidth();
        if (B > WidestIntBits)
          WidestIntBits = B;
      } else {
        AllInteger = false;
      }
      if (!PhiType) {
        PhiType = T;
      } else if (PhiType != T) {
        // Heap-pointer promotion is the existing rule.
        bool BothJavaHeapPtr = PhiType->isPointerTy() && T->isPointerTy() &&
                               PhiType->getPointerAddressSpace() ==
                                   jeandle::AddrSpace::JavaHeapAddrSpace &&
                               T->getPointerAddressSpace() ==
                                   jeandle::AddrSpace::JavaHeapAddrSpace;
        // Integer-widening promotion: both integers, possibly different
        // widths — defer PhiType selection to the widest and emit zext on
        // narrower per-pred inputs below.
        bool BothInteger = PhiType->isIntegerTy() && T->isIntegerTy();
        if (!BothJavaHeapPtr && !BothInteger) {
          BailOffset = true;
          break;
        }
        // For the integer case, leave PhiType as the LARGER type (re-picked
        // from WidestIntBits below).
      }
    }
    if (BailObject)
      break;
    if (BailOffset) {
      // Drop this offset and keep going.
      continue;
    }
    if (!PhiType) {
      // Should be unreachable — a disagreement implies at least two distinct
      // non-unknown entries.
      continue;
    }
    if (AllPointer) {
      PhiType = PointerType::get(A.F.getContext(),
                                 jeandle::AddrSpace::JavaHeapAddrSpace);
    } else if (AllInteger && WidestIntBits != 0) {
      // Select the widest integer type as the PHI type and emit zext on
      // narrower per-pred values below.
      PhiType = IntegerType::get(A.F.getContext(), WidestIntBits);
    }

    // Compute per-pred input value.
    SmallVector<Value *, 4> InValues;
    InValues.reserve(Preds.size());
    bool LocalBail = false;
    for (unsigned i = 0; i < Preds.size(); ++i) {
      jeandle::FieldValue FV = jeandle::FieldValue::unknown();
      auto FIt = Preds[i]->FieldStates.find(ID);
      if (FIt != Preds[i]->FieldStates.end()) {
        auto OIt = FIt->second.find(Off);
        if (OIt != FIt->second.end())
          FV = OIt->second;
      }
      Value *In = nullptr;
      if (FV.isUnknown()) {
        In = jeandle::FieldValue::defaultFor(PhiType);
      } else if (FV.isScalar()) {
        Value *V = FV.getScalar();
        if (V->getType() != PhiType) {
          // Integer-widen via zext if both are integers AND V is strictly
          // narrower than PhiType. Other mismatches (scalar at pointer slot,
          // etc.) still LocalBail. Constants are the overwhelming common case
          // (loop-invariant 0/1 stores); non-const zext would need a per-pred
          // CreatePHI shim (deferred).
          if (V->getType()->isIntegerTy() && PhiType->isIntegerTy() &&
              V->getType()->getIntegerBitWidth() <
                  PhiType->getIntegerBitWidth()) {
            if (auto *CI = dyn_cast<ConstantInt>(V)) {
              // Graal isEntryDefaults mirror (PartialEscapeClosure.java:1110-
              // 1253): the narrow pred must have NO conflicting non-default
              // scalar in the wide type's byte span [Off+1, Off+WideBytes),
              // or the zext would discard adjacent-byte contributions.
              unsigned WideBytes = PhiType->getIntegerBitWidth() / 8;
              auto SpanIt = Preds[i]->FieldStates.find(ID);
              for (unsigned B = 1; B < WideBytes; ++B) {
                if (SpanIt == Preds[i]->FieldStates.end())
                  break;
                auto AOff = SpanIt->second.find(Off + (int64_t)B);
                if (AOff == SpanIt->second.end())
                  continue; // missing entry == default
                const jeandle::FieldValue &AFV = AOff->second;
                if (AFV.isUnknown())
                  continue;
                if (AFV.isScalar())
                  if (auto *ACI = dyn_cast<ConstantInt>(AFV.getScalar()))
                    if (ACI->isZero())
                      continue;
                LocalBail = true; // conflicting non-default adjacent byte
                break;
              }
              if (LocalBail)
                break;
              In = ConstantInt::get(
                  PhiType, CI->getValue().zext(PhiType->getIntegerBitWidth()));
            } else {
              LocalBail = true;
              break;
            }
          } else {
            // Scalar with a non-matching primitive type at a pointer slot, or
            // non-integer width mismatch, or value-wider-than-PhiType.
            LocalBail = true;
            break;
          }
        } else {
          In = V;
        }
      } else if (FV.isMaterializedRef()) {
        if (!PhiType->isPointerTy()) {
          LocalBail = true;
          break;
        }
        In = FV.getMaterialized();
      } else if (FV.isVirtualRef()) {
        if (!PhiType->isPointerTy()) {
          LocalBail = true;
          break;
        }
        jeandle::ObjectID InnerID = FV.getVirtualRef();
        // Materialize the inner object at this pred's terminator. After this,
        // the field's effective input value is OrigAlloc(inner) — the
        // transform's MatPerBlock substitutes it with NewInv at apply. Track
        // whether this call emitted any Effects via the SeqNo delta.
        uint32_t PreSeqNo = Result.NextSeqNo;
        // Case-A inner-VO mat (SkipGlobalRAUW=false): mat at PH end, NewInv
        // dominates all successors — flips the shared ExitInfo (original
        // behavior) so other merges sharing this pred see the materialization.
        A.materializeAtPredFromExitInfo(InnerID, PredBBs[i], *Preds[i],
                                        /*SkipGlobalRAUW=*/false,
                                        MatReason::Phi,
                                        /*TargetMerge=*/nullptr);
        if (Result.NextSeqNo != PreSeqNo)
          Changed = true;
        if (!Eligible.lookup(InnerID)) {
          LocalBail = true;
          break;
        }
        jeandle::VirtualObject &InnerVO = *Result.VirtualObjects[InnerID];
        // Defensively rewrite this pred's outer-VO FieldStates entry for the
        // just-materialized inner to MaterializedRef so a sibling successor
        // of the pred (other-than-BB) that later inherits from Preds[i] sees
        // the materialized pointer rather than a stale VirtualRef(InnerID).
        Preds[i]->FieldStates[ID][Off] =
            jeandle::FieldValue::materializedRef(InnerVO.AllocationCall);
        In = InnerVO.AllocationCall;
      } else {
        LocalBail = true;
        break;
      }
      InValues.push_back(In);
    }
    if (LocalBail) {
      // Per-offset bail. Drop this offset, keep going.
      continue;
    }

    // Loop-PHI cache: stable PHI across fixpoint iterations.
    PHINode *Phi = A.getOrCreateLoopFieldPhi(BB, ID, Off, PhiType, Preds.size(),
                                             "pea.field.phi");
    auto PE = std::make_unique<jeandle::CreatePHIEffect>();
    PE->Block = BB;
    // SeqNo assigned at drain time; see PendingMergePhis comment.
    PE->SeqNo = 0;
    PE->ObjID = ID;
    PE->PhiInst = Phi;
    PE->PHIType = PhiType;
    for (unsigned i = 0; i < Preds.size(); ++i) {
      PE->PHIIncomingValues.push_back(InValues[i]);
      PE->PHIIncomingBlocks.push_back(PredBBs[i]);
    }
    PendingPhiEffects.add(std::move(PE));

    if (PhiType->isPointerTy())
      Merged[Off] = jeandle::FieldValue::materializedRef(Phi);
    else
      Merged[Off] = jeandle::FieldValue::scalar(Phi);
  }
  if (BailObject) {
    Eligible[ID] = false;
    return Changed;
  }

  // Commit this object's field-PHI effects to the merge's deferred buffer;
  // they are flushed to PendingMergePhis[BB] (and assigned SeqNos) only after
  // the fixpoint converges.
  MergeEffects.addAll(PendingPhiEffects);

  // Case B: object stays virtual at BB entry with the merged field state.
  CurrentState.addObject(ID, jeandle::ObjectState());
  if (!Merged.empty())
    FieldStates[ID] = std::move(Merged);
  if (RefLC != 0) {
    LockCounts[ID] = RefLC;
    // The merged live stack is identical to (any) pred's stack, since the
    // StacksMatch check above succeeded.
    const auto &RefStack = Preds[0]->LiveLockEnters.lookup(ID);
    if (!RefStack.empty())
      LiveLockEnters[ID] = RefStack;
  }
  return Changed;
}

void Analyzer::snapshotExitStateInto(BlockExitData &Data) {
  for (auto &VObjUP : Result.VirtualObjects) {
    jeandle::ObjectID ID = VObjUP->getID();
    if (!Eligible.lookup(ID))
      continue;
    const jeandle::ObjectState *OS = CurrentState.getObjectStateOptional(ID);
    if (!OS)
      continue;
    if (OS->isVirtual()) {
      Data.Virtuals.insert(ID);
      auto FIt = FieldStates.find(ID);
      if (FIt != FieldStates.end() && !FIt->second.empty())
        Data.FieldStates[ID] = FIt->second;
      auto LIt = LockCounts.find(ID);
      if (LIt != LockCounts.end() && LIt->second != 0)
        Data.LockCounts[ID] = LIt->second;
      // Snapshot the live monitorenter stack so successor blocks see the
      // same path-specific call set we did.
      auto SIt = LiveLockEnters.find(ID);
      if (SIt != LiveLockEnters.end() && !SIt->second.empty())
        Data.LiveLockEnters[ID] = SIt->second;
    } else if (OS->isMaterialized()) {
      Data.Materialized.insert(ID);
      // Capture the materialized pointer at block exit so downstream
      // merges can use it as the PHI input on this edge.
      Data.MaterializedValues[ID] = OS->getMaterializedValue();
    }
  }
}

void Analyzer::snapshotExitState(BasicBlock *BB) {
  BlockExitInfo Info;
  snapshotExitStateInto(Info);
  BlockExits[BB] = std::move(Info);
}

void Analyzer::deleteOwnedSince(size_t PhiMark, size_t InstMark) {
  // Pop and delete any unparented PHIs/insts added since the marks. The merge
  // only creates unparented PHIs via createUnparentedPhi /
  // getOrCreateLoopFieldPhi's out-of-loop fallback; insertion into a BasicBlock
  // happens in the transform pass, so any value added during a failed merge
  // iteration is still unparented when we discard it. In-loop-cached PHIs (loop
  // headers AND non-header in-loop merge blocks) live in OwnedLoopFieldPhis (a
  // separate bucket) and are intentionally preserved so they stay stable across
  // fixpoint iterations and across per-merge retries.
  while (Result.OwnedPhis.size() > PhiMark) {
    WeakTrackingVH &VH = Result.OwnedPhis.back();
    if (Value *V = VH) {
      if (auto *P = dyn_cast<PHINode>(V))
        if (!P->getParent())
          delete P;
    }
    Result.OwnedPhis.pop_back();
  }
  while (Result.OwnedInsts.size() > InstMark) {
    WeakTrackingVH &VH = Result.OwnedInsts.back();
    if (Value *V = VH) {
      if (auto *I = dyn_cast<Instruction>(V))
        if (!I->getParent())
          I->deleteValue();
    }
    Result.OwnedInsts.pop_back();
  }
}

void Analyzer::processBlockPhis(BasicBlock *BB, jeandle::EffectList &Out) {
  // Walk explicit LLVM PHIs of java-heap pointers. Other PHIs (e.g., scalar
  // i32 PHIs of folded virtual-load results) flow through normal SSA and are
  // not the concern of the analyzer.
  for (PHINode &Phi : BB->phis()) {
    Type *Ty = Phi.getType();
    if (!Ty->isPointerTy())
      continue;
    if (cast<PointerType>(Ty)->getAddressSpace() !=
        jeandle::AddrSpace::JavaHeapAddrSpace)
      continue;

    // Resolve each incoming against its predecessor's exit snapshot. We only
    // check the AliasMap (functions like full resolveVirtualRef would need a
    // pred-specific PEABlockState, which we don't have on hand); pointer
    // PHIs whose incomings are themselves derived through GEP/cast chains
    // are uncommon enough that the alias-only resolution is sufficient.
    // Note that the AliasMap is function-wide so a virtual alias registered
    // in any block is visible here.
    SmallVector<std::optional<jeandle::ObjectID>, 4> InIDs;
    bool AnyVirtual = false;
    bool AnyDerived = false; // a resolved incoming with a non-zero/non-constant
                             // byte offset (a GEP-with-offset, not the object)
    InIDs.reserve(Phi.getNumIncomingValues());
    for (unsigned I = 0; I < Phi.getNumIncomingValues(); ++I) {
      BasicBlock *Pred = Phi.getIncomingBlock(I);
      Value *V = Phi.getIncomingValue(I);
      std::optional<jeandle::ObjectID> Found;
      auto AID = Aliases.getVirtualAlias(V);
      BlockExitData *PredED = exitDataFor(Pred, BB);
      if (AID && PredED && PredED->Virtuals.count(*AID)) {
        Found = *AID;
        // A Case-B/C PHI aliases the whole object at byte offset 0
        // (resolveFieldOffset() of a PHI returns 0), so every incoming must
        // denote the object at offset 0 -- e.g. OrigAlloc itself, a bitcast, a
        // zero-offset GEP, or a whole-object Case-B PHI alias. An incoming with
        // a non-zero OR non-constant byte offset (a GEP-with-offset, including
        // a variable index) cannot be represented by a whole-object alias, so
        // we route the PHI to Case A: materialize per pred and re-derive each
        // incoming at its true offset.
        std::optional<int64_t> FOff = jeandle::pea::resolveFieldOffset(V, DL);
        if (!FOff || *FOff != 0)
          AnyDerived = true;
      }
      InIDs.push_back(Found);
      if (Found)
        AnyVirtual = true;
    }
    if (!AnyVirtual)
      continue;

    // Case B: every incoming agrees on the same ObjectID AND the object is
    // still virtual at merge entry (mergeStates kept it).
    bool AllSame = true;
    std::optional<jeandle::ObjectID> First;
    for (auto &O : InIDs) {
      if (!O) {
        AllSame = false;
        break;
      }
      if (!First)
        First = O;
      else if (*First != *O) {
        AllSame = false;
        break;
      }
    }

    if (AllSame && First && !AnyDerived) {
      const jeandle::ObjectState *OS =
          CurrentState.getObjectStateOptional(*First);
      if (OS && OS->isVirtual()) {
        // Register the PHI as an alias for the same ObjectID so downstream
        // load/store handlers in this block resolve through it.
        Aliases.addVirtualAlias(&Phi, *First);
        // Also record the PHI on the per-VO Case-B alias list so commit() can
        // erase it if the VO is NeverEscapes. The incomings are all the VO's
        // OrigAlloc, which EliminateAllocation RAUWs to poison; the PHI is then
        // dead and we erase it explicitly to avoid a `phi [poison, poison]`
        // artefact past PEA.
        CaseBPhiAliases[*First].push_back(&Phi);
        continue;
      }
    }

    // Case C: every incoming resolves to a virtual ID, but the IDs are not
    // all equal. Attempt to synthesize a single merged VirtualObject. On
    // success the PHI is aliased to the new VO and downstream uses fold
    // through it. On failure (compatibility, identity, or per-entry type
    // checks) we fall through to Case A.
    bool TryCaseC = (First /* at least one virtual */) &&
                    !AllSame; // Case B already returned if AllSame succeeded.
    if (TryCaseC && !AnyDerived) {
      bool EveryInputVirtual = true;
      for (auto &O : InIDs) {
        if (!O) {
          EveryInputVirtual = false;
          break;
        }
      }
      if (EveryInputVirtual && synthesizeCaseC(BB, &Phi, InIDs, Out))
        continue;
    }

    // Case A: mixed virtual + non-virtual incomings, OR a Case C attempt
    // that bailed. For every virtual incoming, materialize at that
    // incoming's predecessor terminator. The PHI itself stays in IR; the
    // point-sensitive resolution sub-pass rewrites each virtual incoming's
    // OrigAlloc use to the per-pred NewInv (and MatPerBlock substitution
    // handles PHI nodes synthesized by CreatePHI elsewhere).
    for (unsigned I = 0; I < Phi.getNumIncomingValues(); ++I) {
      if (!InIDs[I])
        continue;
      BasicBlock *Pred = Phi.getIncomingBlock(I);
      BlockExitData *PredED = exitDataFor(Pred, BB);
      if (!PredED) {
        Eligible[*InIDs[I]] = false;
        continue;
      }
      // Skip if the VO is already materialized for THIS merge. Two cases:
      // (a) escape-point/Case-A already flipped the shared state (VO no
      // longer virtual in BlockExits[Pred]); (b) a per-pred mat for THIS
      // merge (Pred, BB, ID) was already emitted — per-pred does NOT flip
      // the shared state (it must not leak to non-target successors), so the
      // Virtuals check alone would miss it. Re-firing Case-A here would emit
      // a DUPLICATE invoke (the per-pred mat's NewInv already handles this
      // merge's incoming via the resolution sub-pass). A per-pred mat for a
      // DIFFERENT merge is recorded under (Pred, M2, ID), so the check
      // correctly does NOT skip in that case — Case-A fires normally.
      if (!PredED->Virtuals.count(*InIDs[I]) ||
          isMaterializedAtPred(Pred, BB, *InIDs[I]))
        continue;
      // Case-A mat at PH end (SkipGlobalRAUW=false): NewInv dominates all
      // successors — flips the shared ExitInfo so other merges sharing this
      // pred see the materialization.
      materializeAtPredFromExitInfo(*InIDs[I], Pred, *PredED,
                                    /*SkipGlobalRAUW=*/false, MatReason::Phi,
                                    /*TargetMerge=*/nullptr);

      // Re-derive a DERIVED carry (GEP/bitcast of the object) at the back-edge
      // so the PHI carries a value computed over the freshly-materialized base,
      // not the original body GEP — which no materialization point dominates
      // (it lives before the latch), so the point-sensitive resolution sub-pass
      // cannot rewrite its OrigAlloc operand and it would leak poison. This
      // mirrors Graal getAliasAndResolve + setPhiInput (re-derive the incoming
      // from the per-pred materialized object state at the merge), extended to
      // replay the byte offset (LLVM derived pointers have no Graal analog).
      // Object-carry (V == OrigAlloc) is left to the resolution sub-pass.
      jeandle::ObjectID OID = *InIDs[I];
      if (!Eligible.lookup(OID))
        continue; // a prior/sibling incoming already made this object
                  // ineligible.
      Value *V = Phi.getIncomingValue(I);
      Value *OrigAlloc = Result.VirtualObjects[OID]->AllocationCall;
      if (V == OrigAlloc)
        continue; // object-carry: resolution sub-pass rewrites this OrigAlloc
                  // use.
      int64_t Off = 0;
      bool NonConst = false;
      Value *Base =
          jeandle::pea::stripPointerCastsAndOffsets(V, DL, &Off, &NonConst);
      if (Base != OrigAlloc || NonConst) {
        // Variable-index GEP, or a non-structural alias chain (select/load/PHI
        // embedded in the derivation): cannot soundly re-derive a constant
        // byte offset at the latch. Sound fallback — keep the object real.
        // commit()->dropEffectsFor(ID) purges the materialize above (and this
        // would-be effect by ObjID), so the original allocation survives and no
        // poison leaks.
        markIneligible(OID);
        continue;
      }
      auto RE = std::make_unique<jeandle::RewritePhiIncomingEffect>();
      RE->Block = Pred; // same bucket as the per-pred Materialize (e.g. latch)
      RE->SeqNo = Result.nextSeqNo(); // strictly after that Materialize's SeqNo
      RE->ObjID =
          OID; // so dropEffectsFor purges it if the object turns ineligible
      RE->Phi = &Phi;
      RE->Pred = Pred;
      RE->PerPredPlaceholder =
          getOrCreatePerPredMatPlaceholder(Pred, /*TargetMerge=*/nullptr, OID);
      RE->ByteOffset = Off;
      Result.addBlockEffect(std::move(RE));
    }
  }
}

bool Analyzer::synthesizeCaseC(BasicBlock *BB, PHINode *Phi,
                               ArrayRef<std::optional<jeandle::ObjectID>> InIDs,
                               jeandle::EffectList &Out) {
  // TODO(ensure-virtualized): when an EnsureVirtualized bit lands on
  // ObjectState, downgrade it here per-pred (Graal setEnsureVirtualized(false)
  // where not all preds agree). This entry routes the Case-C per-pred
  // materialisations through materializeAtPredFromExitInfo.
  const unsigned N = Phi->getNumIncomingValues();
  assert(InIDs.size() == N);

  // Resolve per-pred VO ids, predecessor blocks, and their exit snapshots.
  SmallVector<jeandle::ObjectID, 4> PerPredIDs;
  PerPredIDs.reserve(N);
  SmallVector<BasicBlock *, 4> Preds;
  Preds.reserve(N);
  SmallVector<BlockExitData *, 4> ExitInfos;
  ExitInfos.reserve(N);
  for (unsigned i = 0; i < N; ++i) {
    if (!InIDs[i])
      return false;
    PerPredIDs.push_back(*InIDs[i]);
    BasicBlock *P = Phi->getIncomingBlock(i);
    Preds.push_back(P);
    // Respect state-split for unwind successors. exitDataFor returns
    // nullptr if the pred's invoke was virtualized (unwind edge dead), in
    // which case Case C cannot synthesize a merged VO that depends on
    // that pred's data — bail.
    BlockExitData *ED = exitDataFor(P, BB);
    if (!ED)
      return false;
    ExitInfos.push_back(ED);
    // Each per-pred VO must be eligible AND still virtual at pred exit.
    if (!Eligible.lookup(PerPredIDs.back()))
      return false;
    if (!ExitInfos.back()->Virtuals.count(PerPredIDs.back()))
      return false;
  }

  // Compatibility check.
  jeandle::VirtualObject &Ref = *Result.VirtualObjects[PerPredIDs[0]];
  for (unsigned i = 1; i < N; ++i) {
    jeandle::VirtualObject &VO = *Result.VirtualObjects[PerPredIDs[i]];
    if (VO.getKind() != Ref.getKind())
      return false;
    if (VO.Klass != Ref.Klass)
      return false;
    if (Ref.isArray()) {
      if (VO.ArrayElementType != Ref.ArrayElementType)
        return false;
      if (VO.ArrayLength != Ref.ArrayLength)
        return false;
      if (VO.SizeInBytes != Ref.SizeInBytes)
        return false;
      if (VO.ArrayIndexScale != Ref.ArrayIndexScale)
        return false;
      if (VO.ArrayBaseOffset != Ref.ArrayBaseOffset)
        return false;
      if (VO.ArrayLengthLimit != Ref.ArrayLengthLimit)
        return false;
    }
  }

  // Lock compatibility (locksEqual).
  unsigned RefLC = ExitInfos[0]->LockCounts.lookup(PerPredIDs[0]);
  for (unsigned i = 1; i < N; ++i) {
    if (ExitInfos[i]->LockCounts.lookup(PerPredIDs[i]) != RefLC)
      return false;
  }
  if (RefLC != 0) {
    const auto &RefStack = ExitInfos[0]->LiveLockEnters.lookup(PerPredIDs[0]);
    for (unsigned i = 1; i < N; ++i) {
      const auto &S = ExitInfos[i]->LiveLockEnters.lookup(PerPredIDs[i]);
      if (S.size() != RefStack.size())
        return false;
      // Compare Call AND BytecodeDepth so that two stacks built
      // from the same call sites but at different bytecode depths are not
      // collapsed by the Case-C identity-merge fast path. Order is ignored
      // here because the loop fixpoint refreshes Order on every re-push,
      // which would diverge across iterations.
      for (unsigned k = 0; k < S.size(); ++k)
        if (S[k].Call != RefStack[k].Call ||
            S[k].BytecodeDepth != RefStack[k].BytecodeDepth)
          return false;
    }
  }

  // Early-bail if any incoming of Phi equals Phi itself (a
  // back-edge self-reference on a loop-header PHI). A self-loop incoming
  // means the per-pred VO this slot resolves to is one that we're about
  // to synthesise *into* — there is no per-pred independent allocation
  // and the identity check cannot meaningfully proceed.
  for (unsigned i = 0; i < N; ++i) {
    if (Phi->getIncomingValue(i) == Phi)
      return false;
  }

  // Identity check (single-usage-allocation). Every VO has identity.
  // For each per-pred VO we require:
  //   (a) the LLVM PHI is the only "external" user of the per-pred alloc.
  //       An "external" user is one that is neither (i) covered by a planned
  //       PEA effect for that ID (EliminateStore, ReplaceLoad, ReplaceCall,
  //       EliminateAllocation, Materialize), nor (ii) registered in the
  //       AliasMap as a virtual alias of the same ID (GEP/cast/freeze
  //       forwarded by propagatePointerAlias).
  //   (b) no OTHER VO at any pred references this VO via virtualRef in its
  //       FieldStates (otherwise materializing that other VO would also
  //       materialize this one and expose identity).
  for (unsigned i = 0; i < N; ++i) {
    jeandle::ObjectID PID = PerPredIDs[i];
    jeandle::VirtualObject &PVO = *Result.VirtualObjects[PID];
    CallBase *OrigAlloc = PVO.AllocationCall;
    if (!OrigAlloc)
      return false;
    // Cheap-out — an alloc with only one use (the PHI itself)
    // trivially satisfies the no-other-external-user requirement, so
    // the expensive Effect-target collection below is unnecessary.
    // hasNUsesOrMore(2) does an O(1) check against the use list.
    bool MaybeOtherUsers = OrigAlloc->hasNUsesOrMore(2);
    if (MaybeOtherUsers) {
      DenseSet<Instruction *> InternalTargets;
      for (auto &Kv : Result.BlockEffects) {
        for (const auto &E : Kv.second) {
          if (E.ObjID == PID)
            if (Instruction *T = E.getTarget())
              InternalTargets.insert(T);
        }
      }
      for (User *U : OrigAlloc->users()) {
        if (U == Phi)
          continue;
        auto *UI = dyn_cast<Instruction>(U);
        if (!UI)
          return false;
        if (InternalTargets.count(UI))
          continue;
        auto AID = Aliases.getVirtualAlias(UI);
        if (AID && *AID == PID)
          continue;
        // An unaccounted-for user — identity-bail.
        return false;
      }
    }
    // No other VO references PID through a virtualRef field entry.
    // Cross-check both per-pred ExitInfos AND the analyzer's
    // live FieldStates / CurrentState. A VO synthesized earlier in
    // this same mergeStates iteration (e.g. a prior Case-C inner)
    // may carry a VirtualRef(PID) entry that isn't yet reflected in
    // any pred's ExitInfo — the analyzer's CurrentState is the only
    // place it lives until snapshotExitState runs. Missing this
    // check leaks identity through the in-flight synthesis.
    for (auto *EI : ExitInfos) {
      for (auto &Kv : EI->FieldStates) {
        if (Kv.first == PID)
          continue;
        for (auto &Off : Kv.second) {
          if (Off.second.isVirtualRef() && Off.second.getVirtualRef() == PID)
            return false;
        }
      }
    }
    for (auto &Kv : FieldStates) {
      if (Kv.first == PID)
        continue;
      for (auto &Off : Kv.second) {
        if (Off.second.isVirtualRef() && Off.second.getVirtualRef() == PID)
          return false;
      }
    }
  }

  // Loop-header cache lookup (forward infra; misses under single-pass).
  CaseCKey CacheKey;
  CacheKey.Block = BB;
  {
    SmallVector<jeandle::ObjectID, 4> Sorted(PerPredIDs.begin(),
                                             PerPredIDs.end());
    llvm::sort(Sorted);
    CacheKey.SourceIDs = std::move(Sorted);
  }
  // Peek the cache for an already-synthesised VO at this loop header. On a
  // hit we fall through into the full synthesize path reusing CachedExistingID
  // as the ObjectID (rather than returning early), so FieldStates[Cached] is
  // repopulated each iteration; the PHI emission below uses
  // getOrCreateLoopFieldPhi so per-offset PHI shells (and FieldStates' Value*)
  // stay stable across iterations.
  bool IsLoopHeader = LI.isLoopHeader(BB);
  jeandle::ObjectID CachedExistingID = jeandle::InvalidObjectID;
  if (IsLoopHeader) {
    auto CIt = CaseCVOCache.find(CacheKey);
    if (CIt != CaseCVOCache.end()) {
      jeandle::ObjectID Cached = CIt->second;
      if (Eligible.lookup(Cached))
        CachedExistingID = Cached;
      // If the cached ID was made ineligible (e.g. materialized in a previous
      // top-level pass), fall through and synthesize a fresh VO.
    }
  }

  // Pre-flight: compute per-entry merged type and per-pred input values for
  // every offset in the union of per-pred FieldStates. We do this BEFORE
  // creating the new VO so a type-mismatch bail leaves Result.VirtualObjects
  // unchanged.
  DenseSet<int64_t> OffsetsSet;
  for (unsigned i = 0; i < N; ++i) {
    auto FIt = ExitInfos[i]->FieldStates.find(PerPredIDs[i]);
    if (FIt == ExitInfos[i]->FieldStates.end())
      continue;
    for (auto &Kv : FIt->second)
      OffsetsSet.insert(Kv.first);
  }
  SmallVector<int64_t, 8> Offsets(OffsetsSet.begin(), OffsetsSet.end());
  llvm::sort(Offsets);

  struct OffsetPlan {
    int64_t Off;
    Type *PhiType = nullptr;
    bool AllSame = true;
    jeandle::FieldValue SoleValue;
    SmallVector<jeandle::FieldValue, 4> PerPredFVs;
  };
  SmallVector<OffsetPlan, 8> Plans;
  Plans.reserve(Offsets.size());

  for (int64_t Off : Offsets) {
    OffsetPlan P;
    P.Off = Off;
    P.PerPredFVs.resize(N);
    Type *PhiType = nullptr;
    bool AllPointer = true;
    for (unsigned i = 0; i < N; ++i) {
      jeandle::FieldValue FV = jeandle::FieldValue::unknown();
      auto FIt = ExitInfos[i]->FieldStates.find(PerPredIDs[i]);
      if (FIt != ExitInfos[i]->FieldStates.end()) {
        auto OIt = FIt->second.find(Off);
        if (OIt != FIt->second.end())
          FV = OIt->second;
      }
      P.PerPredFVs[i] = FV;
      if (FV.isUnknown())
        continue;
      Type *T =
          FV.isScalar() ? FV.getScalar()->getType() : FV.getDeclaredType();
      if (!T)
        return false;
      if (!T->isPointerTy() ||
          T->getPointerAddressSpace() != jeandle::AddrSpace::JavaHeapAddrSpace)
        AllPointer = false;
      if (!PhiType)
        PhiType = T;
      else if (PhiType != T && !(PhiType->isPointerTy() && T->isPointerTy() &&
                                 PhiType->getPointerAddressSpace() ==
                                     jeandle::AddrSpace::JavaHeapAddrSpace &&
                                 T->getPointerAddressSpace() ==
                                     jeandle::AddrSpace::JavaHeapAddrSpace))
        return false;
    }
    if (!PhiType)
      continue; // every pred has this slot unknown — nothing to record.
    if (AllPointer)
      PhiType = PointerType::get(F.getContext(),
                                 jeandle::AddrSpace::JavaHeapAddrSpace);
    P.PhiType = PhiType;
    // Detect AllSame across preds (shallowEquals comparison).
    jeandle::FieldValue First = P.PerPredFVs[0];
    P.SoleValue = First;
    bool Same = true;
    for (unsigned i = 1; i < N; ++i)
      if (!First.shallowEquals(P.PerPredFVs[i])) {
        Same = false;
        break;
      }
    P.AllSame = Same && !First.isUnknown();
    Plans.push_back(std::move(P));
  }

  // Synthesize (or REUSE on cache hit) the new VirtualObject. On a
  // cache hit we reuse the existing ID (and existing VirtualObjects slot)
  // and merely refresh its synthetic metadata. On a miss we duplicate Ref,
  // allocate a fresh ID via createVirtualObject, and tag it synthetic.
  jeandle::ObjectID NewID;
  if (CachedExistingID != jeandle::InvalidObjectID) {
    NewID = CachedExistingID;
    jeandle::VirtualObject &VO = *Result.VirtualObjects[NewID];
    VO.IsSynthetic = true;
    VO.SyntheticSourceIDs.assign(PerPredIDs.begin(), PerPredIDs.end());
    VO.SyntheticPhi = Phi;
  } else {
    auto NewVOUP = Ref.duplicate();
    NewID = Result.createVirtualObject(std::move(NewVOUP));
    jeandle::VirtualObject &NewVO = *Result.VirtualObjects[NewID];
    NewVO.IsSynthetic = true;
    NewVO.SyntheticSourceIDs.assign(PerPredIDs.begin(), PerPredIDs.end());
    NewVO.SyntheticPhi = Phi;
  }
  // Note: NewVO.AllocationCall is shared with Ref (the first per-pred VO).
  // It MUST NOT be used as a Materialize target or for RAUW — the synthetic
  // guard in materializeAt prevents that. We keep the field non-null only
  // because some accessors don't tolerate null AllocationCall (no current
  // path reaches them for a synthetic VO).

  // Rebuild the synthetic VO's Fields as the UNION of every per-pred VO's
  // Fields. duplicate() only copied Ref's (pred-0's) Fields, but the merged
  // FieldStates we build below spans the union of all preds' stored offsets.
  // A later sub-slot / wider load against the merged object scans VObj.Fields
  // to find the containing slot (processLoad); if a FieldDesc that came only
  // from a non-pred-0 path were missing, the scan would find nothing and the
  // load would silently fold to the default zero instead of the merged value.
  // getOrCreateFieldIndex is idempotent on exact matches (safe on the cached
  // loop-header VO and for Ref's already-present fields) and returns -1 on an
  // overlap/size conflict, which we treat as an incompatibility bail — exactly
  // like the type-mismatch bails in the Plans loop below. We poison NewID
  // before returning so the half-built VO is never observable; no PHI effects
  // have been committed at this point.
  {
    jeandle::VirtualObject &MergedVO = *Result.VirtualObjects[NewID];
    for (unsigned i = 0; i < N; ++i) {
      const jeandle::VirtualObject &PVO = *Result.VirtualObjects[PerPredIDs[i]];
      for (const auto &FD : PVO.Fields) {
        if (MergedVO.getOrCreateFieldIndex(FD.Offset, FD.LLVMType, DL) < 0) {
          Eligible[NewID] = false;
          return false;
        }
      }
    }
  }

  // Materialize inner virtuals if any per-pred entry is a VirtualRef. This
  // happens BEFORE we emit CreatePHI effects so the PHI inputs point at the
  // inner VO's original allocation (which the transform later RAUWs onto the
  // materialized invoke via NewAllocFor). Any failure here marks the new VO
  // ineligible and returns false; the per-entry CreatePHI effects we add
  // below (for NewID) get dropped at commit. Inner materializations may have
  // side-effects on snapshot state, but those are independently sound.
  DenseMap<int64_t, jeandle::FieldValue> Merged;
  jeandle::EffectList PendingPhiEffects;

  for (const OffsetPlan &P : Plans) {
    if (P.AllSame) {
      Merged[P.Off] = P.SoleValue;
      continue;
    }
    // Compute per-pred Value* for the synthesized PHI.
    SmallVector<Value *, 4> InValues;
    InValues.reserve(N);
    for (unsigned i = 0; i < N; ++i) {
      const jeandle::FieldValue &FV = P.PerPredFVs[i];
      Value *In = nullptr;
      if (FV.isUnknown()) {
        In = jeandle::FieldValue::defaultFor(P.PhiType);
      } else if (FV.isScalar()) {
        Value *V = FV.getScalar();
        if (V->getType() != P.PhiType) {
          Eligible[NewID] = false;
          return false;
        }
        In = V;
      } else if (FV.isMaterializedRef()) {
        if (!P.PhiType->isPointerTy()) {
          Eligible[NewID] = false;
          return false;
        }
        In = FV.getMaterialized();
      } else if (FV.isVirtualRef()) {
        if (!P.PhiType->isPointerTy()) {
          Eligible[NewID] = false;
          return false;
        }
        jeandle::ObjectID InnerID = FV.getVirtualRef();
        materializeAtPredFromExitInfo(InnerID, Preds[i], *ExitInfos[i],
                                      /*SkipGlobalRAUW=*/false, MatReason::Phi,
                                      /*TargetMerge=*/nullptr);
        if (!Eligible.lookup(InnerID)) {
          Eligible[NewID] = false;
          return false;
        }
        // Same defensive ExitInfo rewrite as the merge per-VO loop; see the
        // matching comment in mergeStates.
        ExitInfos[i]->FieldStates[PerPredIDs[i]][P.Off] =
            jeandle::FieldValue::materializedRef(
                Result.VirtualObjects[InnerID]->AllocationCall);
        In = Result.VirtualObjects[InnerID]->AllocationCall;
      } else {
        Eligible[NewID] = false;
        return false;
      }
      InValues.push_back(In);
    }
    // Route through the LoopFieldPhiCache so the per-(BB, NewID, Off) PHI
    // shell is REUSED across loop-fixpoint iterations. Same Value* across
    // iters keeps FieldStates structurally equivalent for the convergence
    // check, and - for non-header in-loop merge BBs - keeps the PHI alive
    // (OwnedLoopFieldPhis) so the preserved BlockExits[BB] does not reference
    // a PHI that rollback deletes. For BBs outside any loop the cache is
    // bypassed (getOrCreateLoopFieldPhi falls back to createUnparentedPhi).
    PHINode *NewPhi = getOrCreateLoopFieldPhi(BB, NewID, P.Off, P.PhiType, N,
                                              "pea.casec.field.phi");
    auto PE = std::make_unique<jeandle::CreatePHIEffect>();
    PE->Block = BB;
    // SeqNo assigned at drain time; see PendingMergePhis comment.
    PE->SeqNo = 0;
    PE->ObjID = NewID;
    PE->PhiInst = NewPhi;
    PE->PHIType = P.PhiType;
    for (unsigned i = 0; i < N; ++i) {
      PE->PHIIncomingValues.push_back(InValues[i]);
      PE->PHIIncomingBlocks.push_back(Preds[i]);
    }
    PendingPhiEffects.add(std::move(PE));
    if (P.PhiType->isPointerTy())
      Merged[P.Off] = jeandle::FieldValue::materializedRef(NewPhi);
    else
      Merged[P.Off] = jeandle::FieldValue::scalar(NewPhi);
  }

  // Route this synthetic VO's field-PHI effects to the caller's buffer (the
  // MergeProcessor's MergeEffects for a merge, PendingMergePhis[BB] for an
  // entry/single-pred path). They are assigned SeqNos at drain time.
  Out.addAll(PendingPhiEffects);
  CurrentState.addObject(NewID, jeandle::ObjectState());
  Eligible[NewID] = true;
  if (!Merged.empty())
    FieldStates[NewID] = std::move(Merged);
  if (RefLC != 0) {
    LockCounts[NewID] = RefLC;
    const auto &RefStack = ExitInfos[0]->LiveLockEnters.lookup(PerPredIDs[0]);
    if (!RefStack.empty())
      LiveLockEnters[NewID] = RefStack;
  }
  Aliases.addVirtualAlias(Phi, NewID);
  // Only insert if this is a fresh synthesis. On a cache hit the
  // entry already exists for the same key; emplace would be a no-op but we
  // skip explicitly to avoid the (CacheKey already moved-from) hazard above.
  if (IsLoopHeader && CachedExistingID == jeandle::InvalidObjectID)
    CaseCVOCache.emplace(std::move(CacheKey), NewID);
  return true;
}

void Analyzer::processInstruction(Instruction *I) {
  // Graal correspondence: PartialEscapeClosure.processNode /
  // processNodeInternal (PartialEscapeClosure.java:214-276). Graal dispatches
  // in three stages; this function mirrors them, adapted to LLVM's opcode-
  // keyed IR (Graal uses a Virtualizable interface + graph nodes):
  //
  //   (1) ALLOCATION STAGE (Graal requiresProcessing -> processVirtualizable):
  //       isJeandleAllocation -> processAllocation (NewInstanceNode/
  //       NewArrayNode.virtualize).
  //   (2) VIRTUALIZABLE STAGE (Graal hasVirtualInputs + Virtualizable ->
  //       processVirtualizable; MonitorEnter not-deleted branch ->
  //       materializeVirtualLocksBefore): the per-opcode handlers below —
  //       processStore/processLoad (Store/LoadField/Indexed/RawNode),
  //       propagatePointerAlias (LLVM pointer derivation, no Graal analog),
  //       processIntrinsic (LLVM intrinsics), foldICmpEquality
  //       (IsNullNode/UnaryOpLogicNode), processJavaOp (ArrayLength/LoadHub/
  //       InstanceOf/Monitor/MonitorExit/ArrayStoreCheck/...).
  //   (3) processNodeInputs STAGE (Graal:433-451):
  //   materializeAllVirtualOperands
  //       — materializes every virtual operand the specific fold did not itself
  //       account for.
  //
  // CONTRACT: a (2)-stage handler may return / early-exit ONLY once every
  // virtual operand has been folded or materialized. A handler that leaves a
  // virtual operand unaccounted for MUST fall through to (3) so the generic
  // escape path materializes it. (Graal guarantees this by always running
  // processNodeInputs after the virtualizable stage.)
  //
  // PHINodes are handled in processBlockPhis (which runs before this loop)
  // and have their alias status (Case B) or per-pred materialization (Case A)
  // recorded there. Re-walking them in the generic instruction dispatch would
  // hit the hasVirtualInputs fall-through and incorrectly trigger
  // materializeAllVirtualOperands, dropping a successfully-aliased Case-B PHI
  // back to materialized.
  if (isa<PHINode>(I))
    return;

  // Deoptimization state is carried in operand bundles, so record it before
  // any early-returning CallBase handler (notably allocation virtualization).
  if (auto *CB = dyn_cast<CallBase>(I)) {
    recordDeoptBundleState(CB);
    if (jeandle::pea::isJeandleAllocation(CB)) {
      processAllocation(CB);
      return;
    }
  }

  // Store/Load — try them regardless of the hasVirtualInputs gate.
  // processStore/processLoad resolve the pointer through GEPs/casts and
  // early-exit if it doesn't bottom out on a virtual base.
  if (auto *SI = dyn_cast<StoreInst>(I)) {
    // processStore returns true if the POINTER side was a virtual,
    // i.e. it consumed the store. If false, we MUST fall through to the
    // generic hasVirtualInputs path so a VALUE-side virtual reference
    // (e.g. `store ptr %virtAlloc, ptr @G`) escapes correctly instead of
    // silently surviving in IR as a `store poison, ptr @G` after
    // EliminateAllocation RAUWs the virtual to PoisonValue.
    if (processStore(SI))
      return;
    // Fall through.
  } else if (auto *LI = dyn_cast<LoadInst>(I)) {
    processLoad(LI);
    return;
  }

  // TODO(unsafe-inliner): see the access dispatch (processStore/processLoad).
  // atomicrmw/cmpxchg falls through to the generic-escape path below,
  // materializing conservatively.

  // Graal materializeVirtualLocksBefore (PartialEscapeClosure.java:263-264,
  // 641-652): under strict lock order, a REAL (non-virtualized) monitorenter
  // must first materialize every still-virtual object holding a shallower live
  // lock, so each such object's re-emitted lock lands below this real lock on
  // the lightweight-locking thread lock stack (preserving lexical nesting).
  //
  // Placement: Graal reaches this from processNodeInternal's hasVirtualInputs-
  // gated virtualizable stage because a Graal MonitorEnterNode carries a
  // stateAfter FrameState that references the virtual object. Jeandle models
  // deopt state through operand bundles instead of normal FrameState inputs on
  // monitorenter, so a non-virtual receiver has NO virtual input and never
  // enters the gate below — the cascade is therefore checked here, outside the
  // gate. A virtual-receiver monitorenter is handled by foldMonitorEnter inside
  // the gate (elision + its own elide-path pre-cascade), so this fires only
  // when the receiver does NOT resolve to a virtual.
  if (StrictLockOrder) {
    if (auto *CB = dyn_cast<CallBase>(I)) {
      if (jeandle::pea::isJeandleMonitorEnter(CB) && CB->arg_size() >= 1) {
        auto RecvID = jeandle::pea::resolveVirtualRef(
            CB->getArgOperand(0), CurrentState, Aliases, DL);
        if (!RecvID)
          materializeVirtualLocksBefore(CB);
      }
    }
  }

  // Other virtual-input consumers (access folding + scalar-replaced inputs).
  if (Aliases.hasVirtualInputs(I)) {
    // Pointer-derivation forwards the virtual alias to the derived pointer so
    // downstream load/store handlers can pick up the base via the alias map.
    // SelectInst is included: when both arms resolve to the same virtual
    // ObjectID, the Select denotes that virtual on every execution path.
    // propagatePointerAlias additionally guards Select on per-arm byte offset
    // (mirroring processBlockPhis' AnyDerived): a Select whose arms carry a
    // non-zero field offset materializes rather than alias-forwarding, since
    // resolveFieldOffset has no Select case and would otherwise lose the
    // offset.
    if (isa<GetElementPtrInst>(I) || isa<BitCastInst>(I) ||
        isa<AddrSpaceCastInst>(I) || isa<FreezeInst>(I) || isa<SelectInst>(I)) {
      propagatePointerAlias(I);
      return;
    }
    // Known non-escaping LLVM intrinsics (assume, lifetime markers,
    // invariant markers, debug intrinsics, ...) are no-ops for PEA. The
    // virtual stays virtual and the call is left alone in IR (some are
    // DCE'd downstream; others are harmless). launder/strip.invariant.group
    // forward the argument's virtual alias. Must run BEFORE the JavaOp fold +
    // generic-escape fall-through.
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      if (processIntrinsic(II))
        return;
      // default: fall through to the ICmp / JavaOp / generic-escape path.
    }
    // Deferred virtualization handlers — NOT WIRED yet:
    //   - TODO: processArrayCopy / processMemSet — llvm.memcpy/memmove
    //     (System.arraycopy) and llvm.memset (Arrays.fill). The only
    //     llvm.memset producer today is jeandle.new_instance's lower-phase=1
    //     template, inlined AFTER PEA, so neither shape reaches PEA yet.
    //   - TODO: llvm.reachability_fence — upstream LLVM this fork tracks
    //     does not define Intrinsic::reachability_fence, and the frontend
    //     emits no analogue.
    //   - TODO: ObjectClone / FinalFieldBarrier / EnsureVirtualized — no
    //     jeandle.clone / final_field_barrier / ensure_virtualized JavaOp
    //     exists in the frontend inventory.
    //   - TODO(get_class): the frontend DOES emit jeandle.get_class
    //     (jeandleIntrinsicLowering.cpp _getClass), but the Graal fold
    //     (GetClassNode.virtualize:91-99 — replaceWithValue(constant Class via
    //     constantReflection.asJavaClass(type))) is NOT implemented. It needs
    //     (a) a new GetJavaMirror(Klass)->uintptr_t VMCallback and (b) the
    //     ability to embed a GC'd JavaHeap oop constant in IR — every existing
    //     fold emits only a primitive or a CHeap Klass pointer
    //     (foldLoadKlass), so there is no precedent for an embedded GC oop
    //     (statepoint/barrier treatment). Until then a virtual receiver of
    //     jeandle.get_class hits this fall-through and materializes (sound,
    //     conservative).
    //
    // Current frontend JavaOp inventory (grep `jeandle.[a-z_]+` in
    // jeandle-jdk/src/hotspot/share/jeandle/): array_store_check, arraylength,
    // card_table_barrier, check_if_value_based, check_inflated,
    // check_instanceof, check_klass_subtype, check_klass_subtype_slow_path,
    // checkcast, clear_oop_in_lock_stack_top, current_thread,
    // decrement_lock_count, get_class, get_stack_pointer, idiv,
    // increment_lock_count, instanceof, irem, ldiv, load_klass, lrem,
    // monitorenter_*, monitorexit_*, new_instance, newarray, personality,
    // post_barrier, pre_barrier,
    // safepoint_poll, try_acquire_monitor_lock, try_release_monitor_lock. When
    // the frontend grows a new JavaOp, wire its fold in processJavaOp and add
    // the isJeandle* predicate in PartialEscapeUtils.{h,cpp}.
    //
    // Equality compare against a virtual pointer folds (virtuals are never
    // null; identity comparison). Non-equality ICmp on virtual heap pointers
    // (slt/sgt/...) is UB on GC pointers; fall through to conservative
    // materialization.
    if (auto *ICmp = dyn_cast<ICmpInst>(I)) {
      if (foldICmpEquality(ICmp))
        return;
    }
    // Recognise JavaOps that read/inspect a virtual receiver and try to
    // constant-fold them. processJavaOp returns true if the JavaOp was
    // handled (whether by folding to a constant or by being a known-safe
    // non-escaping shape that needs no transform).
    if (auto *CB = dyn_cast<CallBase>(I)) {
      if (processJavaOp(CB))
        return;
      // Fall through to the generic-escape path for unrecognised calls.
    }
    // Any other consumer of a virtual operand. Materialize when sound; if a
    // derived-GEP virtual operand is present, markIneligible instead so it
    // doesn't resolve to poison (see materializeVirtualOperandsSafely).
    materializeVirtualOperandsSafely(I);
    return;
  }

  // Scalar-replaced inputs: nothing to do.
}

// Allocation virtualization. Graal correspondence:
// VirtualizerToolImpl.createVirtualObject (VirtualizerToolImpl.java:345-369) —
// build the VirtualObject (VirtualInstanceNode/VirtualArrayNode in Graal), give
// it an ObjectState (presence marker + lock list), add the virtual alias
// (virtual <-> itself), and record the EliminateAllocation effect. Jeandle's
// AllocSiteToVO cache (ObjectIDs stable across loop-fixpoint iterations) and
// VMCallbacks gates (HasFinalizer / CanVirtualize) are Jeandle-specific
// adaptation; the core creation+alias+eliminate sequence matches Graal.
void Analyzer::processAllocation(CallBase *CB) {
  // In StopNewInLoopNest mode (transiently set by processLoop at a
  // top-level nest whose maximum depth exceeds JeandlePEALoopCutoff),
  // refuse to register NEW virtual allocations inside the nest, but leave
  // every other state intact — already-virtual objects (registered in a
  // shallower enclosing scope, or before the nest entry) continue to be
  // tracked and folded as usual.
  //
  // TODO(ensure-virtualized): GRAAL DIVERGENCE — the refusal below is
  // unconditional, but Graal's processVirtualizable
  // (PartialEscapeClosure.java:287-303) exempts allocations whose usages
  // contain an EnsureVirtualizedNode (the mayEnsureVirtualized scan): such
  // an allocation is still virtualised PAST EscapeAnalysisLoopCutoff
  // (default 20). The marker is produced only by GraalDirectives
  // .ensureVirtualized / ensureVirtualizedHere, lowered by the graph
  // builder to an EnsureVirtualizedNode (EnsureVirtualizedNode.java:50-110;
  // StandardGraphBuilderPlugins.java:2082-2095). Jeandle's frontend has no
  // ensure_virtualized JavaOp / intrinsic, so mayEnsureVirtualized would be
  // uniformly false here — there is currently nothing to override.
  //
  // The override is one leg of a three-part Graal design that must be
  // wired up together:
  //  (1) Override at this site — needs an IR marker the analyser recognises
  //      plus an EnsureVirtualized bit on ObjectState (serialised through
  //      clone / takeLoopSnapshot / restoreLoopSnapshot).
  //  (2) Materialisation guard — Graal's ensureMaterialized
  //      (PartialEscapeClosure.java:538-562) throws RetryableBailoutException
  //      (a non-permanent bailout: retry the whole compilation without PEA)
  //      when an ensure-virtualised object must be materialised inside a
  //      deep nest, which is what keeps the override from going exponential
  //      in nest depth. Jeandle is -fno-exceptions with no per-pass
  //      bailout, so this leg is deopt-adjacent and deferred — see the
  //      matching note in ensureMaterialized below.
  //  (3) Flag bookkeeping — AND-reduce the bit across merge predecessors
  //      with setEnsureVirtualized(false) where not all preds agree
  //      (PartialEscapeClosure.java:991-995, 1321-1324, 1500-1503) and
  //      propagate it transitively in stripKilledLoopLocations (:685-714).
  //      Jeandle's ObjectState has no such bit, so the per-pred materialisation
  //      sites that route through materializeAtPredFromExitInfo currently do no
  //      downgrade either — tagged TODO(ensure-virtualized) at their entry
  //      points (materializeAndBuildPhi, which the AllMaterialized-divergence
  //      arm of mergeObjectState routes through; and synthesizeCaseC).
  //
  // Soundness: the unconditional return below is CONSERVATIVE — Jeandle
  // merely virtualises less than Graal in deep nests; it never miscompiles.
  // Were an ensure-virtualised marker to appear today, the object would
  // simply stay in IR and be materialised at its escape point, preserving
  // correctness while violating the (advisory) directive.
  if (CurrentMode == Mode::StopNewInLoopNest)
    return;

  // In MATERIALIZE_ALL mode the analyzer registers the VO normally
  // (so intra-block processLoad/processStore folds against the new
  // FieldStates), then defers a Materialize effect to end-of-processBlock so
  // the alloc is re-emitted at the block's terminator IP — by which time all
  // stores have updated FieldStates so the materialised invoke captures the
  // final field values. In MATERIALIZE_ALL, a virtualizable node is virtualised
  // AND immediately ensure-materialized before the next fixed node. The
  // end-of-block emission is the practical compromise: intra-block folds
  // work, end-of-block emission is dominance-safe.
  const bool VirtualiseThenMaterialise = (CurrentMode == Mode::MaterializeAll);

  // Stable VO-per-allocation-site cache. Re-processing the same alloc
  // inside a loop fixpoint iteration must produce the same ObjectID so the
  // per-block exit snapshots can be compared for convergence. The cache is
  // monotonic (entries are never removed), and ineligibility carries over
  // across rollback boundaries.
  if (auto It = AllocSiteToVO.find(CB); It != AllocSiteToVO.end()) {
    jeandle::ObjectID ID = It->second;
    if (!Eligible.lookup(ID))
      return; // poisoned in a prior iteration — leave the original alloc.
    // Re-register the alias and the virtual ObjectState in CurrentState.
    // Aliases.addVirtualAlias asserts !already-aliased, so call only when
    // the cache restore has wiped the entry (the common case under
    // restoreLoopSnapshot, which restores Aliases to its pre-loop state).
    if (!Aliases.getVirtualAlias(CB))
      Aliases.addVirtualAlias(CB, ID);
    if (!CurrentState.hasObjectState(ID))
      CurrentState.addObject(ID, jeandle::ObjectState());
    // Re-emit the EliminateAllocation effect. The pre-iter snapshot has
    // wiped BlockEffects[CB->getParent()] of this iteration's prior copy,
    // and addBlockEffect doesn't dedup, so this is exactly the right place.
    auto E = std::make_unique<jeandle::EliminateAllocationEffect>();
    E->Block = CB->getParent();
    E->Target = CB;
    E->SeqNo = Result.nextSeqNo();
    E->ObjID = ID;
    Result.addBlockEffect(std::move(E));
    // Also re-enqueue the per-block materialise on the cache-hit
    // path. For an InvokeInst alloc, the alloc IS the terminator of its
    // block, so we cannot drain at end-of-current-block (the alloc only
    // gets registered AFTER our drain hook would fire). Defer to the
    // NORMAL-dest block: that's where downstream processLoad/processStore
    // fold the VO's FieldStates, and end-of-that-block is the correct
    // dominance-safe materialise IP.
    if (VirtualiseThenMaterialise) {
      BasicBlock *MatBB = CB->getParent();
      if (auto *II = dyn_cast<InvokeInst>(CB))
        MatBB = II->getNormalDest();
      PendingMaterializeAllVOs[MatBB].push_back(ID);
    }
    return;
  }

  // Loop-body allocations are permitted as virtualization candidates. The
  // dominance check in materializeAt (DT.dominates(VI, SafeIP) per
  // FieldStates entry) is what protects the analyzer from forming an unsound
  // replay: any field whose stored value is defined later in the loop body
  // cannot dominate the SafeIP just after the alloc, so the object becomes
  // ineligible and survives in IR untouched. materializeBeforeLoops
  // independently force-materializes any virtual that is still virtual at a
  // loop preheader's exit (modulo loops drained by the loop fixpoint's
  // pessimistic fallback), so objects allocated BEFORE the loop and
  // surviving into the loop are also handled.

  uintptr_t Klass = jeandle::pea::extractAllocationKlass(CB);
  if (Klass == 0)
    return;

  const bool IsInstance = jeandle::pea::isJeandleNewInstance(CB);
  const bool IsArray = jeandle::pea::isJeandleNewArray(CB);
  assert((IsInstance ^ IsArray) &&
         "allocation must be either instance or array");
  assert((!IsArray || CB->arg_size() == 5) &&
         "jeandle.new_array must use the 5-arg protocol");
  (void)IsArray;

  // Refuse to virtualize identity-sensitive allocations.
  // HasFinalizer: classes that override finalize() require HotSpot's
  // RegisterFinalizer hook to fire at the original allocation site;
  // eliding the alloc would skip finalizer registration and break
  // user-visible semantics.
  // CanVirtualize: Reference / Thread (and subtypes) have lifecycle
  // tracked by global runtime state (pending-reference list, thread
  // list); virtualizing them would silently elide that tracking.
  // Both checks fire only on instance allocations (arrays cannot have
  // finalizers, and HotSpot's finalizer / Reference / Thread machinery
  // is keyed on InstanceKlass identity). Lit tests without a cblog
  // continue to virtualize as before because both pointers are
  // nullptr and we fall through to the legacy default-virtualize path.
  if (IsInstance) {
    if (const jeandle::VMCallbacks *VMCB = jeandle::getVMCallbacks()) {
      if (VMCB->HasFinalizer && VMCB->HasFinalizer(Klass))
        return;
      if (VMCB->CanVirtualize && !VMCB->CanVirtualize(Klass))
        return;
    }
  }

  std::unique_ptr<jeandle::VirtualObject> VO;

  if (IsInstance) {
    auto Size = jeandle::pea::extractInstanceSize(CB);
    if (!Size)
      return;
    VO = std::make_unique<jeandle::VirtualObject>(
        jeandle::InvalidObjectID, jeandle::VirtualObject::Instance, CB);
    VO->Klass = Klass;
    VO->SizeInBytes = *Size;
  } else {
    auto Length = jeandle::pea::extractArrayLength(CB);
    if (!Length)
      return;
    if (*Length > MaximumEscapeAnalysisArrayLength)
      return;
    VO = std::make_unique<jeandle::VirtualObject>(
        jeandle::InvalidObjectID, jeandle::VirtualObject::Array, CB);
    VO->Klass = Klass;
    VO->ArrayLength = *Length;
    auto *Size = dyn_cast<ConstantInt>(CB->getArgOperand(2));
    auto *BaseOffset = dyn_cast<ConstantInt>(CB->getArgOperand(3));
    auto *LengthLimit = dyn_cast<ConstantInt>(CB->getArgOperand(4));
    if (!Size || !BaseOffset || !LengthLimit)
      return;
    VO->SizeInBytes = static_cast<uint32_t>(Size->getZExtValue());
    VO->ArrayBaseOffset = static_cast<uint32_t>(BaseOffset->getZExtValue());
    VO->ArrayLengthLimit = static_cast<uint32_t>(LengthLimit->getZExtValue());

    // Populate per-element metadata so matchArrayElementGEP can match
    // typed-GEP / symbolic-byte-offset element accesses. The 5-arg
    // new_array protocol carries ArrayBaseOffset directly; when the
    // element kind is known, VMConstants are used only to derive the
    // element scale and to assert the frontend-provided base offset agrees
    // with the VM layout. If the VMCallback is unregistered or cannot
    // identify the element kind, leave ArrayElementType nullptr and keep
    // the protocol-provided ArrayBaseOffset so the resolveAccess header
    // guard remains precise.
    const jeandle::VMConstants VMConsts =
        jeandle::VMConstants::fromModule(*F.getParent());
    if (auto Kind = jeandle::pea::elementTypeForArrayKlass(Klass)) {
      if (Type *ElemTy =
              jeandle::pea::llvmElementTypeFor(*Kind, F.getContext())) {
        VO->ArrayElementType = ElemTy;
        assert(VO->ArrayBaseOffset ==
                   static_cast<uint32_t>(VMConsts.arrayBaseOffsetFor(*Kind)) &&
               "new_array base_offset must match VM array layout");
        VO->ArrayIndexScale =
            static_cast<uint32_t>(VMConsts.elementSizeFor(*Kind));
      }
    }
  }

  // Graal VirtualizerToolImpl.createVirtualObject (345-369) +
  // replaceWithVirtual (293-300): assign the object id, install a virtual
  // ObjectState, register the virtual alias, "delete" the original allocation,
  // and account the delta. Jeandle's virtualize is structurally the same; the
  // only differences are forced by the deferred-transform design (no IR
  // mutation during analysis):
  //   - the id is cached per allocation site (AllocSiteToVO) so loop-fixpoint
  //     re-processing yields a STABLE ObjectID for the convergence comparison;
  //   - instead of deleting the node (Graal effects.deleteNode) an
  //     EliminateAllocation effect is emitted, applied by the transform later;
  //   - the per-field FieldValue tracking lives in the analyzer-side
  //     FieldStates map (Jeandle's counterpart to Graal ObjectState.entries;
  //     the on-VO ObjectState carries no field state) — see the class comment.
  jeandle::ObjectID ID = Result.createVirtualObject(std::move(VO)); // :354-359
  AllocSiteToVO[CB] = ID; // Jeandle: stable id per site (loop fixpoint).
  Aliases.addVirtualAlias(CB, ID); // addVirtualAlias :361
  // Register a Virtual ObjectState — a presence marker carrying only Kind ==
  // Virtual (addObject :360). resolveVirtualRef only needs the slot present;
  // the per-field FieldValue tracking lives in FieldStates (see class comment).
  CurrentState.addObject(ID, jeandle::ObjectState());
  Eligible[ID] = true;

  // replaceWithVirtual analog (Graal effects.deleteNode :362). Deferred: emit
  // an EliminateAllocation effect the transform applies to erase the alloc.
  auto E = std::make_unique<jeandle::EliminateAllocationEffect>();
  E->Block = CB->getParent();
  E->Target = CB;
  E->SeqNo = Result.nextSeqNo();
  E->ObjID = ID;
  Result.addBlockEffect(std::move(E));

  ++Result.VirtualizationDelta; // effects.addVirtualizationDelta(1) :362-363
  --Result.AllocationDelta;
  // A new virtual object was registered.
  ++JeandlePEAVirtualized;

  // Enqueue end-of-block materialise under VirtualiseThenMaterialise.
  // For an InvokeInst alloc the alloc IS the block terminator; drain in
  // the normal destination instead so subsequent field accesses in that
  // block fold first.
  if (VirtualiseThenMaterialise) {
    BasicBlock *MatBB = CB->getParent();
    if (auto *II = dyn_cast<InvokeInst>(CB))
      MatBB = II->getNormalDest();
    PendingMaterializeAllVOs[MatBB].push_back(ID);
  }
}

std::optional<int64_t> Analyzer::resolveAccess(Value *Ptr,
                                               jeandle::ObjectID BaseID) {
  jeandle::VirtualObject &VObj = *Result.VirtualObjects[BaseID];

  // Array-element GEP fast path: for array VOs with populated element
  // metadata, recognise the typed-element GEP the abstract interpreter emits
  // for indexed accesses. matchArrayElementGEP returns {idx, etype} on a
  // recognised shape; idx is a ConstantInt for constant indices (use the
  // canonical byte offset) and any other Value for symbolic indices (force
  // materialization, matching the "constant index only" policy).
  if (VObj.isArray() && VObj.ArrayElementType) {
    if (auto *G = dyn_cast<GetElementPtrInst>(Ptr)) {
      if (auto Match = VObj.matchArrayElementGEP(G, DL)) {
        if (auto *CI = dyn_cast<ConstantInt>(Match->Index)) {
          int64_t Cidx = CI->getSExtValue();
          if (Cidx < 0 || static_cast<uint64_t>(Cidx) >= VObj.ArrayLength)
            return std::nullopt; // out of bounds
          return static_cast<int64_t>(VObj.ArrayBaseOffset) +
                 Cidx * static_cast<int64_t>(VObj.ArrayIndexScale);
        }
        return std::nullopt; // symbolic index
      }
    }
  }

  // General constant-offset resolver. A non-constant GEP offset yields
  // nullopt (caller materializes).
  std::optional<int64_t> Offset = jeandle::pea::resolveFieldOffset(Ptr, DL);
  if (!Offset)
    return std::nullopt;

  // Header-offset guard: mark/klass words are VM metadata, not Java fields,
  // and must not be virtualized into a field slot. Instances are guarded by
  // instanceBaseOffset; arrays are mirrored by ArrayBaseOffset (a raw GEP
  // into the array header, e.g. an offset-0 store to the mark word, would
  // otherwise replay as a Java-field write on materialization).
  if (VObj.isInstance()) {
    const jeandle::VMConstants VMConsts =
        jeandle::VMConstants::fromModule(*F.getParent());
    if (*Offset < VMConsts.instanceBaseOffset())
      return std::nullopt;
  } else if (VObj.isArray()) {
    if (*Offset < static_cast<int64_t>(VObj.ArrayBaseOffset))
      return std::nullopt;
  }

  return Offset;
}

bool Analyzer::processStore(StoreInst *SI) {
  Value *Ptr = SI->getPointerOperand();
  Value *Val = SI->getValueOperand();

  auto BaseID = jeandle::pea::resolveVirtualRef(Ptr, CurrentState, Aliases, DL);
  if (!BaseID)
    return false;

  jeandle::VirtualObject &VObj = *Result.VirtualObjects[*BaseID];

  // The store could not be virtualized. Keep the base's ORIGINAL allocation
  // real (markIneligible) so a pre-computed derived store address — e.g. a
  // symbolic-index GEP computed earlier in the block — stays valid:
  // materializing the base via the gate would insert a fresh alloc at the
  // store that does not dominate that derived GEP, so its pointer operand
  // would resolve to poison. The stored value, if itself virtual, must also be
  // kept real or it would be eliminated (NeverEscapes -> poison) while the
  // store survives, writing poison into the base's field. The store itself
  // stays as a real store (no EliminateStoreEffect is emitted); returning true
  // keeps processInstruction from re-running the gate on it.
  auto bailKeepingOperandsReal = [&] {
    markIneligible(*BaseID);
    if (auto RefID =
            jeandle::pea::resolveVirtualRef(Val, CurrentState, Aliases, DL))
      markIneligible(*RefID);
  };

  // Shared offset resolution (array-element GEP fast path + constant-offset
  // resolver + header guard). See resolveAccess. Unresolved offset (symbolic
  // array index, non-constant GEP, header offset) -> bail.
  std::optional<int64_t> Offset = resolveAccess(Ptr, *BaseID);
  if (!Offset) {
    bailKeepingOperandsReal();
    return true;
  }

  // TODO(unsafe-inliner): see the access dispatch (processStore/processLoad).
  // Unsafe.put{Int,Long,Short}-into-byte-array decomposition.

  // Type-overlap validation via VirtualObject::getOrCreateFieldIndex. We don't
  // actually use the returned index (FieldStates is keyed by raw offset), but
  // -1 means an overlap/size conflict, or an unknown-size value type such as a
  // vector/struct — bail either way.
  if (VObj.getOrCreateFieldIndex(*Offset, Val->getType(), DL) < 0) {
    bailKeepingOperandsReal();
    return true;
  }

  // Compute the FieldValue for the stored Value.
  if (auto RefID =
          jeandle::pea::resolveVirtualRef(Val, CurrentState, Aliases, DL)) {
    // Nested virtual reference. Recursive materialization handles this at
    // materialize time by first materializing the inner object then storing
    // its materialized pointer into the outer's field. We record the nested
    // reference here and let materializeAt rewrite it later.
    FieldStates[*BaseID][*Offset] =
        jeandle::FieldValue::virtualRef(*RefID, Val->getType());

    auto E = std::make_unique<jeandle::EliminateStoreEffect>();
    E->Block = SI->getParent();
    E->Target = SI;
    E->SeqNo = Result.nextSeqNo();
    E->ObjID = *BaseID;
    Result.addBlockEffect(std::move(E));
    return true;
  }
  Val = Aliases.resolveScalarAlias(Val);
  FieldStates[*BaseID][*Offset] = jeandle::FieldValue::scalar(Val);

  auto E = std::make_unique<jeandle::EliminateStoreEffect>();
  E->Block = SI->getParent();
  E->Target = SI;
  E->SeqNo = Result.nextSeqNo();
  E->ObjID = *BaseID;
  Result.addBlockEffect(std::move(E));
  return true;
}

void Analyzer::processLoad(LoadInst *LI) {
  Value *Ptr = LI->getPointerOperand();

  auto BaseID = jeandle::pea::resolveVirtualRef(Ptr, CurrentState, Aliases, DL);
  if (!BaseID)
    return;

  jeandle::VirtualObject &VObj = *Result.VirtualObjects[*BaseID];

  // Shared offset resolution (array-element GEP fast path + constant-offset
  // resolver + header guard). See resolveAccess.
  std::optional<int64_t> Offset = resolveAccess(Ptr, *BaseID);
  if (!Offset) {
    markIneligible(*BaseID);
    return;
  }

  Type *LoadTy = LI->getType();

  // Locate the FieldDesc whose recorded range contains the load. We detect the
  // relationship between the load and the stored slot so we can bail correctly:
  // a load that straddles slot boundaries (overlaps without being contained)
  // forces materialization here, and a load contained in a wider slot but at a
  // nonzero within-slot offset is a sub-slot read bailed on below (see
  // WithinSlotByteOff). Loads that exactly cover the slot, or read it at the
  // same offset, proceed to coerceToType.
  uint64_t LoadBits = LoadTy->isSized() ? DL.getTypeSizeInBits(LoadTy) : 0;
  uint8_t LoadByteSize = static_cast<uint8_t>((LoadBits + 7) / 8);
  int64_t LoadEnd = *Offset + static_cast<int64_t>(LoadByteSize);
  int64_t EntryOffset = *Offset;
  bool OverlapsNoncontained = false;
  // Fields are sorted by Offset; linear scan is fine for the small per-object
  // field counts we see in practice.
  for (const auto &F : VObj.Fields) {
    int64_t FEnd = F.Offset + static_cast<int64_t>(F.ByteSize);
    if (FEnd <= *Offset)
      continue;
    if (F.Offset >= LoadEnd)
      break;
    // Some overlap. Contained iff F covers [*Offset, LoadEnd).
    if (F.Offset <= *Offset && FEnd >= LoadEnd) {
      EntryOffset = F.Offset;
    } else {
      OverlapsNoncontained = true;
    }
    break;
  }
  if (OverlapsNoncontained) {
    // TODO(unsafe-inliner): see the access dispatch (processStore/processLoad).
    // Any straddling load conservatively forces materialization.
    markIneligible(*BaseID);
    return;
  }

  // A nonzero within-slot offset means the load reads PART of a wider stored
  // field (a sub-slot / "incomplete field" read). Such partial-field reads are
  // not folded, so we bail to materialization. The bail runs after the
  // Unknown/default-value fold below, so a sub-slot read of a never-written
  // field still folds to its default (zero for every primitive).
  int64_t WithinSlotByteOff = *Offset - EntryOffset;

  const jeandle::FieldValue *Existing = nullptr;
  auto It = FieldStates.find(*BaseID);
  if (It != FieldStates.end()) {
    auto It2 = It->second.find(EntryOffset);
    if (It2 != It->second.end())
      Existing = &It2->second;
  }

  if (!Existing || Existing->isUnknown()) {
    // Default value for a never-written field.
    Constant *Def = jeandle::FieldValue::defaultFor(LoadTy);
    auto E = std::make_unique<jeandle::ReplaceLoadEffect>();
    E->Block = LI->getParent();
    E->Target = LI;
    E->Replacement = Def;
    E->SeqNo = Result.nextSeqNo();
    E->ObjID = *BaseID;
    Result.addBlockEffect(std::move(E));
    Aliases.addScalarAlias(LI, Def);
    return;
  }

  // Sub-slot read of a non-Unknown field: a partial-field load is not folded
  // (see WithinSlotByteOff above). Force the object to materialize so the
  // original load survives in IR.
  if (WithinSlotByteOff != 0) {
    markIneligible(*BaseID);
    return;
  }

  if (Existing->isScalar()) {
    Value *V = Aliases.resolveScalarAlias(Existing->getScalar());
    // Coerce to LoadTy: same-type passthrough or same-bit-width primitive↔
    // primitive reinterpret (bitcast). Pointer↔primitive, cross-AS pointer
    // pairs, and any cross-width mismatch (narrowing/widening) bail to
    // ineligible per the stable-slot-kind and width policies.
    Value *Coerced = coerceToType(V, LoadTy, LI);
    if (!Coerced) {
      markIneligible(*BaseID);
      return;
    }
    auto E = std::make_unique<jeandle::ReplaceLoadEffect>();
    E->Block = LI->getParent();
    E->Target = LI;
    E->Replacement = Coerced;
    E->SeqNo = Result.nextSeqNo();
    E->ObjID = *BaseID;
    Result.addBlockEffect(std::move(E));
    Aliases.addScalarAlias(LI, Coerced);
    return;
  }

  if (Existing->isVirtualRef()) {
    // Nested-virtual load: loading a field whose tracked value is another
    // virtual yields that other virtual (still virtual!) and forwards the
    // load to it. Forward the load to the inner virtual's allocation
    // Value and install a virtual alias from the load to
    // InnerID so downstream access handlers (foldArrayLength, foldLoadKlass,
    // etc.) and the generic escape detection see %loaded as a reference to the
    // inner virtual. If the inner later materializes / escapes, the
    // analyzer's existing nested-virtual machinery (a) rewrites every
    // other tracking site (FieldStates, alias map) to the materialized
    // pointer, and (b) at transform time, applyMaterialize records the
    // OrigInnerAlloc → NewInv definition; the point-sensitive resolution
    // sub-pass rewrites the materialized uses to the new invoke.
    jeandle::ObjectID InnerID = Existing->getVirtualRef();

    if (!Eligible.lookup(InnerID)) {
      // The inner was abandoned by some upstream decision — its alloc
      // will survive in IR, but we should not silently keep forwarding to
      // it for the outer because forwarding can mask a missing
      // materialization. Bail conservatively on the outer.
      markIneligible(*BaseID);
      return;
    }

    const jeandle::ObjectState *InnerOS =
        CurrentState.getObjectStateOptional(InnerID);
    if (!InnerOS) {
      // The inner's ObjectState isn't live in this block (shouldn't happen
      // for a VirtualRef field entry that was inherited alongside the
      // outer, but defend). Bail on the outer only; do not poison the
      // inner — it may be cleanly virtualizable on other paths.
      markIneligible(*BaseID);
      return;
    }

    Value *Repl = nullptr;
    if (InnerOS->isMaterialized()) {
      // Defensive: the analyzer rewrites VirtualRef → MaterializedRef on
      // every materialization site, so a VirtualRef field entry whose
      // inner has flipped to Materialized at this point would normally be
      // stale. If we still observe it, fall back to the materialized
      // pointer — same shape as the MaterializedRef branch below.
      Repl = InnerOS->getMaterializedValue();
    } else {
      jeandle::VirtualObject &InnerVO = *Result.VirtualObjects[InnerID];
      Repl = InnerVO.AllocationCall;
    }
    if (!Repl) {
      markIneligible(*BaseID);
      return;
    }

    // Type-compatibility. For ordinary reference loads, both LoadTy and the
    // inner allocation are `ptr addrspace(1)` and coerceToType returns Repl
    // unchanged. Cross-address-space or ptr↔primitive mismatch bails per the
    // stable-slot-kind invariant. (Sub-slot pointer loads were already rejected
    // by the WithinSlotByteOff bail above.) We don't poison InnerID because
    // other paths may still be able to virtualize it.
    Value *Coerced = coerceToType(Repl, LoadTy, LI);
    if (!Coerced) {
      markIneligible(*BaseID);
      return;
    }

    auto E = std::make_unique<jeandle::ReplaceLoadEffect>();
    E->Block = LI->getParent();
    E->Target = LI;
    E->Replacement = Coerced;
    E->SeqNo = Result.nextSeqNo();
    E->ObjID = *BaseID;
    Result.addBlockEffect(std::move(E));

    // Install the virtual alias only when the inner is still virtual at
    // this point. If we fell back to a materialized pointer above, mirror
    // the MaterializedRef branch and register a scalar alias so downstream
    // resolveVirtualRef queries see through to the materialized value.
    if (InnerOS->isMaterialized())
      Aliases.addScalarAlias(LI, Coerced);
    else
      Aliases.addVirtualAlias(LI, InnerID);
    return;
  }

  if (Existing->isMaterializedRef()) {
    // A virtual's reference field carries a materialized pointer (e.g.
    // produced by a field-PHI synthesis at a merge block). Forward the load
    // to the materialized value, matching the Scalar handler.
    Value *V = Existing->getMaterialized();
    if (!V) {
      markIneligible(*BaseID);
      return;
    }
    // A materialized ref slot can only be loaded back as a pointer (and in
    // practice, since LLVM 17 uses opaque pointers, only as the same
    // ptr-AS). coerceToType bails on ptr↔primitive (stable-slot-kind) and
    // cross-AS pointer pairs. (Partial pointer loads were already rejected by
    // the WithinSlotByteOff bail above.)
    Value *Coerced = coerceToType(V, LoadTy, LI);
    if (!Coerced) {
      markIneligible(*BaseID);
      return;
    }
    auto E = std::make_unique<jeandle::ReplaceLoadEffect>();
    E->Block = LI->getParent();
    E->Target = LI;
    E->Replacement = Coerced;
    E->SeqNo = Result.nextSeqNo();
    E->ObjID = *BaseID;
    Result.addBlockEffect(std::move(E));
    Aliases.addScalarAlias(LI, Coerced);
    return;
  }

  // Should be unreachable; FieldValue::Tag is a closed enum.
  markIneligible(*BaseID);
}

// ---------------------------------------------------------------------------
// JavaOp folding on virtual receivers.
// ---------------------------------------------------------------------------

void Analyzer::emitReplaceCall(CallBase *CB, Value *Replacement,
                               jeandle::ObjectID ID) {
  if (Replacement)
    Replacement = Aliases.resolveScalarAlias(Replacement);
  auto E = std::make_unique<jeandle::ReplaceCallEffect>();
  E->Block = CB->getParent();
  E->Target = CB;
  E->Replacement = Replacement;
  E->SeqNo = Result.nextSeqNo();
  E->ObjID = ID;
  Result.addBlockEffect(std::move(E));
  // Scalar-alias non-void call values so downstream resolveVirtualRef queries
  // (e.g. another JavaOp later in the same block whose operand is the call
  // result) see through to the replacement constant. Void JavaOps can use a
  // null Replacement to request deletion only.
  if (Replacement)
    Aliases.addScalarAlias(CB, Replacement);
}

std::optional<bool> Analyzer::evalSubtypeRelation(uintptr_t SubKlass,
                                                  uintptr_t SuperKlass) {
  const jeandle::VMCallbacks *CB = jeandle::getVMCallbacks();
  if (!CB || !CB->IsSubtype || !CB->IsInterface)
    return std::nullopt;
  if (CB->IsSubtype(SubKlass, SuperKlass))
    return true;
  // Virtual objects always have an exact, concrete klass (we know the
  // allocation site). Pass Exact=true to areKlassesIncompatible.
  if (jeandle::areKlassesIncompatible(SubKlass, /*Exact=*/true, SuperKlass))
    return false;
  return std::nullopt;
}

bool Analyzer::foldArrayLength(CallBase *CB) {
  if (CB->arg_size() < 1)
    return false;
  auto BaseID = jeandle::pea::resolveVirtualRef(CB->getArgOperand(0),
                                                CurrentState, Aliases, DL);
  if (!BaseID)
    return false;
  jeandle::VirtualObject &VObj = *Result.VirtualObjects[*BaseID];
  if (!VObj.isArray())
    return false;
  Type *I32 = Type::getInt32Ty(F.getContext());
  Constant *Len = ConstantInt::get(I32, VObj.ArrayLength);
  emitReplaceCall(CB, Len, *BaseID);
  return true;
}

bool Analyzer::foldLoadKlass(CallBase *CB) {
  if (CB->arg_size() < 1)
    return false;
  auto BaseID = jeandle::pea::resolveVirtualRef(CB->getArgOperand(0),
                                                CurrentState, Aliases, DL);
  if (!BaseID)
    return false;
  jeandle::VirtualObject &VObj = *Result.VirtualObjects[*BaseID];
  if (VObj.Klass == 0)
    return false;
  LLVMContext &Ctx = F.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *PtrTy = PointerType::get(Ctx, jeandle::AddrSpace::CHeapAddrSpace);
  Constant *KlassAsInt = ConstantInt::get(I64, VObj.Klass);
  Constant *KlassPtr = ConstantExpr::getIntToPtr(KlassAsInt, PtrTy);
  emitReplaceCall(CB, KlassPtr, *BaseID);
  return true;
}

bool Analyzer::foldCheckCast(CallBase *CB) {
  if (CB->arg_size() < 2)
    return false;
  uintptr_t SuperK = jeandle::extractKlassConstant(CB->getArgOperand(0));
  if (SuperK == 0)
    return false;
  auto BaseID = jeandle::pea::resolveVirtualRef(CB->getArgOperand(1),
                                                CurrentState, Aliases, DL);
  if (!BaseID)
    return false;
  jeandle::VirtualObject &VObj = *Result.VirtualObjects[*BaseID];
  if (VObj.Klass == 0)
    return false;
  auto Folded = evalSubtypeRelation(VObj.Klass, SuperK);
  if (!Folded)
    return false;
  Constant *Res = *Folded ? ConstantInt::getTrue(CB->getType())
                          : ConstantInt::getFalse(CB->getType());
  emitReplaceCall(CB, Res, *BaseID);
  return true;
}

bool Analyzer::foldInstanceOf(CallBase *CB) {
  if (CB->arg_size() < 2)
    return false;
  uintptr_t SuperK = jeandle::extractKlassConstant(CB->getArgOperand(0));
  if (SuperK == 0)
    return false;
  auto BaseID = jeandle::pea::resolveVirtualRef(CB->getArgOperand(1),
                                                CurrentState, Aliases, DL);
  if (!BaseID)
    return false;
  jeandle::VirtualObject &VObj = *Result.VirtualObjects[*BaseID];
  if (VObj.Klass == 0)
    return false;
  // Virtual objects are non-null by construction, so we can collapse the
  // null-check inside instanceof and reduce to a pure subtype query.
  auto Folded = evalSubtypeRelation(VObj.Klass, SuperK);
  if (!Folded)
    return false;
  Type *I32 = Type::getInt32Ty(F.getContext());
  Constant *Res = ConstantInt::get(I32, *Folded ? 1 : 0);
  emitReplaceCall(CB, Res, *BaseID);
  return true;
}

// FOLD a monitorenter against a virtual receiver (Graal
// MonitorEnterNode.virtualize): record the lock on the VO's state (addLock) and
// emit a ReplaceCall(null) so the transform DELETES the original monitorenter
// call. If the object later materializes, the surviving (unbalanced) enters are
// captured into the Materialize effect's Locks list (captureMaterializedLocks
// via MaterializeContext::CaptureLocksIntoEffect) and RE-EMITTED at the
// materialize point by applyMaterialize (Graal: synthetic MonitorEnterNodes
// attached to the CommitAllocationNode during lowering, sorted ascending by
// lock depth). The matching monitorexit that follows the escape is NOT folded
// (the receiver is already materialized by then) and survives to release the
// re-emitted lock.
//
// See ensureMaterialized's lock-capture block and applyMaterialize's re-emit
// loop for the capture + re-emit mechanism.
uint32_t Analyzer::getOrCreateLockDepth(CallBase *CB) {
  // Cache the FIRST NextLockEnterOrder value for this call site in LockDepthCache
  // and reuse it on every subsequent visit — using NextLockEnterOrder directly is
  // unsound across loop-fixpoint iterations (the counter advances on every
  // re-push, mutating ObjectState::Locks[i].BytecodeDepth and breaking
  // equivalentTo). The cached RPO-order value is sound for every load-bearing
  // use of BytecodeDepth (see LockDepthCache's comment).
  auto FIt = LockDepthCache.find(CB);
  if (FIt != LockDepthCache.end())
    return FIt->second;
  uint32_t D = NextLockEnterOrder;
  LockDepthCache[CB] = D;
  return D;
}

void Analyzer::materializeVirtualLocksBefore(CallBase *MonEnter) {
  // Graal PartialEscapeClosure.materializeVirtualLocksBefore
  // (PartialEscapeClosure.java:641-652), fired from processNodeInternal
  // (PClosure:263-264) on the not-deleted MonitorEnter branch under strict
  // lock order. Before a REAL monitorenter whose bytecode depth is D,
  // materialize every still-virtual VO holding an elided lock with a strictly
  // shallower min depth (LiveLockEnters[id].front() = outermost/min depth,
  // Graal's getMinimumLockDepth). This keeps each such VO's re-emitted lock
  // below this real lock on the lightweight-locking thread lock stack,
  // preserving lexical nesting.
  assert(StrictLockOrder && "caller gates on StrictLockOrder");
  uint32_t LockDepth = getOrCreateLockDepth(MonEnter);
  SmallVector<jeandle::ObjectID, 4> Cascade;
  for (auto &Kv : LiveLockEnters) {
    if (Kv.second.empty())
      continue;
    if (Kv.second.front().BytecodeDepth < LockDepth)
      Cascade.push_back(Kv.first);
  }
  llvm::sort(Cascade); // deterministic
  for (jeandle::ObjectID OID : Cascade)
    materializeAt(OID, MonEnter, MatReason::Cascade);
}

bool Analyzer::foldMonitorEnter(CallBase *CB) {
  if (!LockEliminationEnabled)
    return false;

  if (CB->arg_size() < 1)
    return false;
  auto BaseID = jeandle::pea::resolveVirtualRef(CB->getArgOperand(0),
                                                CurrentState, Aliases, DL);
  if (!BaseID)
    return false;

  // Resolve the bytecode lock depth for this enter (see getOrCreateLockDepth).
  uint32_t NewBytecodeDepth = getOrCreateLockDepth(CB);

  // Lock confinement: the lock counter is balanced per-block at commit
  // time. A monitorenter on a virtual is always safe to provisionally
  // elide; if the matching monitorexit is missing, commit() will flip the
  // virtual to ineligible and the effects will be dropped.
  ++LockCounts[*BaseID];
  // Push the elided enter onto the live stack so materializeAt can undo
  // only the unbalanced enters along this path if the object later escapes.
  // Tag every push with the next monotonic Order; Order is excluded from the
  // loop-fixpoint convergence comparison (Call identity only) because it
  // advances on every re-push.
  // Also record BytecodeDepth (sourced from LockDepthCache, see
  // getOrCreateLockDepth) so the depth-aware cascade and merge-time
  // stack-identity comparisons use a stable, per-call-site ordering key.
  uint32_t MyOrder = NextLockEnterOrder++;
  auto &Stack = LiveLockEnters[*BaseID];
  // Depth monotonicity invariant (mirrors Graal ObjectState.java:212 and the
  // assert in ObjectState::addLock): nested monitorenters acquire strictly
  // increasing bytecode depth, so a newly pushed enter must be strictly
  // deeper than the current innermost (back) live enter on this VO.
  assert(Stack.empty() || NewBytecodeDepth > Stack.back().BytecodeDepth);
  Stack.push_back({CB, MyOrder, NewBytecodeDepth});
  // Keep the per-VO ObjectState::Locks mirror in lockstep with the analyzer-
  // side DenseMap. ObjectState::Locks does not carry the Order proxy —
  // structural ObjectState equivalence (used by merge-time
  // identicalObjectStates and the PEABlockState::equivalentTo path) compares
  // Call+BytecodeDepth only.
  if (CurrentState.hasObjectState(*BaseID)) {
    jeandle::ObjectState &OS =
        CurrentState.getObjectStateForModification(*BaseID);
    if (OS.isVirtual())
      OS.addLock({CB, NewBytecodeDepth});
  }
  // Monitor JavaOps return void (the fast/slow dispatch lives inside the
  // JavaOp body, invisible to PEA), so there is no result to replace: emit a
  // null Replacement and let the transform erase the (always unused) call.
  emitReplaceCall(CB, nullptr, *BaseID);
  return true;
}

bool Analyzer::foldMonitorExit(CallBase *CB) {
  if (!LockEliminationEnabled)
    return false;

  if (CB->arg_size() < 1)
    return false;
  auto BaseID = jeandle::pea::resolveVirtualRef(CB->getArgOperand(0),
                                                CurrentState, Aliases, DL);
  if (!BaseID)
    return false;
  auto It = LockCounts.find(*BaseID);
  if (It == LockCounts.end() || It->second == 0) {
    // Unbalanced monitorexit (release without acquire on this virtual). Mark
    // the virtual ineligible and let the generic escape path keep the call.
    markIneligible(*BaseID);
    return false;
  }
  --It->second;
  // Pop the matching enter off the live stack. The pair is now balanced
  // and both calls' ReplaceCall(true) effects stay in place — the eventual
  // dead-code sweep erases them along with the original allocation.
  auto SIt = LiveLockEnters.find(*BaseID);
  assert(SIt != LiveLockEnters.end() && !SIt->second.empty() &&
         "live stack must be non-empty when LockCount > 0");
  SIt->second.pop_back();
  // Mirror the pop on ObjectState::Locks so the on-VO Locks view
  // stays consistent with the analyzer-side LiveLockEnters for any caller
  // that introspects the ObjectState directly (e.g. equivalentTo /
  // identicalObjectStates).
  if (CurrentState.hasObjectState(*BaseID)) {
    jeandle::ObjectState &OS =
        CurrentState.getObjectStateForModification(*BaseID);
    if (OS.isVirtual() && OS.hasLocks())
      OS.removeLock();
  }
  // Monitor JavaOps return void (the fast/slow dispatch lives inside the
  // JavaOp body, invisible to PEA), so there is no result to replace: emit a
  // null Replacement and let the transform erase the (always unused) call.
  emitReplaceCall(CB, nullptr, *BaseID);
  return true;
}

bool Analyzer::foldArrayStoreCheck(CallBase *CB) {
  // jeandle.array_store_check(value, array). The op is read-only on the heap,
  // so a virtual base is NOT an escape when the check is ELIDED (provably
  // compatible / primitive element): the call is deleted, so no operand
  // reference survives.
  //
  // CONTRACT (mirrors Graal processNodeInputs on a non-deleted node): when the
  // check SURVIVES (cannot be proven elidable) it needs real operands, so BOTH
  // the array and any virtual value must materialize. Such paths return FALSE
  // so the generic escape path (materializeAllVirtualOperands) handles every
  // virtual operand. The only return-true paths are the two elisions below,
  // where the call is deleted and holds no surviving operand reference.
  if (CB->arg_size() < 2)
    return false; // malformed — let the generic path materialize any virtual.
  auto ArrayID = jeandle::pea::resolveVirtualRef(CB->getArgOperand(1),
                                                 CurrentState, Aliases, DL);
  if (!ArrayID)
    return false; // array not virtual — a virtual VALUE operand still escapes.
  jeandle::VirtualObject &ArrayObj = *Result.VirtualObjects[*ArrayID];

  const jeandle::VMCallbacks *VMCB = jeandle::getVMCallbacks();
  if (!VMCB || !VMCB->ArrayElementKlass)
    return false; // cannot prove elidable — survive + materialize.
  if (ArrayObj.Klass == 0)
    return false;

  uintptr_t ElementKlass = VMCB->ArrayElementKlass(ArrayObj.Klass);
  if (ElementKlass == 0) {
    // Primitive (type-array) element: array_store_check is a no-op on
    // primitive arrays (no covariant store check for primitives). Elide to
    // true. A primitive array's value is a primitive, never a virtual object,
    // and the elision deletes the call, so no operand reference survives.
    Constant *True = ConstantInt::getTrue(CB->getType());
    emitReplaceCall(CB, True, *ArrayID);
    return true;
  }

  // Object[] element: compare the value's klass against the element klass.
  Value *Val = CB->getArgOperand(0);
  uintptr_t ValueKlass = 0;
  if (auto ValueID =
          jeandle::pea::resolveVirtualRef(Val, CurrentState, Aliases, DL)) {
    // Virtual values carry an exact, concrete klass.
    ValueKlass = Result.VirtualObjects[*ValueID]->Klass;
  } else {
    // Fall back to attribute / metadata sharpening for non-virtual values.
    jeandle::JavaType JT = jeandle::getJavaType(Val);
    ValueKlass = JT.Klass;
  }

  if (ValueKlass == 0)
    return false; // unknown value klass — cannot prove elidable.

  auto Folded = evalSubtypeRelation(ValueKlass, ElementKlass);
  if (!Folded)
    return false; // indeterminate subtype — cannot prove elidable.

  if (*Folded) {
    // Provably compatible — elide. The value does not escape through the
    // (deleted) check.
    Constant *True = ConstantInt::getTrue(CB->getType());
    emitReplaceCall(CB, True, *ArrayID);
    return true;
  }
  // Provably incompatible: at runtime this throws ArrayStoreException. The
  // surviving check must inspect the real value and array klass — materialize
  // both (return false -> generic escape path).
  return false;
}

bool Analyzer::foldPostBarrier(CallBase *CB) {
  if (CB->arg_size() < 1)
    return false;
  auto BaseID = jeandle::pea::resolveVirtualRef(CB->getArgOperand(0),
                                                CurrentState, Aliases, DL);
  if (!BaseID)
    return false;

  // A post barrier for a store into a virtual object has no concrete card to
  // mark. The store is replayed as initialization when the object is
  // materialized, so the original barrier must not survive with the old slot
  // address.
  emitReplaceCall(CB, nullptr, *BaseID);
  return true;
}

bool Analyzer::foldCheckIfValueBased(CallBase *CB) {
  // jeandle.check_if_value_based(oop) -> i1.  Java emits this around
  // monitorenter on receivers whose static type could be a value-based class
  // (e.g. java.lang.Long), so that the runtime raises a
  // DiagnoseSyncOnValueBasedClasses warning when the dynamic klass is in fact
  // value-based.
  //
  // Fold logic: if the exact runtime class is known, the check collapses to
  // a compile-time constant. For a VIRTUAL receiver:
  //
  //   * VObj.Klass unknown (0)              -> bail (keep the call, virtual
  //                                            will materialize via the
  //                                            generic escape path).
  //   * IsValueBased(VObj.Klass) == true    -> force materialize. The runtime
  //                                            warning MUST observe a real
  //                                            oop; eliding it would change
  //                                            user-visible semantics.
  //   * IsValueBased(VObj.Klass) == false   -> elide the call (constant false).
  //                                            The virtual receiver is
  //                                            provably NOT value-based.
  if (CB->arg_size() < 1)
    return false;
  auto BaseID = jeandle::pea::resolveVirtualRef(CB->getArgOperand(0),
                                                CurrentState, Aliases, DL);
  if (!BaseID)
    return false;
  jeandle::VirtualObject &VObj = *Result.VirtualObjects[*BaseID];
  if (VObj.Klass == 0) {
    // Klass unknown — cannot prove either direction. Fall through to the
    // generic-escape path so the virtual gets materialized and the original
    // runtime check survives on a real oop.
    return false;
  }
  const jeandle::VMCallbacks *VMCB = jeandle::getVMCallbacks();
  if (!VMCB || !VMCB->IsValueBased) {
    // No callback registered (offline tests without a callback log). Bail
    // conservatively so the virtual materializes.
    return false;
  }
  if (VMCB->IsValueBased(VObj.Klass)) {
    // The dynamic klass IS value-based: HotSpot's runtime warning hook for
    // DiagnoseSyncOnValueBasedClasses must observe a real oop. Drop this
    // virtual's eligibility — commit() will discard every recorded effect
    // for it and the original allocation + check call stay in IR, where
    // the call ends up operating on the materialized pointer. Matches the
    // foldArrayStoreCheck "unknown value klass" conservative path.
    markIneligible(*BaseID);
    return true;
  }
  // Provably non-value-based: fold the check to false. The query against
  // an exact (virtualized) klass that does not carry the ValueBased marker
  // canonicalizes to false.
  Constant *False = ConstantInt::getFalse(CB->getType());
  emitReplaceCall(CB, False, *BaseID);
  return true;
}

bool Analyzer::foldRegisterFinalizerIfNeeded(CallBase *CB) {
  // jeandle.register_finalizer_if_needed(oop) -> void.
  //
  // The JavaOp's default lowering preserves HotSpot semantics by checking the
  // receiver klass finalizer bit and calling SharedRuntime_register_finalizer
  // only when needed. Finalizability is resolved at the allocation site:
  // processAllocation (new_instance handling) refuses to virtualize any
  // instance whose exact
  // klass has a finalizer, so such an object stays materialized and this call
  // survives to be lowered normally. A virtual receiver reaching this fold is
  // therefore non-finalizable by construction — delete the provably-no-op
  // call without forcing the object header to materialize.
  if (CB->arg_size() < 1)
    return false;
  auto BaseID = jeandle::pea::resolveVirtualRef(CB->getArgOperand(0),
                                                CurrentState, Aliases, DL);
  if (!BaseID)
    return false;
  jeandle::VirtualObject &VObj = *Result.VirtualObjects[*BaseID];
  if (VObj.Klass == 0)
    return false;
  const jeandle::VMCallbacks *VMCB = jeandle::getVMCallbacks();
  if (!VMCB || !VMCB->HasFinalizer)
    return false;
  // A virtual receiver can only reach here if processAllocation virtualized
  // its allocation, and processAllocation (new_instance handling) refuses to
  // virtualize any instance whose exact klass has a finalizer. So by
  // construction the
  // receiver is non-finalizable: the runtime check would always be false and
  // SharedRuntime_register_finalizer would never fire. Assert that invariant,
  // then delete the provably-no-op call so the allocation can be eliminated.
  assert(!VMCB->HasFinalizer(VObj.Klass) &&
         "processAllocation must refuse finalizable klasses");
  emitReplaceCall(CB, nullptr, *BaseID);
  return true;
}

bool Analyzer::processIntrinsic(IntrinsicInst *II) {
  // Debug-only assert: PEA must run BEFORE RewriteStatepointsForGC. A
  // statepoint intrinsic appearing here means the pass scheduling has been
  // broken; bail loudly in debug, fall through (return false) in release —
  // safe, every statepoint operand is forced to materialize via the
  // hasVirtualInputs handler / generic-escape path.
  assert(II->getIntrinsicID() != Intrinsic::experimental_gc_statepoint &&
         II->getIntrinsicID() != Intrinsic::experimental_gc_relocate &&
         II->getIntrinsicID() != Intrinsic::experimental_gc_result &&
         "PEA must not run after RewriteStatepointsForGC");
  switch (II->getIntrinsicID()) {
  case Intrinsic::assume:
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::invariant_start:
  case Intrinsic::invariant_end:
  case Intrinsic::experimental_noalias_scope_decl:
  case Intrinsic::dbg_declare:
  case Intrinsic::dbg_value:
  case Intrinsic::dbg_label:
  case Intrinsic::donothing:
  case Intrinsic::sideeffect:
  // Extended allowlist. None of these intrinsics produce a pointer with a
  // different identity than their argument, none mutate memory we care about,
  // and none cross the heap/abstract boundary — all safe to leave in IR
  // alongside a virtual without forcing escape.
  // ptr.annotation/var.annotation: TBAA-style debug annotation. The call
  //   returns nothing meaningful and its operand is purely informational.
  // is.constant / expect / expect.with.probability: branch-prediction hints;
  //   their value-result is i1/iN derived from a primitive (the predicate or
  //   the comparison value), not from the virtual pointer's identity, so the
  //   virtual doesn't escape through them.
  // allow.runtime.check / allow.ubsan.check: similar — return i1.
  case Intrinsic::ptr_annotation:
  case Intrinsic::var_annotation:
  case Intrinsic::is_constant:
  case Intrinsic::expect:
  case Intrinsic::expect_with_probability:
  case Intrinsic::allow_runtime_check:
  case Intrinsic::allow_ubsan_check:
    return true;
  // launder/strip invariant.group are pointer-identity-preserving.
  // resolveVirtualRef does not recurse through CallInst, so
  // propagatePointerAlias would fall through to materializeAllVirtualOperands.
  // Directly forward the argument's virtual alias to the result instead.
  case Intrinsic::launder_invariant_group:
  case Intrinsic::strip_invariant_group: {
    Value *Arg = II->getArgOperand(0);
    if (auto BaseID =
            jeandle::pea::resolveVirtualRef(Arg, CurrentState, Aliases, DL))
      Aliases.addVirtualAlias(II, *BaseID);
    // Whether or not the arg resolved, the call has no PEA escape effect.
    return true;
  }
  default:
    return false;
  }
}

bool Analyzer::foldICmpEquality(ICmpInst *ICmp) {
  // Equality compare against a virtual pointer folds. Virtual objects are
  // never null (by construction they track an in-flight alloc), so `icmp eq
  // virt, null` -> false, `icmp ne virt, null` -> true. Two virtuals: same ID
  // -> eq=true; different IDs -> eq=false. Mixed virtual + non-null
  // non-virtual pointer: identity differs -> eq folds to false.
  if (!ICmp->isEquality())
    return false;
  Value *Op0 = ICmp->getOperand(0);
  Value *Op1 = ICmp->getOperand(1);
  auto V0 = jeandle::pea::resolveVirtualRef(Op0, CurrentState, Aliases, DL);
  auto V1 = jeandle::pea::resolveVirtualRef(Op1, CurrentState, Aliases, DL);
  bool Op0IsNull = isa<ConstantPointerNull>(Op0);
  bool Op1IsNull = isa<ConstantPointerNull>(Op1);
  bool Folded = false;
  bool EqResult = false;
  jeandle::ObjectID BaseID = jeandle::InvalidObjectID;
  if (V0 && Op1IsNull) {
    Folded = true;
    EqResult = false;
    BaseID = *V0;
  } else if (V1 && Op0IsNull) {
    Folded = true;
    EqResult = false;
    BaseID = *V1;
  } else if (V0 && V1) {
    if (*V0 != *V1) {
      // Different virtual objects -> distinct identity.
      Folded = true;
      EqResult = false;
      BaseID = *V0;
    } else {
      // Same ObjectID. resolveVirtualRef resolved identity and discarded any
      // derived-pointer byte offset (GEP case chases the base), so two
      // operands of the SAME virtual at DIFFERENT offsets (e.g. %o vs
      // gep(%o,8)) would otherwise conflate to equal. Compare the byte
      // offsets too: equal -> equal addresses, different -> distinct. A
      // symbolic offset can't be proven either way; keep the object real
      // (markIneligible) so a derived-GEP operand stays valid and the icmp
      // survives as a real compare — returning false would hit the gate's
      // materializeAllVirtualOperands(icmp) and poison that operand.
      auto O0 = jeandle::pea::resolveFieldOffset(Op0, DL);
      auto O1 = jeandle::pea::resolveFieldOffset(Op1, DL);
      if (!O0 || !O1) {
        markIneligible(*V0); // *V0 == *V1 (same object).
        return true;
      }
      Folded = true;
      EqResult = (*O0 == *O1);
      BaseID = *V0;
    }
  } else if (V0 && !V1 && !Op1IsNull) {
    // Virtual vs. non-virtual non-null pointer: distinct identity.
    Folded = true;
    EqResult = false;
    BaseID = *V0;
  } else if (V1 && !V0 && !Op0IsNull) {
    Folded = true;
    EqResult = false;
    BaseID = *V1;
  }
  if (!Folded)
    return false;
  bool IsEq = (ICmp->getPredicate() == ICmpInst::ICMP_EQ);
  bool FinalResult = IsEq ? EqResult : !EqResult;
  Constant *C = ConstantInt::get(ICmp->getType(), FinalResult ? 1 : 0);
  // Reuse ReplaceLoad: its handler does generic Instruction RAUW + erase,
  // which is exactly what we need here.
  auto E = std::make_unique<jeandle::ReplaceLoadEffect>();
  E->Block = ICmp->getParent();
  E->Target = ICmp;
  E->Replacement = C;
  E->SeqNo = Result.nextSeqNo();
  E->ObjID = BaseID;
  Result.addBlockEffect(std::move(E));
  Aliases.addScalarAlias(ICmp, C);
  return true;
}

bool Analyzer::processJavaOp(CallBase *CB) {
  using namespace jeandle::pea;
  if (isJeandleArrayLength(CB))
    return foldArrayLength(CB);
  if (isJeandleLoadKlass(CB))
    return foldLoadKlass(CB);
  if (isJeandleCheckCast(CB))
    return foldCheckCast(CB);
  if (isJeandleCheckInstanceOf(CB))
    return foldCheckCast(CB);
  if (isJeandleInstanceOf(CB))
    return foldInstanceOf(CB);
  if (isJeandleMonitorEnter(CB))
    return foldMonitorEnter(CB);
  if (isJeandleMonitorExit(CB))
    return foldMonitorExit(CB);
  if (isJeandleArrayStoreCheck(CB))
    return foldArrayStoreCheck(CB);
  if (isJeandlePostBarrier(CB))
    return foldPostBarrier(CB);
  if (isJeandleCheckIfValueBased(CB))
    return foldCheckIfValueBased(CB);
  if (isJeandleRegisterFinalizerIfNeeded(CB))
    return foldRegisterFinalizerIfNeeded(CB);
  return false;
}

static SmallVector<Value *, 32>
collectDeoptInputs(const OperandBundleUse &Deopt) {
  SmallVector<Value *, 32> Inputs;
  Inputs.reserve(Deopt.Inputs.size());
  for (const Use &U : Deopt.Inputs)
    Inputs.push_back(U.get());
  return Inputs;
}

static bool visitDeoptRecords(
    ArrayRef<Value *> Inputs,
    function_ref<bool(unsigned, const jeandle::DeoptValueEncoding &)> Visit) {
  bool AtScopeStart = true;
  for (unsigned I = 0, E = Inputs.size(); I < E;) {
    if (AtScopeStart) {
      assert(jeandle::startsWithDeoptScopeHeader(Inputs, I) &&
             "deopt bundle must start with a scope header");
      I = jeandle::skipDeoptScopeHeader(Inputs, I);
      AtScopeStart = false;
      continue;
    }

    auto *Encoding = cast<ConstantInt>(Inputs[I]);
    assert(Encoding->getType()->isIntegerTy(64) &&
           "expected i64 deopt value encoding");
    jeandle::DeoptValueEncoding Enc =
        jeandle::DeoptValueEncoding::decode(Encoding->getZExtValue());
    auto VT = Enc.valueType();
    unsigned Next = 0;

    if (VT == jeandle::DeoptValueEncoding::LocalType ||
        VT == jeandle::DeoptValueEncoding::StackType ||
        VT == jeandle::DeoptValueEncoding::NarrowOopMarkerType ||
        VT == jeandle::DeoptValueEncoding::OrigPcSlotType ||
        VT == jeandle::DeoptValueEncoding::MethodType) {
      Next = jeandle::deoptRecordEnd(I, E, 2);
    } else if (VT == jeandle::DeoptValueEncoding::MonitorType) {
      Next = jeandle::deoptRecordEnd(I, E, 3);
    } else if (VT == jeandle::DeoptValueEncoding::LazyObjectType) {
      bool IsArray = Enc.basicType() == jeandle::T_ARRAY;
      unsigned CountIndex = I + (IsArray ? 3 : 2);
      assert(CountIndex < E && "truncated lazy-object deopt record");
      auto *CountC = dyn_cast<ConstantInt>(Inputs[CountIndex]);
      assert(CountC && "lazy-object field count must be constant");
      unsigned Count = static_cast<unsigned>(CountC->getZExtValue());
      unsigned HeaderSize = IsArray ? 4 : 3;
      Next = jeandle::deoptRecordEnd(I, E, HeaderSize + Count * 3);
    } else {
      llvm_unreachable("unknown deopt value type");
    }

    if (Visit(I, Enc))
      return true;

    I = Next;
    if (VT == jeandle::DeoptValueEncoding::MethodType)
      AtScopeStart = true;
  }
  return false;
}

static std::optional<unsigned>
deoptObjectOperandIndex(unsigned Begin, const jeandle::DeoptValueEncoding &Enc) {
  if (!jeandle::isObjectDeoptEncoding(Enc))
    return std::nullopt;

  auto VT = Enc.valueType();
  if (VT == jeandle::DeoptValueEncoding::LocalType ||
      VT == jeandle::DeoptValueEncoding::StackType ||
      VT == jeandle::DeoptValueEncoding::MonitorType)
    return Begin + 1;

  return std::nullopt;
}

void Analyzer::recordDeoptBundleState(CallBase *CB) {
  std::optional<OperandBundleUse> Deopt = CB->getOperandBundle("deopt");
  if (!Deopt)
    return;

  SmallVector<Value *, 32> Inputs = collectDeoptInputs(*Deopt);

  SmallVector<jeandle::RecordDeoptStateEffect::ObjectSnapshot, 4> Objects;
  DenseMap<jeandle::ObjectID, unsigned> ObjectIndex;
  DenseMap<jeandle::ObjectID, SmallVector<Value *, 2>> ObjectValues;

  enum class CaptureState : uint8_t { Capturing, Captured, Failed };
  DenseMap<jeandle::ObjectID, CaptureState> CaptureStates;

  auto AddValue = [&](jeandle::ObjectID ID, Value *ObjVal) {
    if (!ObjVal)
      return;
    auto Existing = ObjectIndex.find(ID);
    if (Existing != ObjectIndex.end()) {
      auto &Values = Objects[Existing->second].Values;
      if (!llvm::is_contained(Values, ObjVal))
        Values.push_back(ObjVal);
      return;
    }

    auto &Values = ObjectValues[ID];
    if (!llvm::is_contained(Values, ObjVal))
      Values.push_back(ObjVal);
  };

  auto DominatesDeoptPoint = [&](Value *V) {
    if (auto *FieldI = dyn_cast_or_null<Instruction>(V))
      if (FieldI->getParent() && !DT.dominates(FieldI, CB))
        return false;
    return true;
  };

  std::function<bool(jeandle::ObjectID)> CaptureID =
      [&](jeandle::ObjectID ID) -> bool {
    auto StateIt = CaptureStates.find(ID);
    if (StateIt != CaptureStates.end())
      return StateIt->second != CaptureState::Failed;

    if (ID == jeandle::InvalidObjectID || ID >= Result.VirtualObjects.size() ||
        !Result.VirtualObjects[ID] || !Eligible.lookup(ID)) {
      CaptureStates[ID] = CaptureState::Failed;
      return false;
    }

    jeandle::VirtualObject &VObj = *Result.VirtualObjects[ID];
    // Object-array field states use narrow oops when compressed oops are
    // enabled. Keep them conservative until PEA models that storage form
    // end-to-end; primitive arrays are independent and can be reconstructed
    // by the element-count-aware lazy-object protocol.
    if (VObj.isArray() && VObj.ArrayElementType &&
        VObj.ArrayElementType->isPointerTy()) {
      markIneligible(ID);
      CaptureStates[ID] = CaptureState::Failed;
      return false;
    }
    if ((!VObj.isInstance() && !VObj.isArray()) || VObj.Klass == 0) {
      CaptureStates[ID] = CaptureState::Failed;
      return false;
    }

    CaptureStates[ID] = CaptureState::Capturing;

    jeandle::RecordDeoptStateEffect::ObjectSnapshot Snapshot;
    Snapshot.ID = ID;
    auto ValuesIt = ObjectValues.find(ID);
    if (ValuesIt != ObjectValues.end())
      for (Value *V : ValuesIt->second)
        Snapshot.Values.emplace_back(V);

    auto FSIt = FieldStates.find(ID);
    if (FSIt != FieldStates.end()) {
      SmallVector<int64_t, 8> Offsets;
      Offsets.reserve(FSIt->second.size());
      for (const auto &Kv : FSIt->second)
        Offsets.push_back(Kv.first);
      llvm::sort(Offsets);

      for (int64_t Off : Offsets) {
        const jeandle::FieldValue &FV = FSIt->second.lookup(Off);
        if (FV.isUnknown())
          continue;

        if (VObj.isArray() && VObj.ArrayIndexScale != 0) {
          int64_t Adj = Off - static_cast<int64_t>(VObj.ArrayBaseOffset);
          int64_t Scale = static_cast<int64_t>(VObj.ArrayIndexScale);
          if (Adj < 0 || Adj % Scale != 0 ||
              static_cast<uint64_t>(Adj / Scale) >= VObj.ArrayLength) {
            markIneligible(ID);
            CaptureStates[ID] = CaptureState::Failed;
            return false;
          }

          bool ExactElement = false;
          for (const jeandle::VirtualObject::FieldDesc &F : VObj.Fields) {
            if (F.Offset != Off)
              continue;
            ExactElement =
                F.ByteSize == VObj.ArrayIndexScale && !F.IsReference;
            break;
          }
          if (!ExactElement) {
            markIneligible(ID);
            CaptureStates[ID] = CaptureState::Failed;
            return false;
          }
        }

        if (FV.isScalar()) {
          if (!DominatesDeoptPoint(FV.getScalar())) {
            CaptureStates[ID] = CaptureState::Failed;
            return false;
          }
        } else if (FV.isMaterializedRef()) {
          if (!DominatesDeoptPoint(FV.getMaterialized())) {
            CaptureStates[ID] = CaptureState::Failed;
            return false;
          }
        } else if (FV.isVirtualRef()) {
          if (!CaptureID(FV.getVirtualRef())) {
            CaptureStates[ID] = CaptureState::Failed;
            return false;
          }
        }

        Snapshot.FieldEntries.push_back({Off, FV});
      }
    }

    ObjectIndex[ID] = Objects.size();
    Objects.push_back(std::move(Snapshot));
    CaptureStates[ID] = CaptureState::Captured;
    return true;
  };

  auto Capture = [&](Value *ObjVal) {
    if (!ObjVal || !ObjVal->getType()->isPointerTy())
      return;
    std::optional<jeandle::ObjectID> MaybeID =
        jeandle::pea::resolveVirtualRef(ObjVal, CurrentState, Aliases, DL);
    if (!MaybeID)
      return;

    jeandle::ObjectID ID = *MaybeID;
    if (!Eligible.lookup(ID))
      return;

    AddValue(ID, ObjVal);
    if (!CaptureID(ID))
      materializeAt(ID, CB, MatReason::Unhandled);
  };

  visitDeoptRecords(Inputs, [&](unsigned Begin,
                                const jeandle::DeoptValueEncoding &Enc) {
    if (std::optional<unsigned> ObjIdx = deoptObjectOperandIndex(Begin, Enc))
      Capture(Inputs[*ObjIdx]);
    return false;
  });

  for (auto &Snapshot : Objects) {
    auto E = std::make_unique<jeandle::RecordDeoptStateEffect>();
    E->Block = CB->getParent();
    E->Target = CB;
    E->SeqNo = Result.nextSeqNo();
    E->ObjID = Snapshot.ID;
    E->Snapshot = std::move(Snapshot);
    Result.addBlockEffect(std::move(E));
  }
}

void Analyzer::propagatePointerAlias(Instruction *I) {
  // The instruction is a pointer-identity-preserving transformation whose
  // operand carries a virtual alias. Forward the alias to the result.
  if (Aliases.getVirtualAlias(I))
    return;

  // SelectInst: resolveVirtualRefImpl returns the common ObjectID of the two
  // arms and DISCARDS the per-arm byte offset (resolveFieldOffset has no
  // Select case, so it returns 0 for the Select itself). A `select %c,
  // gep(%v,16), gep(%v,16)` would otherwise alias-forward to %v and a
  // downstream load/store through the select be modelled at offset 0 instead
  // of 16. So alias-forward only when both arms denote the WHOLE object
  // (resolveFieldOffset 0 each — a poison arm also resolves to 0 and cannot
  // execute without UB). Any other shape (non-zero/symbolic offset, or
  // different objects) is handed to materializeVirtualOperandsSafely: it
  // materializes whole-object arms but calls markIneligible on any object whose
  // derived-GEP arm would poison if materialized at the select (same class as
  // the processStore/processLoad bail — see materializeVirtualOperandsSafely).
  if (auto *Sel = dyn_cast<SelectInst>(I)) {
    auto BaseID =
        jeandle::pea::resolveVirtualRef(Sel, CurrentState, Aliases, DL);
    auto TOff = jeandle::pea::resolveFieldOffset(Sel->getTrueValue(), DL);
    auto FOff = jeandle::pea::resolveFieldOffset(Sel->getFalseValue(), DL);
    if (BaseID && TOff && FOff && *TOff == 0 && *FOff == 0) {
      Aliases.addVirtualAlias(I, *BaseID);
      return;
    }
    materializeVirtualOperandsSafely(I);
    return;
  }

  auto BaseID = jeandle::pea::resolveVirtualRef(I, CurrentState, Aliases, DL);
  if (!BaseID) {
    // Couldn't resolve — the underlying chain may have already escaped.
    materializeAllVirtualOperands(I);
    return;
  }
  Aliases.addVirtualAlias(I, *BaseID);
}

static bool isDeoptBundleOperand(Instruction *I, const Use &U) {
  auto *CB = dyn_cast<CallBase>(I);
  if (!CB || !CB->hasOperandBundles() || !CB->isBundleOperand(&U))
    return false;
  unsigned OpIdx = static_cast<unsigned>(&U - CB->op_begin());
  return CB->getOperandBundleForOperand(OpIdx).getTagName() == "deopt";
}

static bool isModeledDeoptObjectOperand(Instruction *I, const Use &U) {
  auto *CB = dyn_cast<CallBase>(I);
  if (!CB || !isDeoptBundleOperand(I, U))
    return false;
  std::optional<OperandBundleUse> Deopt = CB->getOperandBundle("deopt");
  if (!Deopt)
    return false;

  SmallVector<Value *, 32> Inputs = collectDeoptInputs(*Deopt);
  return visitDeoptRecords(
      Inputs, [&](unsigned Begin, const jeandle::DeoptValueEncoding &Enc) {
        if (std::optional<unsigned> ObjIdx =
                deoptObjectOperandIndex(Begin, Enc))
          return &Deopt->Inputs[*ObjIdx] == &U;
        return false;
      });
}

void Analyzer::materializeAllVirtualOperands(Instruction *I) {
  // Trigger materialization for every distinct virtual ObjectID that I uses.
  // After materializeAt, the per-object state in CurrentState flips to
  // Materialized so subsequent resolveVirtualRef queries return nullopt.
  // Operand rewriting is done by the point-sensitive resolution sub-pass: each
  // materialize records an OrigAlloc -> NewInv definition in Defs, and the
  // sub-pass rewrites every surviving OrigAlloc use to the unique dominating
  // def — so I's operand here resolves to NewInv (uses do not auto-update
  // globally).
  SmallVector<jeandle::ObjectID, 4> ToMaterialize;
  DenseSet<jeandle::ObjectID> Seen;
  for (Use &U : I->operands()) {
    if (isModeledDeoptObjectOperand(I, U))
      continue;
    Value *V = U.get();
    if (!V)
      continue;
    if (auto MaybeID =
            jeandle::pea::resolveVirtualRef(V, CurrentState, Aliases, DL)) {
      if (Seen.insert(*MaybeID).second)
        ToMaterialize.push_back(*MaybeID);
    }
  }
  llvm::sort(ToMaterialize);
  for (jeandle::ObjectID ID : ToMaterialize)
    materializeAt(ID, I, MatReason::Unhandled);
}

void Analyzer::materializeVirtualOperandsSafely(Instruction *I) {
  // Materializing at I is sound only when every virtual operand is the object's
  // ORIGINAL allocation (AllocationCall) — a direct use at I that pea.mat at I
  // dominates, so the resolution sub-pass rewrites it. A DERIVED operand (any
  // value != AllocationCall that resolveVirtualRef chases to a virtual: GEP,
  // bitcast, addrspacecast, freeze, inttoptr-roundtrip, PHI, select, or an
  // alias-map entry) is computed before I, so its OrigAlloc use sits at its own
  // def, which pea.mat at I cannot dominate — resolveMaterializedUses leaves it
  // for EliminateAllocation's poison RAUW, turning the operand into
  // `getelementptr ... poison`. When any operand is derived, keep every virtual
  // operand's object real via markIneligible (derived pointers then stay
  // valid); otherwise materialize as usual. The derived test is structural (V
  // != AllocationCall), NOT offset-based: a zero-offset `gep %o, 0` is still a
  // derived instruction and would still poison.
  for (Use &U : I->operands()) {
    Value *V = U.get();
    if (!V)
      continue;
    auto ID = jeandle::pea::resolveVirtualRef(V, CurrentState, Aliases, DL);
    if (!ID)
      continue;
    if (V != Result.VirtualObjects[*ID]->AllocationCall) {
      markVirtualOperandsIneligible(I);
      return;
    }
  }
  materializeAllVirtualOperands(I);
}

void Analyzer::markVirtualOperandsIneligible(Instruction *I) {
  // Bail primitive: mark every distinct virtual ObjectID among I's operands
  // ineligible so each keeps its ORIGINAL allocation real (AlwaysEscapes via
  // commit() -> dropEffectsFor dropping EliminateAllocationEffect). Mirrors
  // materializeAllVirtualOperands enumeration but markIneligible instead of
  // materializeAt. Used wherever materializing at I would poison a derived
  // operand (processStore/processLoad bail, materializeVirtualOperandsSafely).
  SmallVector<jeandle::ObjectID, 4> ToBail;
  DenseSet<jeandle::ObjectID> Seen;
  for (Use &U : I->operands()) {
    Value *V = U.get();
    if (!V)
      continue;
    if (auto MaybeID =
            jeandle::pea::resolveVirtualRef(V, CurrentState, Aliases, DL)) {
      if (Seen.insert(*MaybeID).second)
        ToBail.push_back(*MaybeID);
    }
  }
  llvm::sort(ToBail);
  // TODO(bail-all-conservative): GRAAL DIVERGENCE — when ANY operand is
  // derived, this bails EVERY virtual operand's object (including operands
  // that are themselves the OrigAlloc, i.e. V == AllocationCall, for which
  // materializing at I WOULD be sound: pea.mat@I dominates that OrigAlloc use
  // at I). Graal's processNodeInputs (PartialEscapeClosure.java:433-451) has no
  // derived pointers and always materializes per-input at the use, so it
  // keeps such objects virtual on non-escaping paths. The bail-all is a
  // deliberate soundness net: materializing one operand's object at I could
  // poison a derived GEP of the SAME object on another path that the I-point
  // materialize does not dominate (the e4f0a55f class of traps). Recovering
  // the lost scalar-replacement here means proving, per-object, that none of
  // its live uses across all paths is an undominated derived address — bail
  // only the derived operand's object, materialize the OrigAlloc operands at
  // I. Until that cross-path dominance check exists, keep bail-all.
  for (jeandle::ObjectID ID : ToBail)
    markIneligible(ID);
}

// Live-path materialization uses the escaping call or invoke by default. The
// allocation-sinking option additionally permits the nearest safe prior deopt
// point in the same block. Per-predecessor materialization stays at the
// predecessor terminator. commit() rejects every effect whose actual insertion
// point cannot provide deopt state.

// Returns true iff CB has at least one "deopt" operand bundle.
static bool hasDeoptBundle(CallBase *CB) {
  for (unsigned i = 0, n = CB->getNumOperandBundles(); i < n; ++i)
    if (CB->getOperandBundleAt(i).getTagName() == "deopt")
      return true;
  return false;
}

static CallBase *getCallOrInvokeWithDeoptState(Instruction *I) {
  auto *CB = dyn_cast_or_null<CallBase>(I);
  if (!CB || (!isa<CallInst>(CB) && !isa<InvokeInst>(CB)) ||
      !hasDeoptBundle(CB))
    return nullptr;
  return CB;
}

bool Analyzer::instructionUsesAnyVirtualObject(Instruction *I) {
  for (Use &U : I->operands()) {
    Value *V = U.get();
    if (V && jeandle::pea::resolveVirtualRef(V, CurrentState, Aliases, DL))
      return true;
  }
  return false;
}

bool Analyzer::canHoistMaterializeToDeoptAnchor(Instruction *Anchor,
                                                Instruction *EscapePoint) {
  if (!Anchor || !EscapePoint || Anchor->getParent() != EscapePoint->getParent())
    return false;

  // We reuse the current (escape-point) FieldStates snapshot. Any intervening
  // virtual-object activity could make that snapshot newer than the anchor.
  for (Instruction *I = Anchor->getNextNode(); I && I != EscapePoint;
       I = I->getNextNode()) {
    if (instructionUsesAnyVirtualObject(I))
      return false;
  }
  return true;
}

std::optional<Analyzer::DeoptMaterializeAnchor>
Analyzer::findDeoptMaterializeAnchor(Instruction *EscapePoint) {
  if (!EscapePoint)
    return std::nullopt;

  if (CallBase *CB = getCallOrInvokeWithDeoptState(EscapePoint))
    return DeoptMaterializeAnchor{EscapePoint, CB};

  if (!JeandlePEAEnableAggressiveAllocationSinking)
    return std::nullopt;

  // Same-block only: crossing CFG edges would require a path-sensitive virtual
  // state snapshot at the anchor and replay/validation of all intervening
  // virtual updates.
  for (Instruction *I = EscapePoint->getPrevNode(); I; I = I->getPrevNode()) {
    CallBase *CB = getCallOrInvokeWithDeoptState(I);
    if (!CB)
      continue;
    if (!canHoistMaterializeToDeoptAnchor(I, EscapePoint))
      return std::nullopt;
    return DeoptMaterializeAnchor{I, CB};
  }

  return std::nullopt;
}

// Sweep sibling VOs' FieldStates after a materialisation. Runs for EVERY
// materialisation (outer OR recursive).
void Analyzer::updateOtherStatesForMaterialized(
    jeandle::ObjectID FlippedID, Value *NewPtr,
    DenseMap<jeandle::ObjectID, DenseMap<int64_t, jeandle::FieldValue>> &FS) {
  for (auto &OtherKv : FS) {
    if (OtherKv.first == FlippedID)
      continue;
    for (auto &Entry : OtherKv.second) {
      if (Entry.second.isVirtualRef() &&
          Entry.second.getVirtualRef() == FlippedID) {
        Entry.second = jeandle::FieldValue::materializedRef(NewPtr);
      }
    }
  }
}

void Analyzer::bumpMaterializeStat(MatReason R) {
  ++JeandlePEAMaterialized;
  switch (R) {
  case MatReason::Unhandled:
    ++JeandlePEAMaterializedUnhandled;
    break;
  case MatReason::Cascade:
    // Cascade and nested still count toward the Unhandled
    // rollup (they are byproducts of an upstream Unhandled escape) but
    // are also bookkept in their own counter for fine-grained audits.
    ++JeandlePEAMaterializedUnhandled;
    ++JeandlePEAMaterializedCascade;
    break;
  case MatReason::Nested:
    ++JeandlePEAMaterializedUnhandled;
    ++JeandlePEAMaterializedNested;
    break;
  case MatReason::Merge:
    ++JeandlePEAMaterializedMerge;
    break;
  case MatReason::LoopExit:
    ++JeandlePEAMaterializedLoopExit;
    break;
  case MatReason::Phi:
    ++JeandlePEAMaterializedPHI;
    break;
  }
}

// Capture the surviving unbalanced monitorenters into a Materialize effect's
// Locks list, for re-emit at the materialize point (Graal analog: synthetic
// MonitorEnterNodes attached to the CommitAllocationNode). Each entry is
// self-contained — Callee + non-receiver args + bytecode depth — because the
// new lock model deletes the original enter from IR, so the transform cannot
// read it later. Sorted ascending by bytecode depth to match Graal's
// lock-depth-ordered re-emit (DefaultJavaLoweringProvider
// finishAllocatedObjects).
static void captureMaterializedLocks(ArrayRef<LockEnter> Stack,
                                     jeandle::MaterializeEffect &E) {
  for (const LockEnter &LE : Stack) {
    CallBase *Enter = LE.Call;
    if (!Enter)
      continue;
    Function *Callee = Enter->getCalledFunction();
    if (!Callee)
      continue; // indirect monitor enter — defensive; should not happen.
    jeandle::MaterializedLock ML;
    ML.Callee = Callee;
    for (unsigned i = 1, e = Enter->arg_size(); i < e; ++i)
      ML.NonReceiverArgs.push_back(Enter->getArgOperand(i));
    ML.BytecodeDepth = LE.BytecodeDepth;
    E.Locks.push_back(std::move(ML));
  }
  llvm::sort(E.Locks, [](const jeandle::MaterializedLock &A,
                         const jeandle::MaterializedLock &B) {
    return A.BytecodeDepth < B.BytecodeDepth;
  });
}

// Graal PartialEscapeClosure.ensureMaterialized -> materializeBefore ->
// materializeWithCommit (PartialEscapeBlockState.java:208-343): the single
// materialization algorithm shared by the escape-point path (materializeAt) and
// the merge-driven per-predecessor path (materializeAtPredFromExitInfo). The
// per-path differences are supplied via MaterializeContext C (operative state
// maps, idempotency set, recursion target, lock-capture phase-2 timing, alias
// drop, safe IP, effect flags, state flip), keeping this body path-agnostic.
void Analyzer::ensureMaterialized(jeandle::ObjectID ID, MaterializeContext &C) {
  if (C.MaterializedSet.count(ID))
    return; // idempotent — first escape wins; also breaks nested-cycles.
  if (!Eligible.lookup(ID))
    return; // already gave up on this object; nothing to materialize.
  // Dead-block guard: materializing in a statically-dead block would commit
  // phantom IR the next canonicalisation pass must clean up.
  if (CurrentState.isDead())
    return;

  jeandle::VirtualObject &VObj = *Result.VirtualObjects[ID];

  // Overflow detection (Graal-aligned). Graal's ensureMaterialized
  // (PartialEscapeClosure.java:541-562) throws EffecsClosureOverflowException
  // when currentMode == STOP_NEW_VIRTUALIZATIONS_LOOP_NEST and a virtual
  // object is about to be materialized. In STOP_NEW no new virtualizations
  // occur (processAllocation refuses them), so every virtual object still in
  // scope was created OUTSIDE the active loop nest (pre-loop / outer loop).
  // Materializing such an outer-scope object here would force re-iteration of
  // the whole nest (the materialization must propagate back to where the
  // object was allocated), which is exponential in nest depth. So we bail:
  // poll OverflowFlag (Jeandle is -fno-exceptions; a polled flag is the
  // equivalent of the throw) and skip this materialization. The outermost
  // (depth==1) processLoop catches the flag, restores the snapshot, drains the
  // preheader via processStateBeforeLoopOnOverflow, and redoes the nest in
  // MATERIALIZE_ALL — exactly Graal's catch block
  // (EffectsClosure.java:533-551). No dominance check is needed: in STOP_NEW
  // *any* virtual object reaching materialization is outer-scope by
  // construction. The ensureVirtualized -> RetryableBailoutException branch
  // (PClosure:542-551) is intentionally deferred (deopt-adjacent); see TODO in
  // processAllocation.
  if (CurrentMode == Mode::StopNewInLoopNest) {
    OverflowFlag = true;
    return;
  }

  // PHI Case-C synthetic VOs cannot be materialized (no per-pred allocation to
  // RAUW from). Conservatively drop the synthetic and every per-pred source to
  // ineligible so the original allocations and stores survive.
  //
  // GRAAL DIVERGENCE: Graal materializes a synthetic Case-C VO by materializing
  // each per-pred source VO (mergeObjectEntry / the processPhi fallback,
  // PartialEscapeClosure.java:1340-1365 & 1493-1514) and reusing the existing
  // materializedValuePhi as the materialized value. Jeandle bails instead.
  // TODO(cascade-materialize): implement (a) per-pred-source materialization
  // that threads each pred's NewInv through the Case-C PHI and reuses the
  // existing materializedValuePhi as the materialized pointer, plus (b) the
  // materialize-placement + lock-model alignment noted at materializeAt /
  // foldMonitorEnter (synthetics' borrowed AllocationCall has no dominating
  // alloc point).
  if (VObj.IsSynthetic) {
    markIneligible(ID); // cascades transitively over nested synthetics.
    C.MaterializedSet.insert(ID); // idempotent guard for re-entries.
    return;
  }

  // Cycle prevention: insert BEFORE recursing on any nested VirtualRef
  // ("flip the state then recurse") so a self-referential / cyclic field graph
  // (A.f = B, B.g = A) terminates. The strict-lock cascade below relies on this
  // too.
  C.MaterializedSet.insert(ID);

  // Strict-lock-order cascade (Graal materializeWithCommit:323-333): when this
  // VO has live locks and the runtime requires strict nesting, materialize
  // every other still-locked virtual whose OUTERMOST live lock was acquired
  // strictly before this VO's INNERMOST live lock. LiveLockEnters[id].front()
  // is the min-depth (outermost) lock, .back() is the max-depth (innermost).
  //
  // ORDERING DIVERGENCE from Graal: Jeandle runs this lock cascade BEFORE the
  // recursive entry-prerequisite cascade (below); Graal's materializeWithCommit
  // runs the entry cascade first (PartialEscapeBlockState.java:309-321) then
  // the lock cascade (323-333). This is behavior-preserving in Jeandle: each
  // materialize is an independent Effect with its own SeqNo / SafeIP /
  // dominance check, the two cascades share no writable state (the lock cascade
  // reads LiveLockEnters, the entry cascade reads FieldStates), and the final
  // state is made order-independent by updateOtherStatesForMaterialized.
  // Graal's order matters only because its materializeWithCommit mutates one
  // shared ObjectState in place.
  auto LCIt = C.LockCounts.find(ID);
  bool HasLiveLocks = (LCIt != C.LockCounts.end() && LCIt->second != 0);
  if (HasLiveLocks && StrictLockOrder) {
    auto ThisIt = C.LiveLockEnters.find(ID);
    assert(ThisIt != C.LiveLockEnters.end() && !ThisIt->second.empty() &&
           "LockCount > 0 implies a non-empty LiveLockEnters stack");
    uint32_t ThisMaxDepth = ThisIt->second.back().BytecodeDepth;
    SmallVector<jeandle::ObjectID, 4> Cascade;
    for (auto &Kv : C.LiveLockEnters) {
      if (Kv.first == ID || Kv.second.empty())
        continue;
      if (C.MaterializedSet.count(Kv.first))
        continue;
      if (Kv.second.front().BytecodeDepth < ThisMaxDepth)
        Cascade.push_back(Kv.first);
    }
    llvm::sort(Cascade); // deterministic order
    for (jeandle::ObjectID OtherID : Cascade)
      C.Recurse(OtherID, MatReason::Cascade);
  }

  // Lock model (Graal-aligned): the surviving unbalanced monitorenters on the
  // live stack are captured into the Materialize effect's Locks list for
  // re-emit at the materialize point (captureMaterializedLocks, invoked in the
  // effect- build section below), and the operative lock state is cleared so
  // commit()'s LockCounts!=0 gate doesn't disqualify. The original enter calls
  // stay DELETED (their ReplaceCall(null) effects survive); applyMaterialize
  // emits fresh enters at the materialize point (Graal analog: synthetic
  // MonitorEnterNodes at the CommitAllocationNode during lowering).
  SmallVector<LockEnter, 4> LocksToReEmit;
  auto SIt = C.LiveLockEnters.find(ID);
  if (HasLiveLocks && SIt != C.LiveLockEnters.end() && !SIt->second.empty()) {
    LocksToReEmit.assign(SIt->second.begin(), SIt->second.end());
    C.ClearLockState(ID);
  }

  // Recursive prerequisite materialization: for each field holding a VirtualRef
  // to an inner virtual, materialize the inner first, then rewrite the outer's
  // FieldStates entry to a MaterializedRef at the inner's OrigAlloc. The
  // field-replay value is OrigAlloc on both the live and per-pred paths (both
  // MaterializedValue callbacks return OrigAlloc); the transform's point-
  // sensitive resolution sub-pass (resolveMaterializedUses) rewrites each
  // field-store use to the NewInv that dominates it, recovering per-pred
  // distinctness via dominance — Jeandle's analog of Graal getAliasAndResolve
  // (PartialEscapeClosure.java:1575).
  {
    auto FSIt = C.FieldStates.find(ID);
    if (FSIt != C.FieldStates.end()) {
      SmallVector<int64_t, 4> NestedOffsets;
      for (auto &Kv : FSIt->second)
        if (Kv.second.isVirtualRef())
          NestedOffsets.push_back(Kv.first);
      llvm::sort(NestedOffsets); // determinism
      for (int64_t Off : NestedOffsets) {
        // Re-lookup each iteration; the recursion may have mutated
        // FieldStates (updateStatesForMaterialized below) and invalidated
        // iterators.
        auto It2 = C.FieldStates.find(ID);
        if (It2 == C.FieldStates.end())
          break;
        auto OffIt = It2->second.find(Off);
        if (OffIt == It2->second.end() || !OffIt->second.isVirtualRef())
          continue;
        jeandle::ObjectID InnerID = OffIt->second.getVirtualRef();
        C.Recurse(InnerID, MatReason::Nested);
        if (!Eligible.lookup(InnerID)) {
          // Inner gave up — the outer cannot be materialized either.
          Eligible[ID] = false;
          return;
        }
        // Record the inner's OrigAlloc for field-replay.
        // resolveMaterializedUses resolves it to the dominating NewInv, giving
        // each per-pred materialize its own inner-NewInv.
        Value *InnerVal = C.MaterializedValue(InnerID);
        C.FieldStates[ID][Off] = jeandle::FieldValue::materializedRef(InnerVal);
        // updateStatesForMaterialized: every other still-tracked object whose
        // FieldStates references InnerID must also flip to MaterializedRef.
        updateOtherStatesForMaterialized(InnerID, InnerVal, C.FieldStates);
        // Drop alias-map entries resolving to InnerID (live path only).
        C.DropInnerAliases(InnerID);
      }
    }
  }

  // Safe materialization insertion point (path-specific).
  Instruction *SafeIP = C.ComputeSafeIP();
  assert(SafeIP && "materialization requires a safe insertion point");

  // Per-field dominance check: after VirtualRef rewriting, every Scalar /
  // MaterializedRef field value must dominate SafeIP. Field values are
  // OrigAlloc (or a scalar); OrigAlloc dominates SafeIP by SSA on both paths
  // (the allocation precedes the materialize point).
  auto FSIt = C.FieldStates.find(ID);
  if (FSIt != C.FieldStates.end()) {
    for (auto &Kv : FSIt->second) {
      const jeandle::FieldValue &FV = Kv.second;
      Value *V = nullptr;
      if (FV.isScalar())
        V = FV.getScalar();
      else if (FV.isMaterializedRef())
        V = FV.getMaterialized();
      else
        continue;
      if (!V)
        continue;
      if (auto *VI = dyn_cast<Instruction>(V)) {
        if (!DT.dominates(VI, SafeIP)) {
          Eligible[ID] = false;
          return;
        }
      }
    }
  }

  // Build the Materialize effect (common fields + the per-offset field
  // snapshot), let the context add the path-specific flags, then commit.
  auto E = std::make_unique<jeandle::MaterializeEffect>();
  E->Block = SafeIP->getParent();
  E->SeqNo = Result.nextSeqNo();
  E->InsertBefore = SafeIP;
  E->Target = VObj.AllocationCall;
  E->ObjID = ID;
  if (FSIt != C.FieldStates.end()) {
    SmallVector<int64_t, 8> Offsets;
    Offsets.reserve(FSIt->second.size());
    for (auto &Kv : FSIt->second)
      Offsets.push_back(Kv.first);
    llvm::sort(Offsets); // deterministic snapshot order
    for (int64_t Off : Offsets) {
      const jeandle::FieldValue &FV = FSIt->second.lookup(Off);
      if (FV.isUnknown())
        continue;
      E->FieldEntries.push_back({Off, FV});
    }
  }
  // Capture the surviving unbalanced enters (sorted ascending by depth) into
  // E->Locks for re-emit at the materialize point by applyMaterialize.
  C.CaptureLocksIntoEffect(LocksToReEmit, ID, *E);
  C.SetEffectFlags(*E, SafeIP);
  Result.addBlockEffect(std::move(E));
  bumpMaterializeStat(C.Reason);

  // Flip the per-object state to materialized on this path.
  C.FlipState(ID);

  // Sweep sibling VOs whose FieldStates still hold a VirtualRef to this just-
  // materialised object, so a later store/load through a sibling field observes
  // the materialized pointer.
  updateOtherStatesForMaterialized(ID, C.MaterializedValue(ID), C.FieldStates);
}

void Analyzer::materializeAt(jeandle::ObjectID ID, Instruction *InsertBefore,
                             MatReason Reason) {
  // Live-state path. Delegates the shared cascade / lock-capture /
  // recursive-prereq / dominance / emit / flip algorithm to
  // ensureMaterialized (Graal ensureMaterialized -> materializeBefore ->
  // materializeWithCommit). This wrapper supplies the live analyzer maps, the
  // function-wide Materialized idempotency set, recursion back into
  // materializeAt, and the live-path-specific anchor selection.
  if (Materialized.count(ID) || !Eligible.lookup(ID) || CurrentState.isDead())
    return;

  std::optional<DeoptMaterializeAnchor> Anchor =
      findDeoptMaterializeAnchor(InsertBefore);

  auto ClearLockState = [&](jeandle::ObjectID Oid) {
    LockCounts[Oid] = 0;
    LiveLockEnters.erase(Oid);
    if (CurrentState.hasObjectState(Oid)) {
      jeandle::ObjectState &OS =
          CurrentState.getObjectStateForModification(Oid);
      if (OS.isVirtual())
        OS.clearLocks();
    }
  };
  auto DropInnerAliases = [&](jeandle::ObjectID InnerID) {
    SmallVector<Value *, 4> ToDrop;
    for (auto &AKv : Aliases.virtualAliasesView())
      if (AKv.second == InnerID)
        ToDrop.push_back(AKv.first);
    for (Value *V : ToDrop)
      Aliases.resetAlias(V);
  };
  auto ComputeSafeIP = [&]() -> Instruction * {
    // The default mode reaches only the escaping instruction; the option may
    // select the nearest safe prior deopt point.
    return Anchor ? Anchor->InsertBefore : InsertBefore;
  };
  auto SetEffectFlags = [&](jeandle::MaterializeEffect &E, Instruction *) {
    if (Anchor) {
      E.DeoptBundleSource = Anchor->DeoptSource;
    } else {
      E.CommitToOriginalAllocation = true;
      keepOriginalAllocation(ID);
    }
  };
  auto FlipState = [&](jeandle::ObjectID Oid) {
    CurrentState.getObjectStateForModification(Oid).escape(
        Result.VirtualObjects[Oid]->AllocationCall);
  };

  auto Recurse = [&](jeandle::ObjectID Oid, MatReason R) {
    materializeAt(Oid, InsertBefore, R);
  };
  auto CaptureLocksIntoEffect = [](ArrayRef<LockEnter> Stack, jeandle::ObjectID,
                                   jeandle::MaterializeEffect &E) {
    captureMaterializedLocks(Stack, E);
  };
  // Field-replay value: OrigAlloc (single global-RAUW materialize).
  auto MaterializedValue = [&](jeandle::ObjectID Oid) {
    return Result.VirtualObjects[Oid]->AllocationCall;
  };
  // NOTE: every callback is a named local (not a temporary in the aggregate
  // init) so each outlives C — function_ref does NOT own its callable, and a
  // temporary would be destroyed at the end of the `C{...};` statement, leaving
  // a dangling ref for the ensureMaterialized call on the next line.
  MaterializeContext C{
      FieldStates,      LockCounts,    LiveLockEnters, Materialized,
      Reason,           Recurse,       ClearLockState, CaptureLocksIntoEffect,
      DropInnerAliases, ComputeSafeIP, SetEffectFlags, FlipState,
      MaterializedValue};
  ensureMaterialized(ID, C);
}

void Analyzer::keepOriginalAllocation(jeandle::ObjectID ID) {
  bool Removed = false;
  for (auto &Kv : Result.BlockEffects) {
    Kv.second.eraseIf([&](const jeandle::Effect &E) {
      if (E.ObjID == ID && isa<jeandle::EliminateAllocationEffect>(E)) {
        Removed = true;
        return true;
      }
      return false;
    });
  }
  if (Removed) {
    --Result.VirtualizationDelta;
    ++Result.AllocationDelta;
  }
}

void Analyzer::dropEffectsFor(jeandle::ObjectID ID) {
  bool DroppedAllocation = false;
  for (auto &Kv : Result.BlockEffects) {
    Kv.second.eraseIf([&](const jeandle::Effect &E) {
      if (E.ObjID != ID)
        return false;
      if (isa<jeandle::EliminateAllocationEffect>(E))
        DroppedAllocation = true;
      return true;
    });
  }
  if (DroppedAllocation) {
    --Result.VirtualizationDelta;
    ++Result.AllocationDelta;
  }
  Result.EscapeClassification[ID] =
      jeandle::PEAResult::EscapeKind::AlwaysEscapes;
}

void Analyzer::validateAllocationEliminationPlans() {
  // Soundness guard for NeverEscapes candidates. Eliminating an
  // allocation RAUWs the original allocation result to poison in the
  // transform. That is only valid if every use reachable from the original
  // allocation is also erased/replaced by this PEA plan, or if a global
  // materialized pointer rewrite (Materialize / RAUW-to-PHI) will replace the
  // allocation before the elimination sweep. If a live use remains, keep the
  // original IR by marking the VO ineligible and dropping its effects below.
  struct PlannedRemoval {
    jeandle::ObjectID Owner;
    Instruction *Target;
  };
  struct PlannedEdgeRemoval {
    jeandle::ObjectID Owner;
    BasicBlock *From;
    BasicBlock *To;
  };
  SmallVector<PlannedRemoval, 16> PlannedRemovals;
  SmallVector<PlannedEdgeRemoval, 8> PlannedEdgeRemovals;
  DenseMap<jeandle::ObjectID, DenseSet<Value *>> DeoptRecordedValues;
  DenseSet<jeandle::ObjectID> HasGlobalPointerRewrite;
  DenseSet<jeandle::ObjectID> HasAllocationElimination;
  for (const auto &Kv : Result.BlockEffects) {
    for (const auto &E : Kv.second) {
      switch (E.getKind()) {
      case jeandle::Effect::Kind::ReplaceLoad:
      case jeandle::Effect::Kind::ReplaceCall:
      case jeandle::Effect::Kind::EliminateStore:
      case jeandle::Effect::Kind::EliminateAllocation:
        if (Instruction *Target = E.getTarget()) {
          PlannedRemovals.push_back({E.ObjID, Target});
          if ((E.getKind() == jeandle::Effect::Kind::ReplaceCall ||
               E.getKind() == jeandle::Effect::Kind::EliminateAllocation))
            if (auto *II = dyn_cast<InvokeInst>(Target))
              PlannedEdgeRemovals.push_back(
                  {E.ObjID, II->getParent(), II->getUnwindDest()});
          if (E.getKind() == jeandle::Effect::Kind::EliminateAllocation)
            HasAllocationElimination.insert(E.ObjID);
        }
        break;
      case jeandle::Effect::Kind::Materialize:
        if (const auto *M = dyn_cast<jeandle::MaterializeEffect>(&E);
            M && !M->IsPerPred && !M->CommitToOriginalAllocation)
          HasGlobalPointerRewrite.insert(E.ObjID);
        break;
      case jeandle::Effect::Kind::RecordDeoptState:
        if (const auto *R = dyn_cast<jeandle::RecordDeoptStateEffect>(&E))
          for (const WeakTrackingVH &VH : R->Snapshot.Values)
            if (Value *V = VH)
              DeoptRecordedValues[E.ObjID].insert(V);
        break;
      case jeandle::Effect::Kind::CreatePHI:
        if (const auto *C = dyn_cast<jeandle::CreatePHIEffect>(&E);
            C && C->RAUWOrigToPHI)
          HasGlobalPointerRewrite.insert(E.ObjID);
        break;
      case jeandle::Effect::Kind::RewritePhiIncoming:
        break;
      }
    }
  }

  auto IsIdentityForwarder = [](Instruction *I) {
    return isa<GetElementPtrInst>(I) || isa<BitCastInst>(I) ||
           isa<AddrSpaceCastInst>(I) || isa<FreezeInst>(I) ||
           isa<SelectInst>(I) || isa<PHINode>(I);
  };

  auto IsHarmlessIntrinsicUse = [](IntrinsicInst *II) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::assume:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::experimental_noalias_scope_decl:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::dbg_label:
    case Intrinsic::donothing:
    case Intrinsic::sideeffect:
    case Intrinsic::is_constant:
      return true;
    default:
      return false;
    }
  };

  DenseSet<Instruction *> ActiveRemovedTargets;
  DenseSet<BasicBlock *> ReachableBlocks;

  std::function<bool(Value *, jeandle::ObjectID, SmallPtrSetImpl<Value *> &)>
      HasLiveUseOutsidePlan =
          [&](Value *V, jeandle::ObjectID ID,
              SmallPtrSetImpl<Value *> &Visited) -> bool {
    for (Use &U : V->uses()) {
      auto *I = dyn_cast<Instruction>(U.getUser());
      if (!I)
        return true;

      if (I->getParent() && !ReachableBlocks.count(I->getParent()))
        continue;

      if (ActiveRemovedTargets.count(I))
        continue;

      if (isDeoptBundleOperand(I, U)) {
        auto DIt = DeoptRecordedValues.find(ID);
        if (DIt != DeoptRecordedValues.end() && DIt->second.count(V))
          continue;
      }

      if (auto *II = dyn_cast<IntrinsicInst>(I)) {
        if (IsHarmlessIntrinsicUse(II))
          continue;
        if (II->getIntrinsicID() == Intrinsic::launder_invariant_group ||
            II->getIntrinsicID() == Intrinsic::strip_invariant_group ||
            II->getIntrinsicID() == Intrinsic::ptr_annotation) {
          if (!Visited.insert(I).second)
            continue;
          if (HasLiveUseOutsidePlan(I, ID, Visited))
            return true;
          continue;
        }
        return true;
      }

      if (IsIdentityForwarder(I)) {
        if (!Visited.insert(I).second)
          continue;
        if (HasLiveUseOutsidePlan(I, ID, Visited))
          return true;
        continue;
      }

      return true;
    }
    return false;
  };

  bool EligibilityChanged;
  do {
    ActiveRemovedTargets.clear();
    DenseSet<std::pair<BasicBlock *, BasicBlock *>> ActiveRemovedEdges;
    for (const PlannedRemoval &Plan : PlannedRemovals)
      if (Eligible.lookup(Plan.Owner))
        ActiveRemovedTargets.insert(Plan.Target);
    for (const PlannedEdgeRemoval &Plan : PlannedEdgeRemovals)
      if (Eligible.lookup(Plan.Owner))
        ActiveRemovedEdges.insert({Plan.From, Plan.To});

    ReachableBlocks.clear();
    SmallVector<BasicBlock *, 16> Worklist;
    Worklist.push_back(&F.getEntryBlock());
    while (!Worklist.empty()) {
      BasicBlock *BB = Worklist.pop_back_val();
      if (!ReachableBlocks.insert(BB).second)
        continue;
      for (BasicBlock *Succ : successors(BB))
        if (!ActiveRemovedEdges.count({BB, Succ}))
          Worklist.push_back(Succ);
    }

    EligibilityChanged = false;
    for (auto &VObjUP : Result.VirtualObjects) {
      jeandle::ObjectID ID = VObjUP->getID();
      auto EIt = Eligible.find(ID);
      bool IsEligible = (EIt != Eligible.end()) && EIt->second;
      if (!IsEligible || !HasAllocationElimination.count(ID) ||
          HasGlobalPointerRewrite.count(ID))
        continue;
      SmallPtrSet<Value *, 16> Visited;
      Visited.insert(VObjUP->AllocationCall);
      if (HasLiveUseOutsidePlan(VObjUP->AllocationCall, ID, Visited)) {
        Eligible[ID] = false;
        EligibilityChanged = true;
      }
    }
  } while (EligibilityChanged);
}

void Analyzer::commit() {
  // Every replacement allocation must be emitted exactly at a surviving call
  // or invoke carrying the deopt state it reuses.
  DenseSet<Instruction *> PlannedEffectRemovals;
  for (const auto &Kv : Result.BlockEffects) {
    for (const auto &E : Kv.second) {
      switch (E.getKind()) {
      case jeandle::Effect::Kind::ReplaceLoad:
      case jeandle::Effect::Kind::ReplaceCall:
      case jeandle::Effect::Kind::EliminateStore:
      case jeandle::Effect::Kind::EliminateAllocation:
        if (Instruction *Target = E.getTarget())
          PlannedEffectRemovals.insert(Target);
        break;
      default:
        break;
      }
    }
  }

  for (const auto &Kv : Result.BlockEffects) {
    for (const auto &E : Kv.second) {
      const auto *M = dyn_cast<jeandle::MaterializeEffect>(&E);
      if (!M)
        continue;
      if (M->CommitToOriginalAllocation)
        continue;
      auto *Point = dyn_cast_or_null<Instruction>(
          static_cast<Value *>(M->InsertBefore));
      CallBase *DeoptPoint = getCallOrInvokeWithDeoptState(Point);
      auto *DeoptSource = dyn_cast_or_null<CallBase>(
          static_cast<Value *>(M->DeoptBundleSource));
      if (!DeoptPoint || DeoptSource != DeoptPoint ||
          PlannedEffectRemovals.count(Point))
        Eligible[E.ObjID] = false;
    }
  }

  // Any virtual whose monitor lock count is non-zero at the end of analysis
  // has unbalanced enter/exit pairs (e.g., only an enter was seen, or the
  // matching exit lives in a different block we don't track yet). Mark
  // those ineligible up-front; the per-object loop below will drop their
  // effects.
  for (auto &Kv : LockCounts) {
    if (Kv.second != 0)
      Eligible[Kv.first] = false;
  }

  validateAllocationEliminationPlans();

  // Transitive ineligibility cascade over FieldStates VirtualRef entries.
  // Any virtual referenced by a kept-real (ineligible) object's field must
  // also be kept real: a surviving real store into that field (its
  // EliminateStoreEffect was dropped by dropEffectsFor below) would
  // otherwise write a NeverEscapes object's OrigAlloc, which Pass 2 RAUWs
  // to poison -> miscompile. This is the complete backstop: it catches
  // VirtualRefs recorded at ANY point — before the object was
  // markIneligible'd (e.g. `arr[0]=new T(); arr[i]=new T()` — the constant
  // store precedes the symbolic bail) AND after (`arr[i]=new T();
  // arr[0]=new T()` — the constant store follows it, recorded because
  // processStore does not check Eligible). A walk-time cascade in
  // markIneligible would be order-dependent and miss the latter; doing the
  // closure once at classification time is order-independent and covers
  // every recording path (processStore + the merge/per-pred propagators).
  // Graal has no analog: it materializes in place at the escape point, so
  // the recursive entry cascade (materializeWithCommit,
  // PartialEscapeBlockState.java:293-343) runs at the very point the outer
  // is committed; Jeandle's analysis/transform split defers classification
  // to here. Visited defends cycles (a.f=b, b.g=a).
  {
    SmallVector<jeandle::ObjectID, 8> WList;
    DenseSet<jeandle::ObjectID> VSet;
    for (auto &Kv : Eligible)
      if (!Kv.second)
        WList.push_back(Kv.first);
    while (!WList.empty()) {
      jeandle::ObjectID Cur = WList.pop_back_val();
      if (!VSet.insert(Cur).second)
        continue;
      Eligible[Cur] = false;
      auto FSIt = FieldStates.find(Cur);
      if (FSIt != FieldStates.end())
        for (auto &Kv2 : FSIt->second)
          if (Kv2.second.isVirtualRef())
            WList.push_back(Kv2.second.getVirtualRef());
    }
  }

  // Iterate by dense ObjectID order for determinism. The only remaining
  // post-pass cleanup is dropping effects for objects that became
  // ineligible during the walk (lock imbalance above; nested-virtual
  // discovery; access-handler type mismatch / non-const offset; etc.).
  // Cross-block escapes trigger materialization (they do not disqualify an
  // object).
  for (auto &VObjUP : Result.VirtualObjects) {
    jeandle::ObjectID ID = VObjUP->getID();
    auto EIt = Eligible.find(ID);
    bool IsEligible = (EIt != Eligible.end()) && EIt->second;
    if (!IsEligible)
      dropEffectsFor(ID);
  }

  // -------------------------------------------------------------------------
  // EscapeClassification population.
  //
  // dropEffectsFor() already stamped AlwaysEscapes onto every ineligible VO.
  // For each surviving (eligible) VO, classify based on whether ANY
  // Materialize effect survived in the committed plan:
  //   * no Materialize  -> NeverEscapes      (alloc fully eliminated)
  //   * any Materialize -> PartiallyEscapes  (virtual until an escape point,
  //                                           then committed to a real object)
  // Maps to NEVER vs PARTIAL vs ALWAYS escape classification.
  // -------------------------------------------------------------------------
  DenseSet<jeandle::ObjectID> HasSurvivingMaterialize;
  unsigned SurvivingEliminations = 0;
  for (const auto &Kv : Result.BlockEffects) {
    for (const auto &E : Kv.second) {
      if (isa<jeandle::MaterializeEffect>(E))
        HasSurvivingMaterialize.insert(E.ObjID);
      else if (isa<jeandle::EliminateAllocationEffect>(E))
        ++SurvivingEliminations;
    }
  }
  for (auto &VObjUP : Result.VirtualObjects) {
    jeandle::ObjectID ID = VObjUP->getID();
    // dropEffectsFor stamped AlwaysEscapes; skip those VOs.
    if (Result.EscapeClassification.count(ID))
      continue;
    Result.EscapeClassification[ID] =
        HasSurvivingMaterialize.count(ID)
            ? jeandle::PEAResult::EscapeKind::PartiallyEscapes
            : jeandle::PEAResult::EscapeKind::NeverEscapes;
  }

  // For every VO that ended up NeverEscapes (alloc will be
  // EliminateAllocation'd and its OrigAlloc users will RAUW to poison
  // in Pass 2), schedule the Case-B aliased PHIs for explicit erasure.
  // The transform's post-Pass-2 hook walks Result.CaseBAliasedPhisToErase
  // and runs RAUW(poison) + eraseFromParent on each surviving handle.
  // We deliberately do NOT schedule erasure for PartiallyEscapes or
  // AlwaysEscapes VOs: there, the PHI still routes a live materialized
  // pointer (or the OrigAlloc itself) and erasing would corrupt SSA.
  for (auto &Kv : CaseBPhiAliases) {
    jeandle::ObjectID ID = Kv.first;
    auto CIt = Result.EscapeClassification.find(ID);
    if (CIt == Result.EscapeClassification.end())
      continue;
    if (CIt->second != jeandle::PEAResult::EscapeKind::NeverEscapes)
      continue;
    for (PHINode *Phi : Kv.second) {
      if (!Phi)
        continue;
      Result.CaseBAliasedPhisToErase.emplace_back(Phi);
    }
  }
  // Count of EliminateAllocation effects that actually survived
  // commit() (i.e. allocations that will be removed by the transform).
  JeandlePEAEliminated += SurvivingEliminations;

  // -------------------------------------------------------------------------
  // -jeandle-dump-pea-stats summary line on stderr.
  // -------------------------------------------------------------------------
  if (JeandleDumpPEAStats) {
    unsigned NeverEsc = 0, PartialEsc = 0, AlwaysEsc = 0;
    for (const auto &Kv : Result.EscapeClassification) {
      switch (Kv.second) {
      case jeandle::PEAResult::EscapeKind::NeverEscapes:
        ++NeverEsc;
        break;
      case jeandle::PEAResult::EscapeKind::PartiallyEscapes:
        ++PartialEsc;
        break;
      case jeandle::PEAResult::EscapeKind::AlwaysEscapes:
        ++AlwaysEsc;
        break;
      }
    }
    llvm::errs() << ";; PEA stats @" << F.getName()
                 << ": NeverEscapes=" << NeverEsc
                 << " PartiallyEscapes=" << PartialEsc
                 << " AlwaysEscapes=" << AlwaysEsc << "\n";
  }
}

// Preorder (outer-first) collection of all loops reachable from L. Determinism
// follows LoopInfo::iterator / Loop::getSubLoops ordering, which mirrors the
// source-IR block order.
static void collectLoopsPreorder(Loop *L, SmallVectorImpl<Loop *> &Out) {
  Out.push_back(L);
  for (Loop *Sub : L->getSubLoops())
    collectLoopsPreorder(Sub, Out);
}

// Loop soundness: after the RPO walk has populated BlockExits but before
// commit(), force-materialize every virtual that is still virtual at any
// loop preheader's terminator. This makes loops trivially sound — no
// virtual survives the loop boundary — at the cost of giving up
// virtualization across loops. Combined with the processAllocation refusal of
// loop-body allocs, the analyzer never tracks an object across a back-edge.
//
// Important sequencing: this MUST run after the per-block analysis (we need
// BlockExits[preheader] to know what's still virtual on the way into the loop)
// and BEFORE commit() (so the Materialize effects we add are subject to the
// same eligibility filter that drops effects for objects we've decided to
// abandon).
// A 0-clause cleanup landingpad block that is exactly `landingpad <t> cleanup;
// resume <t> %lp` — the canonical OOM handler — cannot observe a loop-local
// virtual, so a loop exit into it need not force materialization. Any other
// EH-pad successor (catch handler, clause-bearing landingpad, or a cleanup
// that does more than resume) may observe, so it must.
static bool isPureResumeCleanup(const BasicBlock &BB) {
  const Instruction *LP = BB.getFirstNonPHI();
  if (!isa<LandingPadInst>(LP))
    return false;
  const Instruction *Next = LP->getNextNode();
  return Next == &BB.back() && isa<ResumeInst>(Next);
}

void Analyzer::processLoopExit(Loop *L) {
  // Force materialisation at any loop-exit whose successor is an EH pad that
  // may observe a loop-local virtual. Without this, a loop-internal materialise
  // that produced only per-pred MatPerBlock entries leaves exit-edge LCSSA PHIs
  // (and exception handlers reading them) wired to stale OrigAlloc
  // placeholders; the forced drain ensures every still-virtual VO has a real
  // allocation by the time the EH handler executes.
  // Graal runs processLoopExit unconditionally (PartialEscapeClosure.java:737-
  // 766) and force-materializes loop-local virtuals at any exception-handling
  // exit. Jeandle preserves the common OOM-cleanup optimization by skipping
  // only a pure-resume 0-clause cleanup (isPureResumeCleanup), which provably
  // cannot observe; every other EH-pad successor forces materialization.
  // TODO(jeandle-deopt): see PartialEscapeTransform.cpp applyMaterialize().
  SmallVector<BasicBlock *, 4> ExitingBBs;
  L->getExitingBlocks(ExitingBBs);
  for (BasicBlock *ExitingBB : ExitingBBs) {
    Instruction *Term = ExitingBB->getTerminator();
    if (!Term)
      continue;
    bool ExitsToEH = false;
    for (BasicBlock *Succ : successors(ExitingBB)) {
      if (L->contains(Succ))
        continue;
      // Graal runs processLoopExit unconditionally for every LoopExitNode and
      // force-materializes every loop-local virtual whose exit reaches an
      // exception-handling context (PartialEscapeClosure.java:737-766, gated
      // only by isExceptionHandlingBCI). Any EH-pad successor counts —
      // catchswitch, catchpad, cleanuppad, or landingpad: a 0-clause cleanup
      // handler is NOT guaranteed to only `resume`; legal IR may read a
      // loop-local virtual's field from it. Jeandle preserves the OOM-cleanup
      // optimization by skipping only a pure-resume 0-clause landingpad
      // (isPureResumeCleanup). isEHPad() covers all four EH-pad kinds (and any
      // future one), so funclet-EH cleanuppad successors are not missed.
      // TODO(jeandle-deopt): see PartialEscapeTransform.cpp applyMaterialize().
      Instruction *First = &*Succ->getFirstNonPHIIt();
      if (First->isEHPad()) {
        // A catch handler (clauses > 0) or a non-trivial cleanup may observe a
        // loop-local virtual; a pure-resume 0-clause landingpad (OOM handler)
        // cannot. cleanuppad/catchpad/catchswitch have no pure-resume exemption
        // (isPureResumeCleanup requires a LandingPadInst).
        bool PureOOMResume = false;
        if (auto *LP = dyn_cast<LandingPadInst>(First))
          PureOOMResume =
              LP->getNumClauses() == 0 && isPureResumeCleanup(*Succ);
        if (!PureOOMResume) {
          ExitsToEH = true;
          break;
        }
      }
    }
    if (!ExitsToEH)
      continue;
    auto It = BlockExits.find(ExitingBB);
    if (It == BlockExits.end())
      continue;
    BlockExitInfo &Exit = It->second;
    SmallVector<jeandle::ObjectID, 4> Vs(Exit.Virtuals.begin(),
                                         Exit.Virtuals.end());
    llvm::sort(Vs);
    for (jeandle::ObjectID ID : Vs) {
      if (!Eligible.lookup(ID))
        continue;
      // Materialise at the exiting block's terminator. The IP is the
      // last legal point inside the loop body; the EH pad inherits a
      // fully-materialised state.
      materializeAtPredFromExitInfo(ID, ExitingBB, Exit,
                                    /*SkipGlobalRAUW=*/false,
                                    MatReason::LoopExit);
    }
  }
}

void Analyzer::processStateBeforeLoopOnOverflow(Loop *L) {
  // Every VO still virtual at the loop's forward end (preheader exit) is
  // forcibly materialised, so the re-do of the loop body in
  // MATERIALIZE_ALL mode starts from a clean "no live virtuals on entry"
  // state.
  BasicBlock *PH = L->getLoopPreheader();
  if (!PH)
    return;
  auto It = BlockExits.find(PH);
  if (It == BlockExits.end())
    return;
  BlockExitInfo &PHExit = It->second;
  SmallVector<jeandle::ObjectID, 4> Vs(PHExit.Virtuals.begin(),
                                       PHExit.Virtuals.end());
  llvm::sort(Vs);
  for (jeandle::ObjectID ID : Vs) {
    if (!Eligible.lookup(ID))
      continue;
    materializeAtPredFromExitInfo(ID, PH, PHExit, /*SkipGlobalRAUW=*/false,
                                  MatReason::LoopExit);
  }
}

void Analyzer::materializePreheaderVirtualsForUnvisitedLoops() {
  SmallVector<Loop *, 8> AllLoops;
  for (Loop *L : LI)
    collectLoopsPreorder(L, AllLoops);

  for (Loop *L : AllLoops) {
    BasicBlock *PH = L->getLoopPreheader();
    if (!PH) {
      // Loops without a unique preheader are drained in-place by
      // processLoop (it materialises every still-virtual VO at every
      // forward header predecessor). This branch is a defense-in-depth
      // no-op: the safety net cannot pick a single PH to drain at when
      // none exists, so the only sound action here is to skip.
      continue;
    }
    // Strict gate on VisitedLoops. Every loop processLoop touched —
    // whether the body fixpoint converged, fell into the pessimistic
    // MATERIALIZE_ALL fallback, or hit the overflow-recovery retry path
    // — already had its preheader virtuals handled inside processLoop.
    // The only loops that need this safety-net drain are those
    // processLoop never visited (an unreachable top-level loop the RPO
    // walk skipped, or a sub-loop whose outer recursion returned early
    // on OverflowFlag).
    if (VisitedLoops.count(L))
      continue;
    auto It = BlockExits.find(PH);
    if (It == BlockExits.end())
      continue;
    BlockExitInfo &PHExit = It->second;

    // Snapshot+sort the virtual IDs to materialize. Sorting by ObjectID gives
    // deterministic effect ordering across runs.
    SmallVector<jeandle::ObjectID, 4> Vs(PHExit.Virtuals.begin(),
                                         PHExit.Virtuals.end());
    llvm::sort(Vs);
    for (jeandle::ObjectID ID : Vs) {
      if (!Eligible.lookup(ID))
        continue;
      materializeAtPredFromExitInfo(ID, PH, PHExit, /*SkipGlobalRAUW=*/false,
                                    MatReason::LoopExit);
    }
  }
}

// Like materializeAt, but operates against a pred's BlockExitInfo snapshot
// rather than the analyzer's current per-block state (which has moved on by
// the time materializeBeforeLoops runs). The function-wide MaterializedAtPred
// map dedups (and breaks cycles between) recursive nested-virtual
// materializations within a single PH and across multiple call sites
// (e.g. a mixed-state merge and a loop-preheader sweep at the same PH).
//
// TargetMerge is the merge block this materialize is destined for (the
// MergeProcessor::BB in scope), or null for the Case-A / global path. It keys
// the placeholder cache and MaterializedAtPred so distinct target merges each
// get their own per-pred mat at the same PH.
//
// IR-form divergence from Graal (the core rule this function encodes):
//  - Per-pred mat (SkipGlobalRAUW=true): the invoke is placed on ONE critical
//    edge (invoke has a single normal dest + OOM unwind), so its NewInv
//    dominates only that edge. Therefore the per-pred path does NOT flip the
//    shared `ExitInfo` (`BlockExits[PH]`) — a flip would leak to non-target
//    successors, who would inherit a placeholder whose NewInv doesn't dominate
//    them. The placeholder is handed to the caller directly
//    (materializeAndBuildPhi reads it via getOrCreatePerPredMatPlaceholder).
//  - Case-A / global mat (!SkipGlobalRAUW): the invoke is at the pred's
//    terminator end, so its NewInv dominates ALL successors (Graal-aligned —
//    Graal's CommitAllocationNode sits before the control split). The flip of
//    the shared `ExitInfo` is therefore benign and is applied, so other merges
//    sharing this pred see the materialization.
void Analyzer::materializeAtPredFromExitInfo(
    jeandle::ObjectID ID, BasicBlock *PH, BlockExitData &ExitInfo,
    bool SkipGlobalRAUW, MatReason Reason, BasicBlock *TargetMerge) {
  auto ClearLockState = [&](jeandle::ObjectID Oid) {
    LockCounts[Oid] = 0;
    LiveLockEnters.erase(Oid);
    // Case-A / global: clear the shared ExitInfo's lock state (original
    // behavior — the flip makes the VO materialized, so the lock state is
    // irrelevant). Per-pred: do NOT clear ExitInfo — per-pred doesn't flip
    // (NewInv dominates one edge), so clearing the shared pred lock state
    // would leak to other successors AND make the do/while retry see
    // locks-match (taking the wrong mergeFieldStates path instead of
    // re-entering materializeAndBuildPhi). The pred's locks are captured into
    // the Materialize effect once (on first emit); the retry's dedup skips
    // re-capture.
    if (!SkipGlobalRAUW) {
      ExitInfo.LockCounts.erase(Oid);
      ExitInfo.LiveLockEnters.erase(Oid);
    }
  };
  auto ComputeSafeIP = [&]() -> Instruction * {
    Instruction *Term = PH->getTerminator();
    if (JeandlePEAEnableAggressiveAllocationSinking)
      if (CallBase *CB = getCallOrInvokeWithDeoptState(Term->getPrevNode()))
        return CB;
    return Term;
  };
  auto SetEffectFlags = [&](jeandle::MaterializeEffect &E,
                              Instruction *SafeIP) {
    jeandle::VirtualObject &V = *Result.VirtualObjects[ID];
    E.IsPerPred = SkipGlobalRAUW;
    E.TargetMergeBB = TargetMerge;
    // Per-pred-distinct materialized value: a stable placeholder per (PH, M,
    // ID) (M=null for Case-A / global) the transform resolves to this pred's
    // own NewInv via MatPerBlock.
    E.PerPredPlaceholder =
        getOrCreatePerPredMatPlaceholder(PH, TargetMerge, ID);
    if (CallBase *CB = getCallOrInvokeWithDeoptState(SafeIP))
      E.DeoptBundleSource = CB;
    else
      E.DeoptBundleSource = V.AllocationCall;
  };
  auto FlipState = [&](jeandle::ObjectID Oid) {
    // Per-pred (SkipGlobalRAUW): no flip — the NewInv dominates one critical
    // edge, so mutating the shared pred state would leak to non-target
    // successors. Case-A / global (!SkipGlobalRAUW): flip the shared ExitInfo
    // (NewInv at PH end dominates all successors — Graal-aligned, benign).
    if (SkipGlobalRAUW)
      return;
    Value *Ph = getOrCreatePerPredMatPlaceholder(PH, TargetMerge, Oid);
    ExitInfo.Virtuals.erase(Oid);
    ExitInfo.Materialized.insert(Oid);
    ExitInfo.MaterializedValues[Oid] = Ph;
    ExitInfo.FieldStates.erase(Oid);
    ExitInfo.LockCounts.erase(Oid);
  };

  auto Recurse = [&](jeandle::ObjectID Oid, MatReason R) {
    materializeAtPredFromExitInfo(Oid, PH, ExitInfo, SkipGlobalRAUW, R,
                                  TargetMerge);
  };
  auto CaptureLocksIntoEffect = [](ArrayRef<LockEnter> Stack, jeandle::ObjectID,
                                   jeandle::MaterializeEffect &E) {
    captureMaterializedLocks(Stack, E);
  };
  auto DropInnerAliasesNop = [](jeandle::ObjectID) {};
  // Field-replay value: OrigAlloc, identical to the live path. Graal uses a
  // distinct AllocatedObjectNode per predecessor as the field value
  // (PartialEscapeBlockState.java:317); LLVM's analysis/transform split forbids
  // creating the concrete node during analysis, so OrigAlloc stands in and the
  // transform's point-sensitive resolution sub-pass (resolveMaterializedUses)
  // picks the per-pred NewInv that dominates each field store — recovering
  // Graal's per-pred distinctness via dominance. A per-pred placeholder here
  // would leak into the replayed field store before it is resolved to a real
  // NewInv.
  auto MaterializedValue = [&](jeandle::ObjectID Oid) {
    return Result.VirtualObjects[Oid]->AllocationCall;
  };
  // NOTE: every callback is a named local (not a temporary) so each outlives C
  // — function_ref does not own its callable; a temporary would dangle after
  // the `C{...};` statement (see the matching note in materializeAt).
  DenseSet<jeandle::ObjectID> &MatSet = MaterializedAtPred[PH][TargetMerge];
  MaterializeContext C{ExitInfo.FieldStates,
                       ExitInfo.LockCounts,
                       ExitInfo.LiveLockEnters,
                       MatSet,
                       Reason,
                       Recurse,
                       ClearLockState,
                       CaptureLocksIntoEffect,
                       DropInnerAliasesNop,
                       ComputeSafeIP,
                       SetEffectFlags,
                       FlipState,
                       MaterializedValue};
  ensureMaterialized(ID, C);
}

// ===========================================================================
// Real loop fixpoint  (Graal EffectsClosure.processLoop, EClosure.java:364-566)
// ===========================================================================
//
// Structure mirrors Graal: an OUTER retry loop wraps an INNER fixpoint (up to
// MaxLoopFixpointIters = 10 body passes). Each inner pass starts from a clean
// rollback to the pre-loop snapshot (LoopSnapshot). The fixpoint variable is
// the per-block exit state of every loop block: mergeStates(Header) on pass i+1
// sees the preheader BlockExitInfo plus the pass-i backedge BlockExitInfo, so
// decisions stabilise once the per-block exits do (Jeandle's BlockExits-based
// convergence stands in for Graal's equivalentTo on cloned state). Field PHIs
// in loop blocks MUST be stable across passes (same Value*) for the convergence
// comparison and for preserved BlockExits that feed later header merges — that
// is the purpose of LoopFieldPhiCache / OwnedLoopFieldPhis.
//
// On non-convergence OR overflow (OverflowFlag, latched in ensureMaterialized
// under Mode::StopNewInLoopNest), escalate to Mode::MaterializeAll ONCE and
// retry the whole inner fixpoint (Graal EClosure:550-561). The escalation
// restores the snapshot, wipes loop-block BlockExits, and drains preheader
// virtuals (processStateBeforeLoopOnOverflow) so the redo starts with no live
// virtuals on entry. A nested processLoop (depth>1) that observes overflow
// does NOT recover locally — it returns so the outermost (depth==1) loop owns
// the rollback + redo (Graal re-throws the overflow to depth==1,
// EClosure:526-532). TooManyIterationsSeen is LOCAL per processLoop; a second
// failure hard-bails (Graal throws GraalError, EClosure:563 — Jeandle falls
// back soundly by marking still-virtual VOs ineligible).

// Equality on BlockExitData (the per-object base data). Two payloads are
// equivalent iff every per-object dimension (still-virtual set, materialized
// set, FieldStates per offset, LockCounts, LiveLockEnters,
// MaterializedValues) matches by structural comparison.
bool Analyzer::exitDataEquivalent(const BlockExitData &A,
                                  const BlockExitData &B) {
  if (A.Virtuals.size() != B.Virtuals.size())
    return false;
  for (auto ID : A.Virtuals)
    if (!B.Virtuals.count(ID))
      return false;
  if (A.Materialized.size() != B.Materialized.size())
    return false;
  for (auto ID : A.Materialized)
    if (!B.Materialized.count(ID))
      return false;
  if (A.FieldStates.size() != B.FieldStates.size())
    return false;
  for (auto &Kv : A.FieldStates) {
    auto It = B.FieldStates.find(Kv.first);
    if (It == B.FieldStates.end())
      return false;
    if (Kv.second.size() != It->second.size())
      return false;
    for (auto &Off : Kv.second) {
      auto OIt = It->second.find(Off.first);
      if (OIt == It->second.end())
        return false;
      if (!Off.second.shallowEquals(OIt->second))
        return false;
    }
  }
  if (A.LockCounts.size() != B.LockCounts.size())
    return false;
  for (auto &Kv : A.LockCounts) {
    auto It = B.LockCounts.find(Kv.first);
    if (It == B.LockCounts.end() || It->second != Kv.second)
      return false;
  }
  if (A.LiveLockEnters.size() != B.LiveLockEnters.size())
    return false;
  for (auto &Kv : A.LiveLockEnters) {
    auto It = B.LiveLockEnters.find(Kv.first);
    if (It == B.LiveLockEnters.end())
      return false;
    if (Kv.second.size() != It->second.size())
      return false;
    for (size_t i = 0, e = Kv.second.size(); i != e; ++i)
      // Loop fixpoint convergence: only the call-site identity matters; the
      // Order tag is monotonically refreshed on re-pushes and never settles,
      // so comparing it would cause loops with virtual monitorenters to
      // diverge forever.
      if (Kv.second[i].Call != It->second[i].Call)
        return false;
  }
  if (A.MaterializedValues.size() != B.MaterializedValues.size())
    return false;
  for (auto &Kv : A.MaterializedValues) {
    auto It = B.MaterializedValues.find(Kv.first);
    if (It == B.MaterializedValues.end() || It->second != Kv.second)
      return false;
  }
  return true;
}

void Analyzer::takeLoopSnapshot(
    Loop *L, const llvm::SmallPtrSetImpl<BasicBlock *> &LoopBlocks,
    LoopSnapshot &S) {
  S.CurrentState = CurrentState;
  S.Aliases = Aliases.snapshot();
  S.FieldStates = FieldStates;
  S.LockCounts = LockCounts;
  S.LiveLockEnters = LiveLockEnters;
  S.Materialized = Materialized;
  // Snapshot per-VO eligibility so a transient bail in this iter does
  // not wedge the VO ineligible for iter N+1.
  S.EligibleSnapshot = Eligible;
  // Also snapshot the VO count, so on restore we can re-mark every
  // VO created inside this iter as eligible for the next iter's retry.
  S.PreIterVOCount = Result.VirtualObjects.size();
  S.NextSeqNo = Result.NextSeqNo;
  S.OwnedPhisSize = Result.OwnedPhis.size();
  S.OwnedInstsSize = Result.OwnedInsts.size();
  S.SavedBlockExits.clear();
  S.SavedBlockEffects.clear();
  S.SavedMaterializedAtPred.clear();
  S.HadBlockExits.clear();
  S.HadBlockEffects.clear();
  S.HadMaterializedAtPred.clear();
  for (BasicBlock *BB : LoopBlocks) {
    auto EIt = BlockExits.find(BB);
    if (EIt != BlockExits.end()) {
      S.HadBlockExits.insert(BB);
      S.SavedBlockExits[BB] = EIt->second;
    }
    auto FIt = Result.BlockEffects.find(BB);
    if (FIt != Result.BlockEffects.end()) {
      S.HadBlockEffects.insert(BB);
      S.SavedBlockEffects[BB] = FIt->second.clone();
    }
    auto MIt = MaterializedAtPred.find(BB);
    if (MIt != MaterializedAtPred.end()) {
      S.HadMaterializedAtPred.insert(BB);
      S.SavedMaterializedAtPred[BB] = MIt->second;
    }
  }
}

void Analyzer::restoreLoopSnapshot(
    const llvm::SmallPtrSetImpl<BasicBlock *> &LoopBlocks,
    const LoopSnapshot &S) {
  CurrentState = S.CurrentState;
  Aliases.restore(S.Aliases);
  FieldStates = S.FieldStates;
  LockCounts = S.LockCounts;
  LiveLockEnters = S.LiveLockEnters;
  Materialized = S.Materialized;
  // Restore Eligible map in full so transient per-iter bails do not
  // outlive the iteration they occurred in. Then re-mark every VO created
  // AFTER the snapshot was taken (i.e. allocs inside this iter's body) as
  // eligible — they survive in IR (we never erase invokes during analysis)
  // and the next iter's processAllocation hits AllocSiteToVO and would bail
  // early on `!Eligible.lookup(ID)` if we didn't re-prime it. Without
  // these post-snapshot re-marks, body-local allocations would be wedged
  // ineligible across the entire fixpoint after iter 0 — observable as
  // failure to virtualise a loop-local "new" with no escape.
  Eligible = S.EligibleSnapshot;
  for (size_t I = S.PreIterVOCount, E = Result.VirtualObjects.size(); I < E;
       ++I) {
    if (!Result.VirtualObjects[I])
      continue;
    Eligible[Result.VirtualObjects[I]->getID()] = true;
  }

  // Defensive invariant: no PRESERVED BlockExits[BB] (BB in LoopBlocks) may
  // reference an unparented PHI we are about to delete below. BlockExits for
  // loop blocks is deliberately preserved across rollback (see the rationale
  // at the bottom of this function), so every Value* reachable from a
  // preserved entry must outlive rollback. In-loop merge-block field PHIs
  // stay alive because getOrCreateLoopFieldPhi caches them (gated on
  // LI.getLoopFor(BB)) in OwnedLoopFieldPhis, which this cleanup does NOT
  // pop. Were a non-header in-loop merge to bypass that cache (landing its
  // PHI in OwnedPhis), the next iteration's mergeStates(Header) would read a
  // dangling Value* through the preserved BlockExits[BB] - use-after-free.
  // This check makes that regression deterministic.
  assert((([&] {
           SmallPtrSet<Value *, 16> ToDelete;
           for (size_t I = S.OwnedPhisSize; I < Result.OwnedPhis.size(); ++I)
             if (Value *V = Result.OwnedPhis[I])
               ToDelete.insert(V);
           for (BasicBlock *BB : LoopBlocks) {
             auto It = BlockExits.find(BB);
             if (It == BlockExits.end())
               continue;
             for (const auto &FS : It->second.FieldStates)
               for (const auto &FV : FS.second) {
                 if (FV.second.isScalar() &&
                     ToDelete.count(FV.second.getScalar()))
                   return false;
                 if (FV.second.isMaterializedRef() &&
                     ToDelete.count(FV.second.getMaterialized()))
                   return false;
               }
             for (const auto &MV : It->second.MaterializedValues)
               if (ToDelete.count(MV.second))
                 return false;
           }
           return true;
         })()) &&
         "BlockExits[loop-block] references an unparented PHI that "
         "restoreLoopSnapshot is about to delete; the in-loop merge PHI "
         "must be cached via getOrCreateLoopFieldPhi, not OwnedPhis");

  // Pop and delete unparented PHIs / insts created during the rolled-back
  // iteration. OwnedLoopFieldPhis are NOT touched - they're the per-loop
  // PHI cache, and the whole point of the cache is to keep them alive
  // across loop-fixpoint rollback.
  while (Result.OwnedPhis.size() > S.OwnedPhisSize) {
    WeakTrackingVH &VH = Result.OwnedPhis.back();
    if (Value *V = VH) {
      if (auto *P = dyn_cast<PHINode>(V))
        if (!P->getParent())
          delete P;
    }
    Result.OwnedPhis.pop_back();
  }
  while (Result.OwnedInsts.size() > S.OwnedInstsSize) {
    WeakTrackingVH &VH = Result.OwnedInsts.back();
    if (Value *V = VH) {
      if (auto *I = dyn_cast<Instruction>(V))
        if (!I->getParent())
          I->deleteValue();
    }
    Result.OwnedInsts.pop_back();
  }
  Result.NextSeqNo = S.NextSeqNo;

  // Roll back per-loop-block ledgers.
  //
  // BlockExits for loop blocks is deliberately NOT rolled back: the next
  // iteration's mergeStates(Header) reads each back-edge pred's BlockExits to
  // learn the loop-internal contribution (the back-edge pred is later in RPO
  // than the header, so wiping it would leave mergeStates seeing only the
  // preheader). snapshotExitState() at the end of each processBlock
  // overwrites the stale entry with the iteration's fresh result, and the
  // convergence check compares iter (N+1)'s CurExits against iter N's
  // LastExits, so a never-changing BlockExits still converges after one extra
  // iteration.
  //
  // BlockEffects and MaterializedAtPred MUST be rolled back (they accumulate
  // emitted-effect side-data; leaving them would duplicate effects across
  // iterations). LoopFieldPhiCache / OwnedLoopFieldPhis cover the stable
  // PHI Value* need for iter-spanning structural equivalence AND for the
  // BlockExits-preservation invariant above: getOrCreateLoopFieldPhi gates
  // the cache on LI.getLoopFor(BB), so EVERY in-loop merge-block PHI (header
  // or not) is cached and survives this rollback. Were a non-header in-loop
  // merge to bypass the cache, its PHI would land in OwnedPhis and be deleted
  // by the cleanup above while the preserved BlockExits[BB] still references
  // it - the debug assert earlier in this function guards exactly this.
  for (BasicBlock *BB : LoopBlocks) {
    auto SF = S.SavedBlockEffects.find(BB);
    if (S.HadBlockEffects.count(BB))
      Result.BlockEffects[BB] = SF->second.clone();
    else
      Result.BlockEffects.erase(BB);
    auto SM = S.SavedMaterializedAtPred.find(BB);
    if (S.HadMaterializedAtPred.count(BB))
      MaterializedAtPred[BB] = SM->second;
    else
      MaterializedAtPred.erase(BB);
  }
}

SmallVector<BasicBlock *, 32>
Analyzer::loopBlocksInRPO(Loop *L, ArrayRef<BasicBlock *> FunctionRPO) {
  // Function-RPO filtered to L's blocks. FunctionRPO is computed once by
  // Analyzer::run and reused by every loop in the function.
  SmallVector<BasicBlock *, 32> Order;
  for (BasicBlock *BB : FunctionRPO)
    if (L->contains(BB))
      Order.push_back(BB);
  return Order;
}

void Analyzer::processLoopBodyOnePass(Loop *L, ArrayRef<BasicBlock *> LoopRPO,
                                      ArrayRef<BasicBlock *> FunctionRPO) {
  // Process loop blocks in function-RPO order. Sub-loop headers dispatch
  // recursively to processLoop, and the sub-loop's blocks are marked Done so
  // we don't re-process them in this pass. FunctionRPO is computed once by
  // Analyzer::run; LoopRPO is the filtered loop-local view reused across the
  // inner fixpoint iterations.
  llvm::SmallPtrSet<BasicBlock *, 16> Done;
  for (BasicBlock *BB : LoopRPO) {
    if (Done.count(BB))
      continue;
    Loop *Inner = LI.getLoopFor(BB);
    if (Inner && Inner != L && Inner->getHeader() == BB) {
      // Found a sub-loop's header — recurse.
      processLoop(Inner, FunctionRPO);
      for (BasicBlock *SB : Inner->blocks())
        Done.insert(SB);
      continue;
    }
    processBlock(BB);
    Done.insert(BB);
  }
}

void Analyzer::processLoop(Loop *L, ArrayRef<BasicBlock *> FunctionRPO) {
  // Record EVERY processLoop entry (including for loops with no
  // header / no preheader / OverflowFlag short-circuit) so the safety-net
  // pass treats this loop as visited and skips it.
  VisitedLoops.insert(L);
  BasicBlock *Preheader = L->getLoopPreheader();
  BasicBlock *Header = L->getHeader();
  if (!Header) {
    // Defensive: a loop with no header would be malformed LoopInfo state.
    return;
  }
  if (!Preheader) {
    // Loop without a unique preheader. Jeandle schedules LoopSimplifyPass
    // before PEA, so natural reducible-CFG loops reach the fixpoint path
    // above; we land here only for cases LoopSimplify cannot canonicalise —
    // indirectbr-entered loops and genuinely irreducible cycles LoopInfo
    // still recognises as a natural loop with multiple entry edges.
    //
    // "Materialize at every forward predecessor" is not implementable in our
    // effects model: each Materialize records an OrigAlloc -> NewInv definition
    // per predecessor, so with N > 1 forward preds the in-loop uses are
    // dominated by MULTIPLE per-pred NewInvs with no synthesized header PHI to
    // merge them into a single dominating def (the existing Case-A pattern
    // works only because an explicit PHI already exists, which CreatePHI
    // registers as the single dominating def via MatPerBlock).
    //
    // Sound fallback: mark every VO still virtual at any forward predecessor
    // INELIGIBLE. commit() drops its effects and the original IR survives
    // unchanged — the body's pre-loop pointer is the original OrigAlloc on
    // every entry edge, which trivially dominates the body. Loop-local
    // allocs (created and consumed within a single iteration) are still
    // candidates: the body walk below runs once in REGULAR mode. They are
    // never virtual at any forward pred (forward preds are outside L), so the
    // sweep above doesn't touch them, and the missing fixpoint is irrelevant
    // because they don't cross the back-edge.

    // Collect forward (non-loop-back) predecessors of the header.
    llvm::SmallVector<BasicBlock *, 4> ForwardPreds;
    llvm::SmallPtrSet<BasicBlock *, 4> Seen;
    for (BasicBlock *P : predecessors(Header)) {
      if (L->contains(P))
        continue;
      if (Seen.insert(P).second)
        ForwardPreds.push_back(P);
    }

    // Bail every VO that is virtual at any forward pred — drop it back to the
    // original IR. BlockExits[P] is populated by the outer RPO walk (forward
    // preds are processed before processLoop is invoked on L).
    for (BasicBlock *P : ForwardPreds) {
      auto It = BlockExits.find(P);
      if (It == BlockExits.end())
        continue;
      BlockExitInfo &PExit = It->second;
      for (jeandle::ObjectID ID : PExit.Virtuals) {
        Eligible[ID] = false;
      }
    }

    // Body walk in REGULAR mode (single pass — no fixpoint, since there is
    // no way to verify convergence at a non-existent preheader). Loop-local
    // allocs that don't outlive a single iteration are still virtualised.
    processLoopBodyOnePass(L, loopBlocksInRPO(L, FunctionRPO), FunctionRPO);

    // Post-body merge (Graal doMergeWithoutDead run AFTER the body,
    // EffectsClosure.java:461-466). The in-pass header merge (header first in
    // RPO) runs before any loop-body alloc is virtualized, so it cannot resolve
    // an object allocated INSIDE the loop and carried across the back-edge via
    // a header pointer-phi — the back-edge slot is nullopt and the PHI is
    // skipped, which would misclassify the alloc NeverEscapes and RAUW it to
    // poison. Re-running mergeStates(Header) now that the latch BlockExits is
    // populated lets processBlockPhis Case A fire and materialize such a
    // carried object at the back-edge pred's terminator (Graal
    // ensureMaterialized at predecessor.getEndNode(),
    // PartialEscapeClosure.java:996/1504), matching the fixpoint path's
    // post-body merge. This is a one-shot merge (no convergence loop here), so
    // its effects simply persist to commit().
    resetPerBlockState();
    mergeStates(Header);
    PendingMergePhis[Header].clear();

    // Force-materialise at exits that flow into EH pads, matching the fixpoint
    // path (processLoopExit at :5220 / PartialEscapeClosure.java:737-799), so
    // exception handlers never see partially-materialised loop-internal
    // virtuals.
    processLoopExit(L);
    return;
  }

  llvm::SmallPtrSet<BasicBlock *, 8> LoopBlocks;
  for (BasicBlock *BB : L->blocks())
    LoopBlocks.insert(BB);

  // Loop blocks in function-RPO order, computed once and reused across the
  // inner fixpoint body passes below (the loop CFG is stable across PEA).
  SmallVector<BasicBlock *, 32> LoopRPO = loopBlocksInRPO(L, FunctionRPO);

  // At TOP-LEVEL processLoop entry only (loop.getDepth() == 1 gate),
  // compute the maximum loop depth within this nest. If it exceeds
  // JeandlePEALoopCutoff, transiently enter Mode::StopNewInLoopNest for
  // the duration of the fixpoint: processAllocation refuses NEW allocations,
  // but everything else (already-virtual tracking, merges, locks, loads,
  // stores, loop-exit handling) proceeds normally. The mode is restored
  // at the bottom of this function on convergence at depth==1.
  const Mode SavedModeForNest = CurrentMode;
  if (L->getLoopDepth() == 1) {
    // Count one outer-fixpoint entry per top-level processLoop call.
    ++JeandlePEAOuterFixpointIterations;

    // Graal flips STOP_NEW_VIRTUALIZATIONS_LOOP_NEST per-loop in
    // stripKilledLoopLocations when loop.depth > EscapeAnalysisLoopCutoff
    // (PartialEscapeClosure.java:675). Jeandle sets it nest-wide at the
    // outermost loop when the nest's max depth exceeds the cutoff — a
    // conservative equivalent (shallower loops in the nest also run StopNew,
    // but StopNew only suppresses NEW virtualizations, which is harmless).
    unsigned MaxDepth = L->getLoopDepth();
    for (Loop *Sub : L->getLoopsInPreorder())
      MaxDepth = std::max(MaxDepth, Sub->getLoopDepth());
    if (CurrentMode == Mode::Regular && MaxDepth > JeandlePEALoopCutoff)
      CurrentMode = Mode::StopNewInLoopNest;

    // Testing aid: optionally force MATERIALIZE_ALL for lit coverage.
    if (JeandlePEAForceMaterializeAll) {
      CurrentMode = Mode::MaterializeAll;
      ++JeandlePEAModeEscalations;
    }

    // Overflow recovery is scoped to one nest: clear the cross-recursion
    // signal on every top-level entry.
    OverflowFlag = false;
  } else {
    // Nested-loop entry. Graal re-throws EffecsClosureOverflowException up
    // to depth==1 (EffectsClosure.java:526-532); the polled equivalent is
    // to bail immediately if an outer overflow is already in progress.
    if (OverflowFlag)
      return;
  }

  // Snapshot pre-loop state once. The inner fixpoint and the MATERIALIZE_ALL
  // retry both restore from it (Jeandle snapshots per processLoop because
  // convergence is BlockExits-based and each loop level needs its own
  // rollback record; Graal snapshots only at depth==1, EffectsClosure:406).
  LoopSnapshot Pre;
  takeLoopSnapshot(L, LoopBlocks, Pre);

  // Graal's outer retry loop (EffectsClosure.java:439-565): run the Regular
  // inner fixpoint; on non-convergence OR overflow escalate to MATERIALIZE_ALL
  // once and retry the whole fixpoint; a second failure hard-bails.
  // TooManyIterationsSeen is LOCAL to each processLoop (one independent
  // escalation per loop), matching Graal's per-call local (EClosure:439).
  // Single-state B fixpoint context (Graal lastMergedState, paper §5.2.5) as
  // LOCALS — each processLoop call is its own C++ stack frame, so nesting is
  // isolated without a shared member (the outer's locals are untouched while a
  // recursive processLoop(inner) runs). B := A (EClosure:445): Jeandle has no
  // PEA-level killed-location strip (PEReadEliminationClosure machinery), so
  // the entry state is just the preheader's exit data, populated by the outer
  // RPO walk before processLoop is dispatched.
  BlockExitData LastMergedState =
      static_cast<const BlockExitData &>(BlockExits[Preheader]);
  BlockExitData NewMergedState; // B' each pass (post-body header merge result)

  bool TooManyIterationsSeen = false;
  while (true) {
    // ---- inner fixpoint: up to MaxLoopFixpointIters body passes ----
    // Single-state B convergence (paper §5.2.5 / Graal EClosure:439-524). B is
    // the header's merged state (seeded := A, the preheader exit). Each pass
    // runs the body, then a post-body merge computes B' = merge(A, fresh latch
    // exits) (Graal doMergeWithoutDead); converge when B' == B. Because the
    // post-body merge sees iteration 0's latch exits, the loop can converge in
    // a single body pass — matching Graal's structure exactly.
    bool Converged = false;
    for (unsigned Iter = 0; Iter < MaxLoopFixpointIters; ++Iter) {
      if (Iter > 0)
        restoreLoopSnapshot(LoopBlocks, Pre);
      ++JeandlePEALoopFixpointRetries;

      processLoopBodyOnePass(L, LoopRPO, FunctionRPO);

      // Overflow (a STOP_NEW materialization of an outer-scope VO) may have
      // been latched by this pass or a deeper recursion. Stop iterating: the
      // state is half-consistent and is rolled back below.
      if (OverflowFlag)
        break;

      // Post-body merge (Graal doMergeWithoutDead, EClosure:466): compute the
      // TRUE B' = merge(A, fresh latch end-states) AFTER the body pass. On
      // iteration 0 the in-pass header merge is just A — the latch BlockExits
      // is not yet populated when the header is processed — so only a post-body
      // merge sees this pass's latch exits, letting iteration 0 compare
      // meaningfully and the loop converge in a single body pass (paper §5.2.5
      // / Graal's structure).
      //
      // This merge runs AFTER the body (Graal runs the LoopBegin merge after
      // the body too, EffectsClosure.java:461-466), so it is the ONLY place
      // that can resolve an object allocated INSIDE the loop body and carried
      // across the back-edge via a header pointer-phi: the in-pass header merge
      // (header first in RPO) runs before that alloc is virtualized, so its
      // alias is not registered and the back-edge slot resolves to nullopt.
      // Here the latch BlockExits is populated and the alias is known, so
      // processBlockPhis Case A fires and materializes the carried object at
      // the back-edge pred's terminator (Graal ensureMaterialized at
      // predecessor.getEndNode(), PartialEscapeClosure.java:996/1504).
      //
      // The merge's effects are KEPT (Graal keeps blockEffects.get(predecessor)
      // on convergence, EffectsClosure.java:472-474) — no snapshot/restore
      // discard. A non-converged iteration's effects are cleared by the next
      // iteration's restoreLoopSnapshot(Pre) at the top of the loop (it
      // restores per-loop-block BlockEffects/MaterializedAtPred). The per-pred
      // materialized value is stable across iterations
      // (getOrCreatePerPredMatPlaceholder, cached per {PH,ID} in
      // OwnedMatPlaceholders which restoreLoopSnapshot does not pop), so B' is
      // stable and the fixpoint converges rather than escalating to
      // MATERIALIZE_ALL. PendingMergePhis[Header].clear() drops the re-run's
      // CreatePHI effects (redundant with the in-pass merge's already-drained
      // effects for before-loop objects; Case A records its materialize in
      // BlockEffects[latch], not PendingMergePhis).
      {
        resetPerBlockState();
        mergeStates(Header);
        NewMergedState = BlockExitData{};
        snapshotExitStateInto(NewMergedState); // B'
        PendingMergePhis[Header].clear();
      }
      // B' vs B (Graal EClosure:472). No iteration gate: with the post-body
      // merge, iteration 0 already has a true B' to compare against B := A.
      if (exitDataEquivalent(LastMergedState, NewMergedState)) {
        Converged = true;
        LLVM_DEBUG(dbgs() << "PEA: loop @ " << Header->getName()
                          << " converged in " << (Iter + 1)
                          << " iters (B-based, post-body)\n");
        break;
      }
      LastMergedState = NewMergedState; // B := B'   (Graal EClosure:512)
    }

    if (Converged) {
      // Force-materialise at exits that flow into EH pads, so exception
      // handlers never see partially-materialised loop-internal virtuals.
      processLoopExit(L);
      // Graal resets currentMode to REGULAR at depth==1 on success
      // (EffectsClosure.java:482-488). Nested loops leave the mode as-is so
      // an escalation persists through the rest of the nest.
      if (L->getLoopDepth() == 1)
        CurrentMode = SavedModeForNest;
      return;
    }

    LLVM_DEBUG(dbgs() << "PEA: loop @ " << Header->getName()
                      << " did not converge; "
                      << (OverflowFlag ? "overflow" : "iteration cap") << "\n");

    // Not converged: non-convergence (iteration cap) OR overflow.
    if (OverflowFlag && L->getLoopDepth() > 1) {
      // Graal: an inner loop re-throws the overflow until the outermost
      // (depth==1) loop catches it (EffectsClosure.java:526-532). Propagate
      // — do NOT recover locally; the outermost loop owns the rollback +
      // MATERIALIZE_ALL redo of the whole nest.
      return;
    }

    if (!TooManyIterationsSeen) {
      // First exhaustion/overflow: escalate to MATERIALIZE_ALL and retry the
      // whole fixpoint (Graal EffectsClosure.java:550-561 for overflow,
      // 553-561 for the iteration cap).
      TooManyIterationsSeen = true;
      if (CurrentMode != Mode::MaterializeAll)
        ++JeandlePEAModeEscalations;
      restoreLoopSnapshot(LoopBlocks, Pre);
      // Wipe loop-block BlockExits so the retry starts from a true pre-loop
      // view. (restoreLoopSnapshot intentionally preserves loop-block
      // BlockExits across iters for the Regular fixpoint's back-edge
      // contribution; the MATERIALIZE_ALL retry must not inherit the stale
      // virtuals it is trying to forget.)
      for (BasicBlock *BB : LoopBlocks)
        BlockExits.erase(BB);
      // Drain every still-virtual VO at the loop's forward end so the redo
      // starts with no live virtuals on entry — Graal's
      // processStateBeforeLoopOnOverflow (PartialEscapeClosure.java:230-237).
      processStateBeforeLoopOnOverflow(L);
      // Re-seed B := A: processStateBeforeLoopOnOverflow materializes pre-loop
      // virtuals at the loop's forward end, so BlockExits[Preheader] changed.
      // The MATERIALIZE_ALL redo must compare against the POST-overflow entry
      // state, not the stale pre-overflow LastMergedState.
      LastMergedState =
          static_cast<const BlockExitData &>(BlockExits[Preheader]);
      // Consume the overflow signal so the retry's nested processLoops run
      // rather than short-circuit on a stale flag.
      OverflowFlag = false;
      CurrentMode = Mode::MaterializeAll;
      continue;
    }

    // Second exhaustion: hard fail. Graal throws GraalError (fatal,
    // EffectsClosure.java:563). MATERIALIZE_ALL is expected to always
    // converge, so this indicates a pathological-IR / invariant gap; fall
    // back SOUNDLY by marking every still-virtual VO in the loop ineligible
    // (the original IR then survives unchanged — conservative, as in the
    // prior 2-stage path).
    LLVM_DEBUG(dbgs() << "PEA: loop @ " << Header->getName()
                      << " did NOT converge in MATERIALIZE_ALL; sound bail\n");
    for (BasicBlock *BB : LoopBlocks) {
      auto EIt = BlockExits.find(BB);
      if (EIt == BlockExits.end())
        continue;
      for (jeandle::ObjectID ID : EIt->second.Virtuals)
        Eligible[ID] = false;
    }
    if (L->getLoopDepth() == 1)
      CurrentMode = SavedModeForNest;
    return;
  }
}

jeandle::PEAResult Analyzer::run() {
  // Filter PEA to functions matching a name substring. Empty
  // option (the default) lets every gated Java method through unchanged.
  if (!JeandleEscapeAnalyzeOnly.empty() &&
      !F.getName().contains(JeandleEscapeAnalyzeOnly))
    return jeandle::PEAResult();

  // Outer walk: RPO over F. When we hit any block belonging to a top-level
  // loop, dispatch to processLoop on that top-level loop (which recursively
  // handles sub-loops). All other blocks are processed directly.
  llvm::SmallPtrSet<BasicBlock *, 16> Done;
  ReversePostOrderTraversal<Function *> RPOT(&F);
  SmallVector<BasicBlock *, 32> FunctionRPO;
  for (BasicBlock *BB : RPOT)
    FunctionRPO.push_back(BB);
  // Defensive sweep for cycles LoopInfo missed (indirectbr /
  // callbr / catchswitch back-edges). When the RPO walk reaches a non-
  // loop block whose preds include an as-yet-UNVISITED block, that
  // unvisited predecessor is a back-edge that LoopInfo did not
  // recognise. Without intervention, mergeStates would silently drop
  // the back-edge contribution (no exit data for it yet); the resulting
  // virtual state at BB would not reflect the cross-iteration field
  // mutations. Bail every still-virtual VO at BB entry to ineligibility
  // so the original IR survives unchanged.
  auto bailUnvisitedBackEdgeVOs = [&](BasicBlock *BB) {
    if (BB == &F.getEntryBlock())
      return;
    bool HasUnvisitedPred = false;
    for (BasicBlock *P : predecessors(BB)) {
      if (!Done.count(P)) {
        HasUnvisitedPred = true;
        break;
      }
    }
    if (!HasUnvisitedPred)
      return;
    // Conservative ineligibility flip on every VO that is currently
    // tracked as virtual in ANY processed pred's exit. Mirrors the
    // forward-pred sweep in the no-preheader fallback.
    for (BasicBlock *P : predecessors(BB)) {
      auto It = BlockExits.find(P);
      if (It == BlockExits.end())
        continue;
      for (jeandle::ObjectID ID : It->second.Virtuals)
        Eligible[ID] = false;
    }
    BlockExits.erase(BB);
  };
  for (BasicBlock *BB : FunctionRPO) {
    if (Done.count(BB))
      continue;
    Loop *L = LI.getLoopFor(BB);
    if (!L) {
      bailUnvisitedBackEdgeVOs(BB);
      processBlock(BB);
      Done.insert(BB);
      continue;
    }
    // Find the top-level loop containing BB. Sub-loops are dispatched
    // recursively from inside processLoop on the top-level loop.
    Loop *Top = L;
    while (Top->getParentLoop())
      Top = Top->getParentLoop();
    // If we haven't reached the top-level header in RPO yet (e.g. when a
    // block inside the loop appears before its header), still dispatch on
    // the top-level loop and mark all its blocks Done.
    processLoop(Top, FunctionRPO);
    for (BasicBlock *SB : Top->blocks())
      Done.insert(SB);
  }

  // Safety net — drain preheader virtuals ONLY for loops the RPO
  // walk never reached (unreachable top-level loops, or sub-loops whose
  // outer recursion bailed before recursing). Everything else has been
  // handled by processLoop directly. Loops with no preheader are a no-op
  // either way (the function cannot pick a drain IP without a PH).
  materializePreheaderVirtualsForUnvisitedLoops();
  commit();
  return std::move(Result);
}

} // anonymous namespace

PartialEscapeAnalysis::Result
PartialEscapeAnalysis::run(Function &F, FunctionAnalysisManager &FAM) {
  // Gate on jeandle.java_method_compilation: only Java methods are analyzed
  // (template module / runtime stubs are skipped).
  Module *M = F.getParent();
  if (!M || !M->getNamedMetadata(jeandle::Metadata::JavaMethodCompilation))
    return jeandle::PEAResult();

  // Request DominatorTree and LoopInfo eagerly so they're cached for later
  // PEA passes that need them.
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = FAM.getResult<LoopAnalysis>(F);

  Analyzer A(F, DT, LI, Options);
  return A.run();
}
