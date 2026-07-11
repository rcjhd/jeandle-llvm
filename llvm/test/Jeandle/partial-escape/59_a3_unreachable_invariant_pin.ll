; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; PEA A3 invariant pin: documents that the "alloc doesn't dominate merge"
; subcase of mergeStates' true-mixed branch is unreachable for non-synthetic
; VOs in the current single-pass RPO analyzer.
;
; Argument: for ObjectID X to appear in every predecessor of a merge BB,
; X's OrigAlloc must reach every pred via SSA flow (the analyzer only tracks
; an ID in a per-block state when that state was inherited from a pred that
; in turn knew the ID, eventually bottoming out at processAllocation in the
; alloc's own block). SSA dominance then requires OrigAlloc to dominate BB.
; If any pred lacks the ID, mergeStates short-circuits at MissingOnSomePred
; before reaching the true-mixed branch.
;
; Synthetic VOs (Case C) break the invariant — they have no real backing
; allocation. The bail in materializeAtPredFromExitInfo and the
; IsSynthetic check at the true-mixed branch jointly drop synthetics
; cleanly; see test 58.
;
; This test exercises a "shape" that is suggestive of A3 but is actually the
; alloc-dominates-merge fast path: alloc in a block (%n) that dominates the
; merge, escape on the if-then arm, virtual on the if-else arm. The merge
; sees mixed-state but alloc dominates.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @test_a3_invariant_pin(i1 %c0, i1 %c1)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %n
n:
  ; Alloc here dominates the merge below by SSA.
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %inner_if unwind label %u
inner_if:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c0, label %t1, label %e1
t1:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %m1
e1:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %m1
m1:
  ; m1's preds (t1, e1) both have %o tracked: t1 has Mat, e1 has Virtual.
  ; True-mixed branch fires. Alloc %o is in %n, which dominates %m1.
  ; The assertion in mergeStates passes.
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c1, label %t2, label %e2
t2:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %m2
e2:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %m2
m2:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret ptr addrspace(1) %o
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; Mat fires at the escape point in %t1, RAUWs %o to the new invoke. Both
; downstream merges (m1, m2) see the single materialized pointer.
; CHECK-LABEL: define ptr addrspace(1) @test_a3_invariant_pin
; Per-arm materialize + materializedValuePhi (the collapse).
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: call void @sink(ptr addrspace(1) %{{[A-Za-z0-9._]+}})
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: = phi ptr addrspace(1)
; CHECK: ret ptr addrspace(1) %{{[A-Za-z0-9._]+}}

!java-method-compilation = !{}
