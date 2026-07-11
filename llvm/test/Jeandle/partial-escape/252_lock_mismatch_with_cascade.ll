; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Lock-mismatch on A triggers the per-pred materialize;
; the then-pred's materialize for A then cascades through the narrow
; strict-lock-order rule onto B (B is locked at lower depth — depth 0 —
; while A is locked at higher depth — depth 1 — in the then path; the
; rule "other.minOrder < this.maxOrder" matches). B becomes Materialized
; at the then-pred via cascade. Meanwhile the else-pred has no locks on
; either A or B, so the materialize at else for A doesn't cascade B.
;
; Resulting state at the merge:
;   * A: materialized on both preds (ptr PHI built on next retry).
;   * B: materialized on then-pred (cascade), still virtual on else-pred.
;     The mixed-merge path picks this up on retry; B's then-pred enter is
;     un-elided in IR with its receiver retargeted to the then-pred's
;     NewInv for B (no SSA violation — MatPerBlock-based ReplaceInput).
;
; The else-pred B remains harmless: no live locks at else, no surviving
; enter to retarget, and the merge has no downstream use of B.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_lock_mismatch_with_cascade(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock = alloca i64, align 8
  %oA = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
              ptr inttoptr (i64 11111 to ptr), i32 16)
       to label %allocB unwind label %u
allocB:
  %oB = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
              ptr inttoptr (i64 22222 to ptr), i32 16)
       to label %branch unwind label %u
branch:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %then, label %else
then:
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
              ptr addrspace(1) %oB, ptr %lock)
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
              ptr addrspace(1) %oA, ptr %lock)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
else:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_lock_mismatch_with_cascade
; then-pred: B (22222, the locked one) materializes first (lower SeqNo, the
; cascade prereq), then A (11111) materializes (the tail). The merged lock
; re-emit fires ONCE from the tail (A's MatCont), depth-sorted: B's enter
; (depth 0) then A's enter (depth 1) — both after both invokes.
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16)
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 11111 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock
; else-pred: A (11111) and B (22222) per-pred materialize via the merge
; collapse (no locks on this arm, so no surviving enters to re-emit).
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 11111 to ptr), i32 16)
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16)
; No more enters past this point.
; CHECK-NOT: call hotspotcc void @jeandle.monitorenter_with_thin_lock

!java-method-compilation = !{}
