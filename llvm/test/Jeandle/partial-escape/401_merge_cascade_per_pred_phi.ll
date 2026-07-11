; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Regression test for the MergeProcessor fast-path move (align with Graal
; PartialEscapeClosure.java:935). Two preds join at %merge with lock counts
; 0 (left) and 1 (right) on the same virtual object.
;
; Lock-cascade fires (counts disagree) and materializes at every pred:
;   left  (LC=0): emit the new alloc invoke, no live enters.
;   right (LC=1): emit the new alloc invoke and un-elide its one enter,
;                 snapping the receiver to right's own NewInv.
;
; After both materialize, each pred's per-pred materialized state is DISTINCT
; (each pred's own NewInv). The merge must build a ptr addrspace(1)
; materializedValuePhi selecting the two per-pred NewInvs and thread the
; post-merge return through it.
;
; WHY THIS TEST EXISTS: with the fast-path moved into the iterative merge
; do/while, a deep-value equivalence check (Graal uses reference-identity)
; would see both preds' materialized values collide and FALSE-POSITIVE the
; fast-path, inheriting one pred and dropping the selecting PHI — breaking
; SSA on the other pred's path. The per-pred-distinct value model makes the
; equivalence check sound, so the PHI is correctly built. This test fails
; on the raw move and passes once the fast-path is sound.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @test_merge_cascade_per_pred_phi(i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock = alloca i64, align 8
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %branch unwind label %u
branch:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %left, label %right
left:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
right:
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
              ptr addrspace(1) %o, ptr %lock)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret ptr addrspace(1) %o
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define ptr addrspace(1) @test_merge_cascade_per_pred_phi
; left (zero locks) materializes with no enter:
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK-NOT: call hotspotcc void @jeandle.monitorenter_with_thin_lock
; right (one lock) materializes and un-elides one enter snapped to its NewInv:
; CHECK: %[[MATR:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MATR]],
; No more enters or new-instance invokes after this.
; CHECK-NOT: call hotspotcc void @jeandle.monitorenter_with_thin_lock
; CHECK-NOT: invoke hotspotcc{{.*}}@jeandle.new_instance
; The merge selects the two per-pred NewInvs via a PHI, and the return
; consumes the PHI value (NOT one pred's NewInv).
; CHECK: %[[PHI:[A-Za-z0-9._]+]] = phi ptr addrspace(1)
; CHECK: ret ptr addrspace(1) %[[PHI]]

!java-method-compilation = !{}
