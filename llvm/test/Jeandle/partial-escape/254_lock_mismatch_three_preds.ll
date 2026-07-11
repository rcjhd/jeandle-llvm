; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Three preds joining at the same merge with lock counts
; 0 / 1 / 2 on the same virtual object.
;
;   c0: no enter         → LC=0, no live enters.
;   c1: 1 enter          → LC=1, one live enter call.
;   c2: 2 enters         → LC=2, two live enter calls.
;
; Lock-cascade fires (counts disagree). Materialize at every pred:
;   c0: no live locks → just emit the new alloc invoke.
;   c1: un-elide its one enter (operand snapped to c1's NewInv).
;   c2: un-elide its two enters (operand snapped to c2's NewInv).
;
; After all three materialize, the retry sees AllMaterialized with
; per-pred-placeholder MaterializedValues; the lock-cascade fix detects the
; placeholders and builds a ptr addrspace(1) PHI of the three per-pred
; NewInvs, then RAUWs OrigAlloc → PHI for the post-merge sink user.
;
; User semantics preserved exactly: 0 enters on c0, 1 enter on c1, 2
; enters on c2, no synthesized enters anywhere.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_lock_mismatch_three_preds(i32 %sel) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock = alloca i64, align 8
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %switchblk unwind label %u
switchblk:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  switch i32 %sel, label %c0 [ i32 1, label %c1
                               i32 2, label %c2 ]
c0:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
c1:
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
              ptr addrspace(1) %o, ptr %lock)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
c2:
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
              ptr addrspace(1) %o, ptr %lock)
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
              ptr addrspace(1) %o, ptr %lock)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_lock_mismatch_three_preds
; Walk the IR in RPO order. c0 (zero locks) materializes with no enter:
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK-NOT: call hotspotcc void @jeandle.monitorenter_with_thin_lock
; c1 (one lock) materializes and un-elides one enter:
; CHECK: %[[MAT1:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT1]],
; CHECK-NOT: call hotspotcc void @jeandle.monitorenter_with_thin_lock
; c2 (two locks) materializes and un-elides two enters:
; CHECK: %[[MAT2:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT2]],
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT2]],
; No more enters or new-instance invokes after this.
; CHECK-NOT: call hotspotcc void @jeandle.monitorenter_with_thin_lock
; CHECK-NOT: invoke hotspotcc{{.*}}@jeandle.new_instance
; ptr addrspace(1) PHI at merge wires the three per-pred NewInvs, then
; sink consumes the PHI value.
; CHECK: %[[PHI:[A-Za-z0-9._]+]] = phi ptr addrspace(1)
; CHECK: call void @sink(ptr addrspace(1) %[[PHI]])

!java-method-compilation = !{}
