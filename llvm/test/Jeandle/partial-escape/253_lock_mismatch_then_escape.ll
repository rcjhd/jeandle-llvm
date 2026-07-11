; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Lock in one branch, escape (sink) at the merge after
; the lock-cascade has materialized at both preds.
;
; Sequence:
;   then: monitorenter(o) → LC[o]=1, enter is folded.
;   else: no lock → LC[o]=0.
;   merge: counts disagree → lock-cascade fires. Both preds materialize:
;     * then-pred: un-elides the enter (its receiver is rewritten to the
;       then-pred's NewInv via the PH-tagged ReplaceInput).
;     * else-pred: no live locks, just emits the alloc invoke.
;   merge retry: AllMaterialized branch builds a ptr addrspace(1) PHI of
;     the two per-pred NewInvs.
;   sink(o) at the merge uses the PHI as its receiver.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_lock_mismatch_then_escape(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock = alloca i64, align 8
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %branch unwind label %u
branch:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %then, label %else
then:
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
              ptr addrspace(1) %o, ptr %lock)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
else:
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

; CHECK-LABEL: define void @test_lock_mismatch_then_escape
; Two per-pred materializations and one surviving un-elided enter.
; CHECK: %[[MAT1:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT1]],
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; merge: ptr addrspace(1) PHI threads both per-pred NewInvs.
; CHECK: %[[PHI:[A-Za-z0-9._]+]] = phi ptr addrspace(1)
; CHECK: call void @sink(ptr addrspace(1) %[[PHI]])
; No second enter (else side had none).
; CHECK-NOT: call hotspotcc void @jeandle.monitorenter_with_thin_lock

!java-method-compilation = !{}
