; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Lock-count mismatch cascade at merge.
;
; entry: alloc o (virtual). branch on %c.
; then: monitorenter(o) — folded, LockCount[o]=1 at then exit.
; else: nothing — LockCount[o]=0 at else exit.
; merge: counts disagree → lock-cascade fires.
;
; Expected outcome:
; each pred materializes the VO with its OWN lock list. The then-pred
; materialize cascades through the un-elide so the original monitorenter
; survives in IR with its first operand snapped to the then-pred's new
; materialized invoke. The else-pred materialize has no live locks and
; just emits a fresh allocation invoke. The original entry alloc is
; eliminated; downstream uses of %o resolve through the per-pred MatConts.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_lock_mismatch_one_arm_locked(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
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
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_lock_mismatch_one_arm_locked
; The original entry alloc is eliminated; the lock-cascade materializes a fresh
; allocation invoke at each pred. The first one we see comes from one
; pred; the un-elided monitorenter in that pred's MatCont references its
; per-pred NewInv (with the SSA-correct receiver). The second invoke
; comes from the other pred. We don't rely on a specific RPO ordering of
; then/else — we only need to see (a) a materialized invoke, (b) a
; surviving monitorenter on a per-pred NewInv, and (c) a second
; materialized invoke for the other pred.
; CHECK: %[[MAT1:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT1]],
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)

!java-method-compilation = !{}
