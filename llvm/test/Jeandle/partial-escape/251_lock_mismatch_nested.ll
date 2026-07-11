; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Nested lock-count mismatch. The then-pred holds two
; live monitorenters on o (count=2), the else-pred holds one (count=1).
; The lock-cascade materializes at both preds with each pred's OWN live lock stack —
; the then-pred un-elides two enters, the else-pred un-elides one. The
; user's IR semantics are preserved exactly: two enters on then's path,
; one on else's path. No synthesized enters appear.
;
; Each pred is fed its OWN locks list — no extra enters are added on the
; lower-count side.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_lock_mismatch_nested(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
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
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
              ptr addrspace(1) %o, ptr %lock)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
else:
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
              ptr addrspace(1) %o, ptr %lock)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_lock_mismatch_nested
; Per-pred materializations: one InvokeInst per pred (2 total) plus the
; un-elided enters interleaved. Layout in output (RPO-dependent):
;   then-MatCont: invoke (MAT1), 2 enters on MAT1
;   else-MatCont: invoke (MAT2), 1 enter on MAT2
; (or with `then`/`else` swapped — the structure is symmetric: 2 enters
; on whichever pred had count=2, 1 enter on the other.)
; CHECK: %[[MAT1:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT1]],
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT1]],
; CHECK: %[[MAT2:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT2]],
; No further enter (else-side had only one).
; CHECK-NOT: call hotspotcc void @jeandle.monitorenter_with_thin_lock

!java-method-compilation = !{}
