; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; MergeProcessor / materializeAndBuildPhi lock-count mismatch (1 vs 2).
;
; Both arms lock the same virtual object but with different counts
; (then=1 enter, else=2 enters). The counts disagree at the merge, so the
; per-VO disposition routes to the materialize+materializedValuePhi path:
; each pred is materialized with its OWN lock list, then a ptr addrspace(1)
; PHI selects between the per-pred NewInvs at the merge and OrigAlloc is
; RAUW'd onto the PHI for the post-merge sink user.
;
; This is a regression anchor for the collapsed single-pass
; materializeAndBuildPhi (the pre-refactor code handled lock mismatch as a
; two-pass materialize-then-PHI across fixpoint iterations). 250 (0 vs 1)
; and 254 (0/1/2 over three preds) cover adjacent counts; this covers the
; 1-vs-2 two-pred shape.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_lockmismatch_1v2(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
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

; The original entry alloc is eliminated; each pred materializes a fresh
; allocation invoke carrying its own lock list (then=1 enter, else=2 enters),
; and a ptr addrspace(1) PHI at the merge wires the two NewInvs for the sink.
; We assume RPO visits `then` (1 enter) before `else` (2 enters).
; CHECK-LABEL: define void @test_lockmismatch_1v2
; CHECK: %[[MAT1:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT1]],
; CHECK: %[[MAT2:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT2]],
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %[[MAT2]],
; CHECK: %[[PHI:[A-Za-z0-9._]+]] = phi ptr addrspace(1)
; CHECK: call void @sink(ptr addrspace(1) %[[PHI]])

!java-method-compilation = !{}
