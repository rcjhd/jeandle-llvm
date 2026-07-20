; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A virtual allocated before a loop must not be force-materialized merely
; because a throwing loop instruction exits to a non-trivial cleanup. The
; cleanup does not observe %o, and the regular loop fixed point can carry its
; scalar field state across the backedge.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare i32 @may_throw()
declare void @observe(i32)
declare i32 @__gxx_personality_v0(...)

define void @test_preloop_virtual_nonobserving_cleanup(i1 %again) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
           ptr inttoptr (i64 12345 to ptr), i32 32)
       to label %preheader unwind label %oom

preheader:
  %field = getelementptr inbounds i8, ptr addrspace(1) %o, i64 8
  store atomic i32 0, ptr addrspace(1) %field unordered, align 4
  br label %loop

loop:
  %i = phi i32 [ 0, %preheader ], [ %next, %back ]
  %value = load atomic i32, ptr addrspace(1) %field unordered, align 4
  call void @observe(i32 %value)
  %delta = invoke i32 @may_throw()
           to label %back unwind label %cleanup

back:
  %next = add i32 %i, %delta
  store atomic i32 %next, ptr addrspace(1) %field unordered, align 4
  br i1 %again, label %loop, label %exit

exit:
  ret void

oom:
  %oom.lp = landingpad i64 cleanup
  resume i64 %oom.lp

cleanup:
  %lp = landingpad i64 cleanup
  call void @observe(i32 -1)
  resume i64 %lp
}

; CHECK-LABEL: define void @test_preloop_virtual_nonobserving_cleanup
; CHECK-NOT: @jeandle.new_instance
; CHECK-NOT: getelementptr
; CHECK-NOT: load atomic
; CHECK-NOT: store atomic
; CHECK: ret void

!java-method-compilation = !{}
