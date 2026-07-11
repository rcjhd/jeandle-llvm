; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A virtual object referenced only by a deopt bundle must stay virtual on the
; normal path. PEA rewrites the deopt state to a LazyObjectType lazy-object
; record instead of materializing the object at the safepoint.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.safepoint_poll()
declare i32 @__gxx_personality_v0(...)

define void @test_deopt_lazy_object_preserves_narrow_marker(ptr addrspace(3) %narrow) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 66666 to ptr), i32 16)
       to label %n unwind label %u
n:
  %fp = getelementptr inbounds i8, ptr addrspace(1) %o, i64 12
  store i32 9, ptr addrspace(1) %fp
  call hotspotcc void @jeandle.safepoint_poll()
      [ "deopt"(i32 0, i32 0, i64 458768, ptr addrspace(3) %narrow,
                  i64 12, ptr addrspace(1) %o) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_deopt_lazy_object_preserves_narrow_marker
; CHECK-NOT: jeandle.new_instance
; CHECK-NOT: store i32 9
; CHECK: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0, i64 458768, ptr addrspace(3) %narrow, i64 4295491596, i64 66666, i64 1, i64 12, i64 589834, i32 9, i64 12, i64 1) ]
; CHECK: ret void
define void @test_deopt_lazy_object() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %n unwind label %u
n:
  %fp = getelementptr inbounds i8, ptr addrspace(1) %o, i64 12
  store i32 42, ptr addrspace(1) %fp
  call hotspotcc void @jeandle.safepoint_poll()
      [ "deopt"(i32 0, i32 0, i64 12, ptr addrspace(1) %o) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

define void @test_deopt_nested_lazy_object() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %outer = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 11111 to ptr), i32 16)
       to label %outer.n unwind label %u
outer.n:
  %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 22222 to ptr), i32 16)
       to label %n unwind label %u
n:
  %fp = getelementptr inbounds i8, ptr addrspace(1) %outer, i64 12
  store ptr addrspace(1) %inner, ptr addrspace(1) %fp
  call hotspotcc void @jeandle.safepoint_poll()
      [ "deopt"(i32 0, i32 0, i64 12, ptr addrspace(1) %outer) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}


define void @test_deopt_cyclic_lazy_objects() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %a = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 33333 to ptr), i32 16)
       to label %a.n unwind label %u

a.n:
  %b = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 44444 to ptr), i32 16)
       to label %n unwind label %u
n:
  %afp = getelementptr inbounds i8, ptr addrspace(1) %a, i64 12
  store ptr addrspace(1) %b, ptr addrspace(1) %afp
  %bfp = getelementptr inbounds i8, ptr addrspace(1) %b, i64 12
  store ptr addrspace(1) %a, ptr addrspace(1) %bfp
  call hotspotcc void @jeandle.safepoint_poll()
      [ "deopt"(i32 0, i32 0, i64 12, ptr addrspace(1) %a) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}


define void @test_deopt_lazy_object_method_scope_header() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 55555 to ptr), i32 16)
       to label %n unwind label %u
n:
  %fp = getelementptr inbounds i8, ptr addrspace(1) %o, i64 12
  store i32 7, ptr addrspace(1) %fp
  call hotspotcc void @jeandle.safepoint_poll()
      [ "deopt"(i64 393233, i64 777, i32 0, i32 0, i64 12, ptr addrspace(1) %o) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_deopt_lazy_object
; CHECK-NOT: jeandle.new_instance
; CHECK-NOT: store i32 42
; CHECK: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0, i64 4295491596, i64 12345, i64 1, i64 12, i64 589834, i32 42, i64 12, i64 1) ]
; CHECK: ret void

; CHECK-LABEL: define void @test_deopt_nested_lazy_object
; CHECK-NOT: jeandle.new_instance
; CHECK-NOT: store ptr addrspace(1)
; CHECK: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0, i64 4295491596, i64 11111, i64 1, i64 12, i64 589836, i64 2, i64 8590458892, i64 22222, i64 0, i64 12, i64 1) ]
; CHECK: ret void

; CHECK-LABEL: define void @test_deopt_cyclic_lazy_objects
; CHECK-NOT: jeandle.new_instance
; CHECK-NOT: store ptr addrspace(1)
; CHECK: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0, i64 4295491596, i64 33333, i64 1, i64 12, i64 589836, i64 2, i64 8590458892, i64 44444, i64 1, i64 12, i64 589836, i64 1, i64 12, i64 1) ]
; CHECK: ret void

; CHECK-LABEL: define void @test_deopt_lazy_object_method_scope_header
; CHECK-NOT: jeandle.new_instance
; CHECK-NOT: store i32 7
; CHECK: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i64 393233, i64 777, i32 0, i32 0, i64 4295491596, i64 55555, i64 1, i64 12, i64 589834, i32 7, i64 12, i64 1) ]
; CHECK: ret void

!java-method-compilation = !{}
