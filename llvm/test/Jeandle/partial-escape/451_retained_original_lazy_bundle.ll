; RUN: opt -S -verify-each -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A retained original allocation still reaches code generation, so its deopt
; bundle must be fully rebuilt before virtual allocations are erased.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.safepoint_poll()
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @retained_original_rebuilds_virtual_input() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
                ptr inttoptr (i64 30101 to ptr), i32 16)
           to label %inner.cont unwind label %u
inner.cont:
  %inner.field = getelementptr inbounds i8, ptr addrspace(1) %inner, i64 12
  store i32 41, ptr addrspace(1) %inner.field
  %outer = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
                ptr inttoptr (i64 30102 to ptr), i32 16)
           [ "deopt"(i32 7, i32 7, i64 12, ptr addrspace(1) %inner) ]
           to label %outer.cont unwind label %u
outer.cont:
  %outer.field = getelementptr inbounds i8, ptr addrspace(1) %outer, i64 12
  store i32 42, ptr addrspace(1) %outer.field
  call void @sink(ptr addrspace(1) %outer)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @retained_original_rebuilds_virtual_input
; CHECK-NOT: @jeandle.new_instance(ptr inttoptr (i64 30101 to ptr)
; CHECK: [[OUTER:%.*]] = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 30102 to ptr), i32 16) [ "deopt"(i32 7, i32 7, i64 4295491596, i64 30101, i64 1, i64 12, i64 589834, i32 41, i64 12, i64 1) ]
; CHECK-NOT: poison
; CHECK-NOT: %inner.field = getelementptr
; CHECK-NOT: %outer.field = getelementptr
; CHECK: %pea.matslot = getelementptr inbounds i8, ptr addrspace(1) [[OUTER]], i64 12
; CHECK-NEXT: store atomic i32 42, ptr addrspace(1) %pea.matslot unordered, align 4
; CHECK: call void @sink(ptr addrspace(1) [[OUTER]])

define void @retained_original_normalizes_poison() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 30103 to ptr), i32 16)
       [ "deopt"(i32 8, i32 8, i64 12, ptr addrspace(1) poison) ]
       to label %n unwind label %u
n:
  %field = getelementptr inbounds i8, ptr addrspace(1) %o, i64 12
  store i32 43, ptr addrspace(1) %field
  call void @sink(ptr addrspace(1) %o)
  %v = call hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 30104 to ptr), i32 16)
  %v.field = getelementptr inbounds i8, ptr addrspace(1) %v, i64 12
  store i32 44, ptr addrspace(1) %v.field
  call hotspotcc void @jeandle.safepoint_poll()
       [ "deopt"(i32 9, i32 9, i64 12, ptr addrspace(1) %v) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @retained_original_normalizes_poison
; CHECK: [[O:%.*]] = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 30103 to ptr), i32 16) [ "deopt"(i32 8, i32 8, i64 12, ptr addrspace(1) null) ]
; CHECK-NOT: poison
; CHECK-NOT: %field = getelementptr
; CHECK: %pea.matslot = getelementptr inbounds i8, ptr addrspace(1) [[O]], i64 12
; CHECK-NEXT: store atomic i32 43, ptr addrspace(1) %pea.matslot unordered, align 4
; CHECK: call void @sink(ptr addrspace(1) [[O]])

!java-method-compilation = !{}
