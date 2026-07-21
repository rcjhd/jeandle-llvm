; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p3:32:32"

@narrow_sink = global ptr addrspace(3) null

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(ptr addrspace(1))
declare hotspotcc ptr addrspace(1) @jeandle.decode_heap_oop(ptr addrspace(3))
declare void @escape(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define i32 @nested_narrow_scalar_replacement() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %outer = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
      ptr inttoptr (i64 1001 to ptr), i32 24)
      to label %outer.cont unwind label %unwind

outer.cont:
  %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
      ptr inttoptr (i64 1002 to ptr), i32 24)
      to label %body unwind label %unwind

body:
  %inner.field = getelementptr inbounds i8, ptr addrspace(1) %inner, i64 12
  store atomic i32 42, ptr addrspace(1) %inner.field unordered, align 4
  %encoded = call hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(
      ptr addrspace(1) %inner)
  %outer.field = getelementptr inbounds i8, ptr addrspace(1) %outer, i64 16
  store atomic ptr addrspace(3) %encoded,
      ptr addrspace(1) %outer.field unordered, align 4
  %narrow = load atomic ptr addrspace(3),
      ptr addrspace(1) %outer.field unordered, align 4
  %wide = call hotspotcc ptr addrspace(1) @jeandle.decode_heap_oop(
      ptr addrspace(3) %narrow)
  %loaded.field = getelementptr inbounds i8, ptr addrspace(1) %wide, i64 12
  %value = load atomic i32, ptr addrspace(1) %loaded.field unordered, align 4
  ret i32 %value

unwind:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define i32 @nested_narrow_scalar_replacement(
; CHECK-NOT: jeandle.new_instance
; CHECK-NOT: jeandle.encode_heap_oop
; CHECK-NOT: jeandle.decode_heap_oop
; CHECK-NOT: load
; CHECK-NOT: store
; CHECK: ret i32 42

define void @nested_narrow_materialization() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %outer = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
      ptr inttoptr (i64 2001 to ptr), i32 24)
      to label %outer.cont unwind label %unwind

outer.cont:
  %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
      ptr inttoptr (i64 2002 to ptr), i32 24)
      to label %body unwind label %unwind

body:
  %encoded = call hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(
      ptr addrspace(1) %inner)
  %outer.field = getelementptr inbounds i8, ptr addrspace(1) %outer, i64 16
  store atomic ptr addrspace(3) %encoded,
      ptr addrspace(1) %outer.field unordered, align 4
  call void @escape(ptr addrspace(1) %outer)
      [ "deopt"(i32 0, i32 0, i64 12, ptr addrspace(1) %outer) ]
  ret void

unwind:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @nested_narrow_materialization(
; CHECK-COUNT-2: @jeandle.new_instance(
; CHECK: [[ENCODED:%.*]] = call hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(ptr addrspace(1) {{%.*}})
; CHECK: store atomic ptr addrspace(3) [[ENCODED]], ptr addrspace(1) {{%.*}} unordered, align 4
; CHECK: call void @escape(ptr addrspace(1) {{%.*}})

define i32 @narrow_real_store_keeps_allocation() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
      ptr inttoptr (i64 3001 to ptr), i32 24)
      to label %body unwind label %unwind

body:
  %inner.field = getelementptr inbounds i8, ptr addrspace(1) %inner, i64 12
  store atomic i32 41, ptr addrspace(1) %inner.field unordered, align 4
  %value = load atomic i32, ptr addrspace(1) %inner.field unordered, align 4
  %encoded = call hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(
      ptr addrspace(1) %inner)
  store atomic ptr addrspace(3) %encoded,
      ptr @narrow_sink unordered, align 4
  ret i32 %value

unwind:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define i32 @narrow_real_store_keeps_allocation(
; CHECK: %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance
; CHECK-NOT: %inner.field =
; CHECK: %encoded = call hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(ptr addrspace(1) %inner)
; CHECK-NEXT: %pea.matslot = getelementptr inbounds i8, ptr addrspace(1) %inner, i64 12
; CHECK-NEXT: store atomic i32 41, ptr addrspace(1) %pea.matslot unordered, align 4
; CHECK-NOT: load atomic i32
; CHECK: store atomic ptr addrspace(3) %encoded, ptr @narrow_sink unordered, align 4
; CHECK: ret i32 41

!java-method-compilation = !{}
