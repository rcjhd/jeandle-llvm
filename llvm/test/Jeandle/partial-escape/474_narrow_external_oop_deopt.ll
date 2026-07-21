; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A virtual object's compressed reference field stores the addrspace(3)
; encoding of a real external oop. The lazy descriptor must carry the original
; wide addrspace(1) oop while remembering that normal field replay/load uses
; the narrow declared type. Otherwise descriptor planning rejects the field
; and keeps the allocation real at every deopt point.

target datalayout = "e-p:64:64-p1:64:64-p3:32:32"

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(ptr addrspace(1))
declare void @sink(i32)
declare i32 @__gxx_personality_v0(...)

define void @narrow_field_holds_external_oop(
    i32 %x, ptr addrspace(1) %ext)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %obj = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 100 to ptr), i32 24)
         to label %body unwind label %u

body:
  %encoded = call hotspotcc ptr addrspace(3)
      @jeandle.encode_heap_oop(ptr addrspace(1) %ext)
  %int.field = getelementptr inbounds i8, ptr addrspace(1) %obj, i64 8
  %ref.field = getelementptr inbounds i8, ptr addrspace(1) %obj, i64 16
  store atomic i32 %x, ptr addrspace(1) %int.field unordered, align 4
  store atomic ptr addrspace(3) %encoded,
      ptr addrspace(1) %ref.field unordered, align 4
  call void @sink(i32 %x)
       [ "deopt"(i32 99, i32 99, i64 12, ptr addrspace(1) %obj) ]
  ret void

u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @narrow_field_holds_external_oop
; CHECK-NOT: jeandle.new_instance
; CHECK: call void @sink(i32 %x)
; CHECK-SAME: [ "deopt"(
; descriptor obj (vo_id=0, ScalarValueType, T_OBJECT)
; CHECK-SAME: i64 262156, i64 100, i32 2,
; int field at offset 8
; CHECK-SAME: i64 34359738378, i32 %x,
; compressed field at offset 16 is represented by the live wide oop
; CHECK-SAME: i64 68719476748, ptr addrspace(1) %ext,
; local slot becomes VORef(vo_id=0)
; CHECK-SAME: i64 524300, i32 0) ]
; CHECK-NOT: addrspace(1) %obj

!java-method-compilation = !{}
