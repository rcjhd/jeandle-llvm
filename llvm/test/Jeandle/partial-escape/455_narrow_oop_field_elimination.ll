; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A narrow oop occupies four bytes in the field model. When the object is never
; observed, PEA can eliminate both the allocation and the narrow field store.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare i32 @__gxx_personality_v0(...)

define void @narrow_oop_field(ptr addrspace(3) %v) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
         to label %n unwind label %u
n:
  %f = getelementptr inbounds i8, ptr addrspace(1) %o, i64 8
  store atomic ptr addrspace(3) %v, ptr addrspace(1) %f unordered, align 4
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @narrow_oop_field(
; CHECK-NOT: @jeandle.new_instance
; CHECK-NOT: store
; CHECK: ret void

!java-method-compilation = !{}
