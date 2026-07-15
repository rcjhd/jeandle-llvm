; RUN: opt -S -verify-each -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A replacement allocation's own deopt state uses a lazy snapshot because the
; allocation may fail before producing a real object. Once that allocation
; succeeds, later dominated deopt points must reference the real definition.
; Reusing the pre-materialization snapshot there reconstructs a stale object.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare hotspotcc void @jeandle.safepoint_poll()
declare i32 @__gxx_personality_v0(...)

define void @deopt_after_materialization_uses_real_object() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
           ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %n unwind label %u
n:
  %field = getelementptr inbounds i8, ptr addrspace(1) %o, i64 12
  store i32 42, ptr addrspace(1) %field
  call void @sink(ptr addrspace(1) %o)
      [ "deopt"(i32 1, i32 1, i64 12, ptr addrspace(1) %o) ]
  call hotspotcc void @jeandle.safepoint_poll()
      [ "deopt"(i32 2, i32 2, i64 12, ptr addrspace(1) %o) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @deopt_after_materialization_uses_real_object
; CHECK: [[MAT:%.*]] = invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance
; CHECK-SAME: [ "deopt"(i32 1, i32 1, i64 4295491596,
; CHECK: call void @sink(ptr addrspace(1) [[MAT]])
; CHECK: call hotspotcc void @jeandle.safepoint_poll()
; CHECK-SAME: [ "deopt"(i32 2, i32 2, i64 12, ptr addrspace(1) [[MAT]]) ]
; CHECK-NOT: i64 4295491596
; CHECK: ret void

!java-method-compilation = !{}
