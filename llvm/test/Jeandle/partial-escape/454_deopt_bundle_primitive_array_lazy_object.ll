; RUN: opt -S -verify-each -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A primitive array referenced by a deopt bundle is encoded as a T_ARRAY
; lazy-object record. The record carries the logical array length separately
; from its sparse explicit element list.

declare hotspotcc ptr addrspace(1) @jeandle.new_array(ptr, i32, i32, i32, i32)
declare hotspotcc void @jeandle.safepoint_poll()
declare i32 @__gxx_personality_v0(...)

define void @array_in_deopt_bundle_keeps_original() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %arr = invoke hotspotcc ptr addrspace(1) @jeandle.new_array(
             ptr inttoptr (i64 12345 to ptr), i32 1, i32 24, i32 16, i32 1024)
         to label %n unwind label %u
n:
  %elem = getelementptr inbounds i8, ptr addrspace(1) %arr, i64 16
  store i16 65, ptr addrspace(1) %elem, align 2
  call hotspotcc void @jeandle.safepoint_poll()
      [ "deopt"(i32 0, i32 0, i64 12, ptr addrspace(1) %arr) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @array_in_deopt_bundle_keeps_original
; CHECK-NOT: @jeandle.new_array
; CHECK-NOT: store i16
; CHECK: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0, i64 4295491597, i64 12345, i64 1, i64 1, i64 16, i64 589834, i16 65, i64 12, i64 1) ]

!java-method-compilation = !{}
