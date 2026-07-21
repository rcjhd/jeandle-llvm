; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A p3 address space narrower than the Java heap address space no longer disables
; PEA for the whole function. Even a compressed-oop module must eliminate an
; otherwise unused allocation.

target datalayout = "e-p:64:64:64-p1:64:64:64-p3:32:32:32"

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare i32 @__gxx_personality_v0(...)

define void @test_unused_alloc() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %obj = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
         to label %normal unwind label %unwind

normal:
  ret void

unwind:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_unused_alloc()
; CHECK-NOT: @jeandle.new_instance
; CHECK: ret void

!java-method-compilation = !{}
