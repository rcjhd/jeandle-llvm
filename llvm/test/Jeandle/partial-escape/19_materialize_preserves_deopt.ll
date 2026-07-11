; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; If the escape point has no deopt bundle and there is no safe prior deopt
; anchor in the same block, PEA must not materialize at the escape using the
; original allocation's stale deopt state. Keep the original allocation real
; instead.
declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @test_mat_deopt() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       [ "deopt"(i32 42, i32 42) ]
       to label %n unwind label %u
n:
  call void @sink(ptr addrspace(1) %o)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_mat_deopt
; CHECK: %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16) [ "deopt"(i32 42, i32 42) ]
; CHECK-NOT: %pea.mat = invoke
; CHECK: call void @sink(ptr addrspace(1) %o)

!java-method-compilation = !{}
