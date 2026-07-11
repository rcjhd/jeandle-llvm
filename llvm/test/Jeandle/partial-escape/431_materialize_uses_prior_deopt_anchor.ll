; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; The escape point itself has no deopt bundle, but a safe same-block deopt
; anchor immediately precedes it. Materialization is moved to that anchor and
; the pea.mat invoke uses the anchor's deopt bundle, not the original
; allocation's older state.
declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.safepoint_poll()
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @materialize_at_prior_deopt_anchor() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       [ "deopt"(i32 1, i32 1) ]
       to label %n unwind label %u
n:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 2, i32 2) ]
  call void @sink(ptr addrspace(1) %o)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @materialize_at_prior_deopt_anchor
; CHECK: n:
; CHECK-NEXT: %pea.mat = invoke hotspotcc nonnull "java-klass"="12345" "java-klass-exact" ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16) [ "deopt"(i32 2, i32 2) ]
; CHECK-NEXT: to label %mat.cont unwind label %u
; CHECK: mat.cont:
; CHECK-NEXT: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 2, i32 2) ]
; CHECK-NEXT: call void @sink(ptr addrspace(1) %pea.mat)

!java-method-compilation = !{}
