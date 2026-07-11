; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Preserve operand bundles when materializing an escaping virtual object. The
; escape point itself carries both a non-"deopt" bundle and a "deopt" bundle,
; so PEA materializes at that point and the materialization invoke copies both.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare void @cfg_target_fn()
declare i32 @__gxx_personality_v0(...)

define void @multi_bundle() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %n unwind label %u
n:
  call void @sink(ptr addrspace(1) %o)
       [ "deopt"(i32 7, i32 7), "cfguardtarget"(ptr @cfg_target_fn) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; Both bundles survive on the materialisation invoke.
; CHECK-LABEL: define void @multi_bundle
; CHECK: %pea.mat = invoke {{.*}}@jeandle.new_instance
; CHECK-SAME: [ "cfguardtarget"(ptr @cfg_target_fn), "deopt"(i32 7, i32 7) ]
; CHECK: call void @sink(ptr addrspace(1) %pea.mat) [ "deopt"(i32 7, i32 7), "cfguardtarget"(ptr @cfg_target_fn) ]

!java-method-compilation = !{}
