; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; When a canonical Jeandle deopt bundle appears on an escape sink, PEA
; materializes the object for the normal operand and rewrites deopt object
; slots through lazy-object records so the materialization invoke does not
; carry an OrigAlloc self-reference.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @deopt_bundle_scrubbed() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %n unwind label %u
n:
  ; Sink call escape; its deopt object slot references the VO.
  call void @sink(ptr addrspace(1) %o)
       [ "deopt"(i32 99, i32 99, i64 12, ptr addrspace(1) %o) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @deopt_bundle_scrubbed
;
; The materialisation invoke must not carry an OrigAlloc self-reference.
; CHECK: %pea.mat = invoke {{.*}}@jeandle.new_instance(ptr {{.*}}, i32 16)
; CHECK-SAME: [ "deopt"(i32 99, i32 99,
; CHECK-NOT: ptr addrspace(1) %o
; CHECK-NEXT: to label %{{.*}} unwind label %{{.*}}
;
; The surviving sink uses the materialized object on the normal path.
; CHECK: call void @sink(ptr addrspace(1) %pea.mat)

!java-method-compilation = !{}
