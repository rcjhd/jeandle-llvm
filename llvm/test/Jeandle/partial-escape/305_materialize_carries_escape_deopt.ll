; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; DeoptBundleSource prefers the escape-point CallBase when it carries a
; "deopt" bundle. The materialization invoke must inherit that bundle so the
; allocation inserted at the escape point has the same deoptimization state;
; the original sink also keeps its bundle and its object operand is resolved to
; the new materialized pointer.
declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @escape_with_bundle() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  ; No "deopt" bundle on the allocation itself.
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %n unwind label %u
n:
  ; Rich "deopt" bundle on the sink — this is what must be carried over.
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 42, i32 42) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @escape_with_bundle
; CHECK: %pea.mat = invoke {{.*}}@jeandle.new_instance(ptr {{.*}}, i32 16) [ "deopt"(i32 42, i32 42) ]
; CHECK-NEXT: to label %{{.*}} unwind label %{{.*}}
; CHECK: call void @sink(ptr addrspace(1) %pea.mat) [ "deopt"(i32 42, i32 42) ]

!java-method-compilation = !{}
