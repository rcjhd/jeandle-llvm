; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; PEA A3 (Mixed-merge non-dominating alloc): the alloc-dominates-merge
; fast path. Alloc happens in %entry (which dominates the merge); the if-then
; arm escapes the object via @sink, the if-else arm leaves it virtual.
; mergeStates' mixed-state branch inherits Materialized at the merge using
; the OrigAlloc placeholder; the transform's safe-IP-hoisted materializeAt
; produces a single materialized invoke that dominates the merge, and RAUW
; redirects every IR use of the original alloc onto the new invoke.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @test_alloc_in_entry_mixed_merge(i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %n unwind label %u
n:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %then, label %else
then:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
else:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret ptr addrspace(1) %o
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; The transform must produce exactly one materialization invoke (the alloc
; dominates the merge so we don't need a per-pred PHI), and the @sink and
; %ret must consume it.
; CHECK-LABEL: define ptr addrspace(1) @test_alloc_in_entry_mixed_merge
; Per-arm materialize + materializedValuePhi (the collapse).
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: call void @sink(ptr addrspace(1) %{{[A-Za-z0-9._]+}})
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: = phi ptr addrspace(1)
; CHECK: ret ptr addrspace(1) %{{[A-Za-z0-9._]+}}

!java-method-compilation = !{}
