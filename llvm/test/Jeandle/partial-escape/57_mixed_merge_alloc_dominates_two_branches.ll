; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Alloc-dominates fast path with escapes on BOTH arms via DIFFERENT
; sink calls. The mergeStates "AllMaterialized" branch fires (every incoming
; has the object Materialized). Because every incoming's MaterializedValue
; resolves to the same OrigAlloc placeholder, no ptr addrspace(1) PHI is
; needed at the merge; RAUW makes both materializations point at the live
; invoke. (This exercises the AllMaterialized fast path alongside the
; alloc-dominates invariant.)

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink1(ptr addrspace(1))
declare void @sink2(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @test_both_arms_escape(i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %n unwind label %u
n:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %left, label %right
left:
  call void @sink1(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
right:
  call void @sink2(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret ptr addrspace(1) %o
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define ptr addrspace(1) @test_both_arms_escape
; Each arm escapes and materializes at its own sink; the merge builds a PHI.
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: call void @sink1
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: call void @sink2
; CHECK: = phi ptr addrspace(1)
; CHECK: ret ptr addrspace(1)

!java-method-compilation = !{}
