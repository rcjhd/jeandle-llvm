; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; PEA A3 (Mixed-merge alloc-dominates fast path) with field stores in the
; if-then arm. The alloc is in %entry; the if-then arm writes a field then
; escapes; the if-else arm leaves the object virtual. The materialization
; replays the field write and the merge inherits Materialized via OrigAlloc;
; RAUW threads the live invoke into the return.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @test_mixed_merge_with_field_writes(i1 %c, i32 %v)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %n unwind label %u
n:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %then, label %else
then:
  %slot = getelementptr inbounds i8, ptr addrspace(1) %o, i64 8
  store atomic i32 %v, ptr addrspace(1) %slot unordered, align 4
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

; Single materialization (alloc dominates merge). The field write at offset 8
; is replayed at the top of the materialization continuation block, then the
; sink and return consume the new invoke.
; CHECK-LABEL: define ptr addrspace(1) @test_mixed_merge_with_field_writes
; The escape arm materializes (replaying the field write) and sinks; the virtual
; arm materializes at its pred-end; the return consumes the merge PHI.
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: store atomic i32 %v
; CHECK: call void @sink(ptr addrspace(1) %{{[A-Za-z0-9._]+}})
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: = phi ptr addrspace(1)
; CHECK: ret ptr addrspace(1) %{{[A-Za-z0-9._]+}}

!java-method-compilation = !{}
