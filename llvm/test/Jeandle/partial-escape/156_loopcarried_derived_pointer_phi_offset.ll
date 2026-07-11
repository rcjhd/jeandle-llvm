; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s
;
; Regression test for the offset-loss miscompile in Case B/C. An all-derived-
; same-object PHI %p = phi [%sf1,%a1],[%sf2,%a2] (both gep %X,8) is used for a
; field LOAD. Before the fix, Case B aliased %p to the object (offset 0), so the
; load folded to field[0]'s default (0) instead of field[8] (the stored 111/222)
; — i.e. the output was `call void @sink_i32(i32 0)`, a miscompile. With the
; !AnyDerived gate, the derived PHI routes to Case A: the object is materialized
; per arm and the load stays a real load over the re-derived offset-8 pointer.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink_i32(i32)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_156_derived_phi_offset(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %X = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 5555 to ptr), i32 32)
          to label %cont unwind label %u
cont:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %a1, label %a2
a1:
  %sf1 = getelementptr inbounds i8, ptr addrspace(1) %X, i64 8
  store i32 111, ptr addrspace(1) %sf1
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %m
a2:
  %sf2 = getelementptr inbounds i8, ptr addrspace(1) %X, i64 8
  store i32 222, ptr addrspace(1) %sf2
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %m
m:
  %p = phi ptr addrspace(1) [ %sf1, %a1 ], [ %sf2, %a2 ]
  %v = load i32, ptr addrspace(1) %p
  call void @sink_i32(i32 %v)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; The load must NOT fold to field[0]'s default 0; it stays a real load over a
; re-derived offset-8 pointer (field[8] holds 111 / 222). The re-derived GEPs
; sit in the per-arm materialization blocks, which precede the merge load.
; CHECK-LABEL: define void @test_156_derived_phi_offset
; CHECK: getelementptr inbounds i8, ptr addrspace(1) %pea.mat
; CHECK: load i32, ptr addrspace(1) %p
; CHECK: call void @sink_i32(i32 %v)
; CHECK-NOT: call void @sink_i32(i32 0)
; CHECK-NOT: poison

!java-method-compilation = !{}
