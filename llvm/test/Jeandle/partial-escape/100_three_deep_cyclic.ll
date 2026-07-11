; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Edge case: three-deep cyclic nested virtuals. A.x=B, B.x=C, C.x=A.
; Only A escapes (returned). Transitive materialization should produce all
; three allocs without infinite recursion (idempotency in materializeAt's
; Materialized set guards the cycle).

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @test_three_deep_cyclic() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %a = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 11111 to ptr), i32 16)
       to label %nA unwind label %u1
nA:
  %b = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 22222 to ptr), i32 16)
       to label %nB unwind label %u2
nB:
  %c = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 33333 to ptr), i32 16)
       to label %nC unwind label %u3
nC:
  %slotA = getelementptr inbounds i8, ptr addrspace(1) %a, i64 8
  store atomic ptr addrspace(1) %b, ptr addrspace(1) %slotA unordered, align 8
  %slotB = getelementptr inbounds i8, ptr addrspace(1) %b, i64 8
  store atomic ptr addrspace(1) %c, ptr addrspace(1) %slotB unordered, align 8
  %slotC = getelementptr inbounds i8, ptr addrspace(1) %c, i64 8
  store atomic ptr addrspace(1) %a, ptr addrspace(1) %slotC unordered, align 8
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret ptr addrspace(1) %a
u1:
  %lp1 = landingpad i64 cleanup
  resume i64 %lp1
u2:
  %lp2 = landingpad i64 cleanup
  resume i64 %lp2
u3:
  %lp3 = landingpad i64 cleanup
  resume i64 %lp3
}

; All three klasses materialize. Cycle prevention prevents infinite recursion.
; CHECK-LABEL: define ptr addrspace(1) @test_three_deep_cyclic
; CHECK-DAG: invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 11111 to ptr), i32 16)
; CHECK-DAG: invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16)
; CHECK-DAG: invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 33333 to ptr), i32 16)
; All three replayed stores use real NewInvs — the cycle's back edge C.x = A (A
; materialized last in the cascade) resolves instead of lowering to poison.
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK-NOT: poison
; CHECK: ret ptr addrspace(1) %{{.*}}

!java-method-compilation = !{}
