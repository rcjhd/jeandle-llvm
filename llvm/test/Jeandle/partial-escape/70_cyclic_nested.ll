; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; (§8.1.13): Cyclic nested virtuals. Two virtuals A and B form a cycle —
; A.f = B and B.g = A. Returning A escapes A; transitive materialization must
; materialize B (because A.f references B) and then, when materializing B,
; must NOT recurse back into A indefinitely because A is already in the
; Materialized set (idempotency in materializeAt). Both materialization
; invokes appear; A's MatCont stores the new B at offset 8; B's MatCont
; stores the new A at offset 8.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @test_cyclic_nested() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %a = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 11111 to ptr), i32 16)
       to label %nA unwind label %u1
nA:
  %b = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 22222 to ptr), i32 16)
       to label %nB unwind label %u2
nB:
  %slotA = getelementptr inbounds i8, ptr addrspace(1) %a, i64 8
  store atomic ptr addrspace(1) %b, ptr addrspace(1) %slotA unordered, align 8
  %slotB = getelementptr inbounds i8, ptr addrspace(1) %b, i64 8
  store atomic ptr addrspace(1) %a, ptr addrspace(1) %slotB unordered, align 8
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret ptr addrspace(1) %a
u1:
  %lp1 = landingpad i64 cleanup
  resume i64 %lp1
u2:
  %lp2 = landingpad i64 cleanup
  resume i64 %lp2
}

; Cycle prevention works: both A (klass 11111) and B (klass 22222) materialize.
; CHECK-LABEL: define ptr addrspace(1) @test_cyclic_nested
; CHECK-DAG: invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 11111 to ptr), i32 16)
; CHECK-DAG: invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16)
; Both replayed stores use real NewInvs — the back edge B.g = A (A materialized
; after B in the cascade) resolves instead of lowering to poison.
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK-NOT: poison
; CHECK: ret ptr addrspace(1) %{{.*}}

!java-method-compilation = !{}
