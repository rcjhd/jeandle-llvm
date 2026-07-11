; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A cyclic field graph with a non-peer SCALAR field: A.x = 42 (offset 0, i64)
; alongside A.f = B (offset 8) and B.g = A (offset 8, back edge). Returning A
; escapes the cascade. The tail replays all three stores: the scalar store uses
; the constant 42; the two reference stores use real NewInvs (the back edge B.g=A
; resolves). Exercises FieldValue::isScalar and isMaterializedRef in one cascade.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @cyclic_with_scalar_field()
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %a = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 11111 to ptr), i32 16) to label %na unwind label %u1
na:
  %b = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16) to label %nb unwind label %u2
nb:
  %sx = getelementptr inbounds i8, ptr addrspace(1) %a, i64 0
  store atomic i64 42, ptr addrspace(1) %sx unordered, align 8
  %sa = getelementptr inbounds i8, ptr addrspace(1) %a, i64 8
  store atomic ptr addrspace(1) %b, ptr addrspace(1) %sa unordered, align 8
  %sb = getelementptr inbounds i8, ptr addrspace(1) %b, i64 8
  store atomic ptr addrspace(1) %a, ptr addrspace(1) %sb unordered, align 8
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret ptr addrspace(1) %a
u1:
  %lp1 = landingpad i64 cleanup
  resume i64 %lp1
u2:
  %lp2 = landingpad i64 cleanup
  resume i64 %lp2
}

!java-method-compilation = !{}

; CHECK-LABEL: define ptr addrspace(1) @cyclic_with_scalar_field
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 11111 to ptr)
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 22222 to ptr)
; The scalar field (A.x = 42) is replayed as a constant store, and both
; reference stores use real NewInvs (the back edge B.g = A resolves). Stores are
; CHECK-DAG because their in-tail order is not significant.
; CHECK-DAG: store atomic i64 42, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK-DAG: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK-DAG: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK-NOT: poison
; CHECK: ret ptr addrspace(1) %{{.*}}
