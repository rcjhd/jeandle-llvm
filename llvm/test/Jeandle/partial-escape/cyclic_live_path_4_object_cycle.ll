; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Four-object cycle: A.f = B, B.g = C, C.h = D, D.i = A. Returning A escapes the
; whole cascade (4 objects, one shared escape point). The back edge D.i = A (A
; materialized last) resolves to A's NewInv because every replayed store sits in
; the cascade tail, dominated by all four NewInvs. Larger cascade size than the
; 2-/3-object cyclic tests.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @cyclic_live_path_4_object_cycle()
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %a = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 11111 to ptr), i32 16) to label %na unwind label %u1
na:
  %b = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16) to label %nb unwind label %u2
nb:
  %c = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 33333 to ptr), i32 16) to label %nc unwind label %u3
nc:
  %d = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 44444 to ptr), i32 16) to label %nd unwind label %u4
nd:
  %sa = getelementptr inbounds i8, ptr addrspace(1) %a, i64 8
  store atomic ptr addrspace(1) %b, ptr addrspace(1) %sa unordered, align 8
  %sb = getelementptr inbounds i8, ptr addrspace(1) %b, i64 8
  store atomic ptr addrspace(1) %c, ptr addrspace(1) %sb unordered, align 8
  %sc = getelementptr inbounds i8, ptr addrspace(1) %c, i64 8
  store atomic ptr addrspace(1) %d, ptr addrspace(1) %sc unordered, align 8
  %sd = getelementptr inbounds i8, ptr addrspace(1) %d, i64 8
  store atomic ptr addrspace(1) %a, ptr addrspace(1) %sd unordered, align 8
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
u4:
  %lp4 = landingpad i64 cleanup
  resume i64 %lp4
}

!java-method-compilation = !{}

; CHECK-LABEL: define ptr addrspace(1) @cyclic_live_path_4_object_cycle
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 11111 to ptr)
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 22222 to ptr)
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 33333 to ptr)
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 44444 to ptr)
; All four replayed stores use real NewInvs (no poison): the back edge D.i = A
; (A materialized last) resolves to A's NewInv.
; CHECK-COUNT-4: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK-NOT: poison
; CHECK: ret ptr addrspace(1) %{{.*}}
