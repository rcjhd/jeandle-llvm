; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Cyclic field graph (A.f = B, B.g = A) combined with an unbalanced
; monitorenter on A. Returning A escapes the whole cascade; both objects
; materialize at the escape point. The cascade tail replays the field stores and
; then re-emits the surviving monitorenter — stores precede locks within the
; tail (a single IRBuilder emits both in order), matching Graal's
; CommitAllocationNode lowering (field writes, then MonitorEnters). The back
; edge B.g = A resolves to A's NewInv.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @cyclic_with_locks_cascade()
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lk1 = alloca i64, align 8
  %a = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 11111 to ptr), i32 16) to label %na unwind label %u
na:
  %b = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16) to label %nb unwind label %u
nb:
  %sa = getelementptr inbounds i8, ptr addrspace(1) %a, i64 8
  store atomic ptr addrspace(1) %b, ptr addrspace(1) %sa unordered, align 8
  %sb = getelementptr inbounds i8, ptr addrspace(1) %b, i64 8
  store atomic ptr addrspace(1) %a, ptr addrspace(1) %sb unordered, align 8
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %a, ptr %lk1)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret ptr addrspace(1) %a
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

!java-method-compilation = !{}

; CHECK-LABEL: define ptr addrspace(1) @cyclic_with_locks_cascade
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 11111 to ptr)
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 22222 to ptr)
; In the cascade tail, both replayed field stores precede the re-emitted
; monitorenter (a single IRBuilder emits stores-then-locks). The back edge
; B.g = A uses A's NewInv (no poison).
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock
; CHECK-NOT: poison
; CHECK: ret ptr addrspace(1) %{{.*}}
