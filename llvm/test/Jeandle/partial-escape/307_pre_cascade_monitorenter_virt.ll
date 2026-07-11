; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Two virtual-locked objects escaping at separate points.
;
; A and B are each virtual-monitorenter'd, then escape at distinct sink calls
; (A first, then B). Each escape is its own single materialize point: A
; materializes at sink(%a) and re-emits its monitorenter there, B materializes
; at sink(%b). Because the two escape at different instructions they do not
; form a cascade group, so each re-emits its own monitorenter per-effect; the
; re-emitted enters keep source order (A before B), matching the acquisition
; order. (A single shared escape point with interleaved locks would instead
; be globally depth-sorted via the merged re-emit — see 427.)

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1), ptr)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @pre_cascade() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %la = alloca i64, align 8
  %lb = alloca i64, align 8
  %a = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 11111 to ptr), i32 16)
       to label %n1 unwind label %u
n1:
  call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(
                  ptr addrspace(1) %a, ptr %la)
  %b = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 22222 to ptr), i32 16)
       to label %n2 unwind label %u
n2:
  ; Virtual monitorenter on B.
  call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(
                  ptr addrspace(1) %b, ptr %lb)
  ; Escape A first.
  call void @sink(ptr addrspace(1) %a) [ "deopt"(i32 0, i32 0) ]
  ; Escape B later.
  call void @sink(ptr addrspace(1) %b) [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; Both A and B must be materialised. Each materialize re-emits its surviving
; monitorenter then the sink consumes it (per-object grouping).
; CHECK-LABEL: define void @pre_cascade
; CHECK: %[[MATA:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 11111 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) %[[MATA]],
; CHECK: call void @sink(ptr addrspace(1) %[[MATA]])
; CHECK: %[[MATB:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) %[[MATB]],
; CHECK: call void @sink(ptr addrspace(1) %[[MATB]])

!java-method-compilation = !{}
