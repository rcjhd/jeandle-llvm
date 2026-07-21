; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Both monitor invokes are scalar-replaced, so their unwind edges are removed
; by the transform. processLoopExit must use the same effective CFG as the
; block-state analysis and ignore those killed EH exits. Otherwise the
; non-trivial handler forces the otherwise non-escaping object to materialize.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1), ptr)
declare hotspotcc void @jeandle.monitorexit_with_lightweight_lock(ptr addrspace(1), ptr)
declare void @escape(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @loop_with_killed_monitor_unwind_edges(i32 %limit)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock = alloca i64, align 8
  %obj = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
             ptr inttoptr (i64 12345 to ptr), i32 16)
         to label %preheader unwind label %alloc_unwind

preheader:
  br label %header

header:
  %i = phi i32 [ 0, %preheader ], [ %next, %latch ]
  invoke hotspotcc void @jeandle.monitorenter_with_lightweight_lock(
             ptr addrspace(1) %obj, ptr %lock)
         to label %critical unwind label %monitor_unwind

critical:
  invoke hotspotcc void @jeandle.monitorexit_with_lightweight_lock(
             ptr addrspace(1) %obj, ptr %lock)
         to label %latch unwind label %monitor_unwind

latch:
  %next = add nuw i32 %i, 1
  %done = icmp uge i32 %next, %limit
  br i1 %done, label %exit, label %header

exit:
  ret void

monitor_unwind:
  %monitor_lp = landingpad i64 cleanup
  call void @escape(ptr addrspace(1) %obj)
  resume i64 %monitor_lp

alloc_unwind:
  %alloc_lp = landingpad i64 cleanup
  resume i64 %alloc_lp
}

; CHECK-LABEL: define void @loop_with_killed_monitor_unwind_edges(
; CHECK-NOT: @jeandle.new_instance
; CHECK-NOT: @jeandle.monitorenter_with_lightweight_lock
; CHECK-NOT: @jeandle.monitorexit_with_lightweight_lock
; CHECK: ret void

!java-method-compilation = !{}
