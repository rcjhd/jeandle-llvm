; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Loop-body partial escape (Case B): an object allocated OUTSIDE the loop
; escapes only on a path that EXITS the loop (the `ret` on the `i==7`
; iteration). Graal keeps the object scalar-replaced on the normal path and
; materializes it only on the escape path (materializedValuePhi is NOT needed
; here because the escape path does not flow to the latch).
;
; Consequence: the normal path's field load folds to the prep-stored %x
; (object is virtual there), and the allocation appears only on the `ret`
; path. With the old hoist-to-alloc-normal-dest fallback the object was
; fully materialized before the loop and the load was a real load.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @use(i32)
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define ptr addrspace(1) @test_loopbody_escape_return(i32 %n, i32 %x) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %prep unwind label %u
prep:
  %s = getelementptr inbounds i8, ptr addrspace(1) %o, i64 8
  store atomic i32 %x, ptr addrspace(1) %s unordered, align 4
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %loop
loop:
  %i = phi i32 [ 0, %prep ], [ %i1, %cont ]
  %c = icmp slt i32 %i, %n
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %body, label %exit
body:
  %ce = icmp eq i32 %i, 7
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %ce, label %ret, label %cont
ret:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret ptr addrspace(1) %o
cont:
  %v = load atomic i32, ptr addrspace(1) %s unordered, align 4
  call void @use(i32 %v)
  %i1 = add i32 %i, 1
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %loop
exit:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret ptr addrspace(1) null
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; The normal path keeps the object virtual: the field load folds to %x.
; CHECK-LABEL: define ptr addrspace(1) @test_loopbody_escape_return
; The allocation is materialized on the escape (ret) path, replaying %x.
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(
; CHECK: store atomic i32 %x, ptr addrspace(1) %{{.*}} unordered, align 4
; CHECK: ret ptr addrspace(1) %{{.*}}
; The normal-path field load folds to %x (object stays virtual there).
; CHECK: call void @use(i32 %x)
; Exactly one allocation survives.
; CHECK-NOT: jeandle.new_instance

!java-method-compilation = !{}
