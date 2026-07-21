; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; The allocation invoke exits the loop on its unwind edge. Its result exists
; only in the normal-edge state, so processLoopExit must not force-materialize
; it for the non-trivial cleanup. Using the block.s
; unified normal state for that unwind edge conflicts with the normal-path
; normal state otherwise retains a fully scalar-replaceable allocation.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink_i32(i32)
declare void @observe_exception()
declare i32 @__gxx_personality_v0(...)

define void @loop_allocation_unwind_uses_preallocation_state(i32 %n) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  br label %hdr

hdr:
  %i = phi i32 [0, %entry], [%inext, %latch]
  %c = icmp slt i32 %i, %n
  br i1 %c, label %body, label %exit

body:
  %obj = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
                  ptr inttoptr (i64 7777 to ptr), i32 16)
              [ "deopt"(i32 1, i32 1) ]
              to label %b unwind label %u
b:
  %slot = getelementptr inbounds i8, ptr addrspace(1) %obj, i64 8
  store atomic i32 7, ptr addrspace(1) %slot unordered, align 4
  %v = load atomic i32, ptr addrspace(1) %slot unordered, align 4
  call void @sink_i32(i32 %v)
  br label %latch
latch:
  %inext = add i32 %i, 1
  br label %hdr

exit:
  ret void
u:
  %lp = landingpad i64 cleanup
  call void @observe_exception()
  resume i64 %lp
}

; CHECK-LABEL: define void @loop_allocation_unwind_uses_preallocation_state
; Exact unwind-state selection lets scalar replacement remove the allocation.
; CHECK-NOT: @jeandle.new_instance
; CHECK-NOT: load atomic i32
; CHECK: call void @sink_i32(i32 7)
; CHECK-NOT: call void @observe_exception()

!java-method-compilation = !{}
