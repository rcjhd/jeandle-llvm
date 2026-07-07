; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" -jeandle-vm-callback-log=%S/Inputs/219_postbarrier_phi_addr_survives.cblog %s | FileCheck %s

; jeandle.post_barrier(addr, oop) where `addr` is a PHI of two DISTINCT virtual
; arrays (allocated on opposite sides of a diamond). resolveVirtualRef only
; returns an ObjectID when all live PHI incomings resolve to the SAME virtual
; object; here they resolve to two different ObjectIDs, so it returns nullopt
; and foldPostBarrier returns false. The generic escape path materializes both
; arrays and the barrier survives guarding the real store. Exercises the
; resolveVirtualRef PHI-handling bail path for the new fold.

declare hotspotcc ptr addrspace(1) @jeandle.new_array(ptr, i32, i32, i32, i32)
declare hotspotcc void @jeandle.post_barrier(ptr addrspace(1), ptr addrspace(1))

declare i32 @__gxx_personality_v0(...)

define void @test_postbarrier_phi_addr_survives(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  br i1 %c, label %a, label %b
a:
  %arr1 = invoke hotspotcc ptr addrspace(1) @jeandle.new_array(
             ptr inttoptr (i64 8881 to ptr), i32 4, i32 32, i32 16, i32 1048576)
          to label %merge unwind label %u
b:
  %arr2 = invoke hotspotcc ptr addrspace(1) @jeandle.new_array(
             ptr inttoptr (i64 8882 to ptr), i32 4, i32 32, i32 16, i32 1048576)
          to label %merge unwind label %u
merge:
  %arr = phi ptr addrspace(1) [ %arr1, %a ], [ %arr2, %b ]
  %base = getelementptr inbounds i8, ptr addrspace(1) %arr, i32 16
  %p2 = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %base, i64 2
  store atomic ptr addrspace(1) null, ptr addrspace(1) %p2 unordered, align 4
  call hotspotcc void @jeandle.post_barrier(ptr addrspace(1) %p2,
                                           ptr addrspace(1) null)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; Both arrays must MATERIALIZE (the PHI of distinct virtuals cannot stay
; virtual), and the barrier survives guarding the real store.
; CHECK-LABEL: define void @test_postbarrier_phi_addr_survives
; CHECK: jeandle.new_array
; CHECK: store atomic
; CHECK: jeandle.post_barrier

!java-method-compilation = !{}
