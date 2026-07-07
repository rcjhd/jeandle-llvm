; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" -jeandle-vm-callback-log=%S/Inputs/217_postbarrier_variable_index_survives.cblog %s | FileCheck %s

; Object[] virtual array; the store (and the post_barrier) target a VARIABLE
; element index. processStore cannot resolve a constant field offset for a
; variable index, so it marks the array ineligible. foldPostBarrier still
; resolves the address operand to the (still-virtual-at-analysis-point) array
; and emits a ReplaceCallEffect carrying the array's ObjID. At commit(),
; dropEffectsFor(arrayID) purges that ReplaceCallEffect (and the array's
; EliminateAllocationEffect), so the ORIGINAL jeandle.post_barrier must survive
; guarding the real store, and the allocation must stay. This ties the
; barrier-erasure to the array's eligibility — the same ObjID linkage
; foldArrayStoreCheck uses.

declare hotspotcc ptr addrspace(1) @jeandle.new_array(ptr, i32, i32, i32, i32)
declare hotspotcc void @jeandle.post_barrier(ptr addrspace(1), ptr addrspace(1))

declare i32 @__gxx_personality_v0(...)

define void @test_postbarrier_variable_index_survives(i64 %i) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %arr = invoke hotspotcc ptr addrspace(1) @jeandle.new_array(
            ptr inttoptr (i64 8888 to ptr), i32 4, i32 32, i32 16, i32 1048576)
         to label %n unwind label %u
n:
  %base = getelementptr inbounds i8, ptr addrspace(1) %arr, i32 16
  %pi = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %base, i64 %i
  store atomic ptr addrspace(1) null, ptr addrspace(1) %pi unordered, align 4
  call hotspotcc void @jeandle.post_barrier(ptr addrspace(1) %pi,
                                           ptr addrspace(1) null)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_postbarrier_variable_index_survives
; CHECK: jeandle.new_array
; CHECK: store atomic
; CHECK: jeandle.post_barrier

!java-method-compilation = !{}
