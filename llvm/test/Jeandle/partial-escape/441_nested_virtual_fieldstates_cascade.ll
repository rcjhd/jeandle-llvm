; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" -jeandle-vm-callback-log=%S/Inputs/441_nested_virtual_fieldstates_cascade.cblog %s | FileCheck %s

; Nested-virtual poison via FieldStates VirtualRef (review §1.1).
;
; An Object[] array `arr` receives a CONSTANT-index store of a fresh virtual
; instance (arr[0] = inner0): processStore succeeds, records
; FieldStates[arr][16] = VirtualRef(inner0) and emits an EliminateStoreEffect
; for arr. A second store at a SYMBOLIC index `%i` (arr[i] = innerI) fails
; resolveAccess -> bailKeepingOperandsReal -> markIneligible(arr) + innerI.
; commit() then dropEffectsFor(arr) erases arr's EliminateStoreEffect, so the
; arr[0] store survives as a real store. If inner0 were left eligible it would
; be classified NeverEscapes (no surviving Materialize) -> Pass 2 RAUWs
; inner0.OrigAlloc to poison -> the surviving real arr[0] store writes poison
; into the escaped array -> miscompile.
;
; The fix is a transitive ineligibility cascade in commit() over FieldStates
; VirtualRef entries: any virtual referenced by a kept-real (ineligible)
; object's field is also kept real (its OrigAlloc survives), so the surviving
; store writes the real pointer. This runs once at classification time, so it
; catches VirtualRefs recorded BEFORE the bail (test_nested_virtual... —
; arr[0] precedes arr[i]) AND AFTER (test_reverse_order... — arr[i] precedes
; arr[0], recorded because processStore does not check Eligible). A walk-time
; cascade in markIneligible would be order-dependent and miss the latter.
; Mirrors Graal's recursive entry cascade (materializeWithCommit,
; PartialEscapeBlockState.java:293-343) on the conservative path; Graal
; materializes in place so it has no analysis/transform split.

declare hotspotcc ptr addrspace(1) @jeandle.new_array(ptr, i32, i32, i32, i32)
declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

; Order A: constant store FIRST, symbolic bail SECOND. The VirtualRef is in
; FieldStates[arr] before markIneligible(arr) runs.
define void @test_nested_virtual_fieldstates_cascade(i64 %i) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %arr = invoke hotspotcc ptr addrspace(1) @jeandle.new_array(
            ptr inttoptr (i64 8888 to ptr), i32 10, i32 56, i32 16, i32 1048576)
         to label %n0 unwind label %u
n0:
  %inner0 = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 5555 to ptr), i32 24)
           to label %n1 unwind label %u
n1:
  %innerI = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 5556 to ptr), i32 24)
           to label %n2 unwind label %u
n2:
  %base = getelementptr inbounds i8, ptr addrspace(1) %arr, i32 16
  %slot0 = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %base, i64 0
  store atomic ptr addrspace(1) %inner0, ptr addrspace(1) %slot0 unordered, align 4
  %slotI = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %base, i64 %i
  store atomic ptr addrspace(1) %innerI, ptr addrspace(1) %slotI unordered, align 4
  call void @sink(ptr addrspace(1) %arr)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; Order B: symbolic bail FIRST, constant store SECOND. The VirtualRef is
; recorded AFTER markIneligible(arr) ran, so a walk-time cascade would miss
; it; only the commit-time cascade catches it.
define void @test_reverse_order_fieldstates_cascade(i64 %i) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %arr = invoke hotspotcc ptr addrspace(1) @jeandle.new_array(
            ptr inttoptr (i64 8888 to ptr), i32 10, i32 56, i32 16, i32 1048576)
         to label %n0 unwind label %u
n0:
  %innerI = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 5555 to ptr), i32 24)
           to label %n1 unwind label %u
n1:
  %inner0 = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 5556 to ptr), i32 24)
           to label %n2 unwind label %u
n2:
  %base = getelementptr inbounds i8, ptr addrspace(1) %arr, i32 16
  %slotI = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %base, i64 %i
  store atomic ptr addrspace(1) %innerI, ptr addrspace(1) %slotI unordered, align 4
  %slot0 = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %base, i64 0
  store atomic ptr addrspace(1) %inner0, ptr addrspace(1) %slot0 unordered, align 4
  call void @sink(ptr addrspace(1) %arr)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_nested_virtual_fieldstates_cascade
; CHECK: jeandle.new_array
; CHECK-COUNT-2: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK-NOT: poison

; CHECK-LABEL: define void @test_reverse_order_fieldstates_cascade
; CHECK: jeandle.new_array
; Both inner instances survive real; the arr[0] store writes the real inner0
; pointer (not poison), even though inner0's VirtualRef was recorded after
; markIneligible(arr).
; CHECK-COUNT-2: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK-NOT: poison

!java-method-compilation = !{}
