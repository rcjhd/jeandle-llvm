; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Scalar values captured in virtual field state must be canonicalized during
; analysis. The second store therefore records 42 rather than %first, and the
; third records 42 rather than %second. Transform intentionally does not repair
; deferred replacement chains after an earlier load has been erased.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare i32 @__gxx_personality_v0(...)

define i32 @test_replace_load_replacement_chain()
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %obj = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 24)
         to label %normal unwind label %unwind

normal:
  %slot0 = getelementptr inbounds i8, ptr addrspace(1) %obj, i64 8
  %slot1 = getelementptr inbounds i8, ptr addrspace(1) %obj, i64 12
  %slot2 = getelementptr inbounds i8, ptr addrspace(1) %obj, i64 16

  store atomic i32 42, ptr addrspace(1) %slot0 unordered, align 4
  %first = load atomic i32, ptr addrspace(1) %slot0 unordered, align 4
  store atomic i32 %first, ptr addrspace(1) %slot1 unordered, align 4
  %second = load atomic i32, ptr addrspace(1) %slot1 unordered, align 4
  store atomic i32 %second, ptr addrspace(1) %slot2 unordered, align 4
  %third = load atomic i32, ptr addrspace(1) %slot2 unordered, align 4
  ret i32 %third

unwind:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define i32 @test_replace_load_replacement_chain()
; CHECK-NOT: @jeandle.new_instance
; CHECK-NOT: store atomic
; CHECK-NOT: load atomic
; CHECK: ret i32 42

!java-method-compilation = !{}
