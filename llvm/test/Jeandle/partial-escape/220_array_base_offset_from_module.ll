; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" -jeandle-vm-callback-log=%S/Inputs/220_array_base_offset_from_module.cblog %s | FileCheck %s

; new_array 5-arg protocol verification:
;   The allocation carries base_offset=24 directly. The module also declares
;   matching runtime-defined globals (base=24, element=4), as HotSpot would,
;   so the typed-GEP matcher can derive the element scale and the debug assert
;   sees the same VM layout. With base==24, the i8 GEP at offset 24 (rather
;   than 16) must match the typed-int GEP chain for index 0, so the
;   store/load round-trip is eliminated and the load folds to 42.

@arrayOopDesc.base_offset_in_bytes.int = private constant i32 24
@arrayOopDesc.element_size.int         = private constant i32 4

declare hotspotcc ptr addrspace(1) @jeandle.new_array(ptr, i32, i32, i32, i32)

declare i32 @__gxx_personality_v0(...)

define i32 @test_base_offset_24() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %arr = invoke hotspotcc ptr addrspace(1) @jeandle.new_array(
            ptr inttoptr (i64 54321 to ptr), i32 2, i32 32, i32 24, i32 1048576)
         to label %n unwind label %u
n:
  ; Note: base offset is 24, not the compile-time default 16.
  %base = getelementptr inbounds i8, ptr addrspace(1) %arr, i32 24
  %p0 = getelementptr inbounds i32, ptr addrspace(1) %base, i64 0
  store atomic i32 42, ptr addrspace(1) %p0 unordered, align 4
  %v0 = load atomic i32, ptr addrspace(1) %p0 unordered, align 4
  ret i32 %v0
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define i32 @test_base_offset_24
; CHECK-NOT: jeandle.new_array
; CHECK-NOT: store
; CHECK-NOT: load
; CHECK: ret i32 42

!java-method-compilation = !{}
