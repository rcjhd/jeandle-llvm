; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A nested materialization on one branch must not destroy the allocation-to-VO
; alias needed by a sibling branch. RPO visits %nested before %direct here:
; %nested escapes %outer and recursively materializes %inner, while %direct
; later escapes %inner itself. Both paths must replay inner.field = 42.
;
; The old implementation reset %inner's function-wide alias while processing
; %nested. The still-virtual state on %direct then became unreachable, leaving
; that path with the original zero-initialized allocation.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @test_nested_materialize_preserves_sibling_alias(i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
               ptr inttoptr (i64 11111 to ptr), i32 16)
            to label %init unwind label %u1

init:
  %inner.field = getelementptr inbounds i8, ptr addrspace(1) %inner, i64 8
  store atomic i32 42, ptr addrspace(1) %inner.field unordered, align 4
  %outer = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
               ptr inttoptr (i64 22222 to ptr), i32 16)
            to label %branch unwind label %u2

branch:
  %outer.field = getelementptr inbounds i8, ptr addrspace(1) %outer, i64 8
  store atomic ptr addrspace(1) %inner, ptr addrspace(1) %outer.field unordered, align 8
  br i1 %c, label %direct, label %nested

direct:
  call void @sink(ptr addrspace(1) %inner) [ "deopt"(i32 1, i32 1) ]
  ret void

nested:
  call void @sink(ptr addrspace(1) %outer) [ "deopt"(i32 2, i32 2) ]
  ret void

u1:
  %lp1 = landingpad i64 cleanup
  resume i64 %lp1

u2:
  %lp2 = landingpad i64 cleanup
  resume i64 %lp2
}

; CHECK-LABEL: define void @test_nested_materialize_preserves_sibling_alias
; CHECK-COUNT-2: store atomic i32 42,

!java-method-compilation = !{}
