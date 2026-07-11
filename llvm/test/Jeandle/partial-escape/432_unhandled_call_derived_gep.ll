; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Generic dispatch fallback (processInstruction): an UNHANDLED call consuming a
; derived-GEP operand of a virtual object. Materialising at the call would
; place pea.mat after the GEP, so the GEP's OrigAlloc use is not dominated and
; resolves to poison. materializeVirtualOperandsSafely detects the derived
; operand structurally (V != AllocationCall — NOT by offset) and keeps the
; object real (markIneligible) so the GEP stays valid.
;
; t_derived_off16 and t_derived_off0 both exercise the derived-operand bail
; (offset-0 derived GEPs poison just like non-zero ones, so the check must be
; structural). t_whole_object is a non-regression: a whole-object call argument
; is the original allocation directly, so it is still materialised (sound, and
; Graal's processNodeInputs analog).

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @external(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @t_derived_off16() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 32)
         to label %n unwind label %u
n:
  %g = getelementptr inbounds i8, ptr addrspace(1) %o, i64 16
  store atomic i32 42, ptr addrspace(1) %g unordered, align 4
  call void @external(ptr addrspace(1) %g) [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

define void @t_derived_off0() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 32)
         to label %n unwind label %u
n:
  %g = getelementptr inbounds i8, ptr addrspace(1) %o, i64 0
  call void @external(ptr addrspace(1) %g) [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

define void @t_whole_object() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 32)
         to label %n unwind label %u
n:
  call void @external(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; Derived-operand calls: the object stays real and the GEP keeps a real base
; (CHECK-NOT poison right after each LABEL so its scope spans the body where
; the pre-fix poison GEP sits).
; CHECK-LABEL: define void @t_derived_off16
; CHECK-NOT: getelementptr{{.*}}poison
; CHECK: invoke{{.*}}@jeandle.new_instance
; CHECK: call void @external(ptr addrspace(1) %g)

; CHECK-LABEL: define void @t_derived_off0
; CHECK-NOT: getelementptr{{.*}}poison
; CHECK: invoke{{.*}}@jeandle.new_instance
; CHECK: call void @external(ptr addrspace(1) %g)

; Whole-object call: materialised (the argument is the allocation directly).
; CHECK-LABEL: define void @t_whole_object
; CHECK: pea.mat{{.*}}@jeandle.new_instance
; CHECK: call void @external(ptr addrspace(1) %pea.mat)

!java-method-compilation = !{}
