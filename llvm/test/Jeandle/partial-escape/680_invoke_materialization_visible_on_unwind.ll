; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; An ordinary invoke that receives a virtual object materializes it before the
; call. The materialized state must reach both the normal and unwind edges.
; The callee may update the object before throwing, so the handler's load
; cannot be folded from the pre-call virtual field snapshot.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @mutate_then_throw(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define i32 @test_294() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %obj = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
             ptr inttoptr (i64 11111 to ptr), i32 24)
          to label %call unwind label %alloc_unwind

call:
  %slot = getelementptr inbounds i8, ptr addrspace(1) %obj, i64 16
  store atomic i32 1, ptr addrspace(1) %slot unordered, align 4
  invoke void @mutate_then_throw(ptr addrspace(1) %obj)
       to label %normal unwind label %handler

normal:
  ret i32 0

handler:
  %lp = landingpad i64 cleanup
  %handler_slot = getelementptr inbounds i8, ptr addrspace(1) %obj, i64 16
  %value = load atomic i32, ptr addrspace(1) %handler_slot unordered, align 4
  ret i32 %value

alloc_unwind:
  %alloc_lp = landingpad i64 cleanup
  resume i64 %alloc_lp
}

; The allocation remains concrete because it escapes to the opaque invoke.
; Most importantly, the handler reads the concrete field after the call; PEA
; must not replace it with the stale constant 1.
; CHECK-LABEL: define i32 @test_294
; CHECK: invoke void @mutate_then_throw
; CHECK: handler:
; CHECK: %[[VALUE:.*]] = load atomic i32, ptr addrspace(1) %{{.*}} unordered
; CHECK: ret i32 %[[VALUE]]
; CHECK-NOT: ret i32 1

!java-method-compilation = !{}
