; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; §2.1: when a virtual object is materialized at an escape point that sits
; inside a Windows-EH funclet, the materialization invoke MUST carry a
; `[ "funclet"(token %pad) ]` operand bundle naming the ENCLOSING funclet pad,
; or the verifier rejects it. The pad is found via colorEHFunclets keyed on the
; invoke's actual host block (the post-split Origin), NOT MatCont->getFirstNonPHI
; (the materialize invoke is emitted at the end of Origin, so it belongs to
; Origin's funclet; a pad never sits at an arbitrary block head).
;
; Jeandle does not currently target Windows; this is the IR-defensiveness rule
; (review requirement #6 — PEA must tolerate any legal IR).
;
; Critical shape: the object %o is allocated in `entry` (OUTSIDE any funclet, so
; the original allocation carries NO funclet bundle) but only escapes (via
; @sink) INSIDE the cleanup funclet. PEA therefore virtualizes %o in entry and
; materializes it at the escape inside the funclet — the materialize invoke has
; no source funclet bundle to copy, so the colorEHFunclets synthesis path is the
; only thing that can supply the required bundle.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @may_throw()
declare void @sink(ptr addrspace(1))
declare i32 @__CxxFrameHandler3(...)

define void @test_funclet_bundle() personality ptr @__CxxFrameHandler3 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 7 to ptr), i32 16)
       to label %afteralloc unwind label %ehalloc

afteralloc:
  %slot = getelementptr inbounds i8, ptr addrspace(1) %o, i64 8
  store atomic i32 42, ptr addrspace(1) %slot unordered, align 4
  invoke void @may_throw()
      to label %cont unwind label %cleanup

cont:
  ret void

cleanup:
  %cp = cleanuppad within none []
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0), "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller

ehalloc:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_funclet_bundle
; CHECK: %[[CP:[A-Za-z0-9._]+]] = cleanuppad within none []
; The materialization invoke (inside the funclet) carries the enclosing pad.
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 7 to ptr), i32 16){{.*}}[ "funclet"(token %[[CP]]), "deopt"(i32 0, i32 0) ]

!java-method-compilation = !{}
