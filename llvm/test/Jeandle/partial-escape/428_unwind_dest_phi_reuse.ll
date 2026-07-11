; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; findOrSynthesizeUnwindDest PHI reuse. %uw has TWO invoke
; predecessors (the alloc %o and a second invoke) and a USED PHI over them.
; When %o escapes it is materialized; the materialization invoke's unwind dest
; is chosen by findOrSynthesizeUnwindDest. Before the fix, Strategy 1 reused %uw
; verbatim without checking phis().empty(), so the new invoke made its block a
; predecessor of %uw with no matching PHI incoming — a PHI/predecessor mismatch
; the verifier rejects (the PHI is used, so it is not DCE'd). After the fix, a
; PHI-carrying unwind dest is not reused: a synthesized pea.unwind block is.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @may_throw()
declare void @sink(ptr addrspace(1))
declare void @use_i32(i32)
declare i32 @__gxx_personality_v0(...)

define void @test_unwind_dest_phi_reuse() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
  ptr inttoptr (i64 12345 to ptr), i32 16)
  to label %cont unwind label %uw
cont:
  invoke void @may_throw() to label %escape unwind label %uw
escape:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  ret void
uw:
  %sel = phi i32 [ 0, %entry ], [ 1, %cont ]
  %lp = landingpad i64 cleanup
  call void @use_i32(i32 %sel)
  resume i64 %lp
}

; CHECK-LABEL: define void @test_unwind_dest_phi_reuse
; The materialization invoke must unwind to a synthesized pea.unwind block
; (not the PHI-carrying %uw), so the module verifies and %uw's PHI is not
; corrupted by a path-wrong incoming.
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: to label %mat.cont unwind label %pea.unwind
; CHECK: call void @sink(ptr addrspace(1)

!java-method-compilation = !{}
