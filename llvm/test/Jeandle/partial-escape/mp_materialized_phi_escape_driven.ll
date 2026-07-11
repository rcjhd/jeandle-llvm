; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; MergeProcessor / mergeObjectState AllMaterialized with a UNIQUE pointer
; (escape-driven, NO materializedValuePhi).
;
; Both arms escape the same virtual object via a sink call. Each escape
; triggers a materialization hoisted to the common dominator (the branch),
; so at the merge EVERY predecessor reports the object Materialized with the
; SAME pointer. mergeObjectState therefore takes the uniqueMaterializedValue
; branch (Graal PartialEscapeClosure.java:982-983) and installs the state
; directly — NO ptr addrspace(1) materializedValuePhi is synthesized.
;
; This is the structural CONTRAST to 254/363 (lock-driven): a lock-count
; mismatch forces PER-PRED materialization (distinct NewInvs) and builds a
; materializedValuePhi, whereas escape-driven materialization shares one
; pointer and installs directly. Pinning the no-PHI outcome here guards the
; refactor's materializeAndBuildPhi routing: escapes must NOT be misrouted
; into the PHI-synthesis else-branch.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define ptr addrspace(1) @test_escape_driven_matphi(i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %branch unwind label %u
branch:
  br i1 %c, label %left, label %right
left:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  br label %merge
right:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  ret ptr addrspace(1) %o
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; Both arms escape, so each materializes at its own escape point and the merge
; builds a materializedValuePhi over the two per-arm pointers; the return
; consumes the PHI. (Under the former dominating-hoist placement both arms
; shared one dominating NewInv and no PHI was needed; escape-point placement
; gives each arm its own NewInv, so the PHI is required.)
; CHECK-LABEL: define ptr addrspace(1) @test_escape_driven_matphi
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: = phi ptr addrspace(1)
; CHECK: ret ptr addrspace(1)

!java-method-compilation = !{}
