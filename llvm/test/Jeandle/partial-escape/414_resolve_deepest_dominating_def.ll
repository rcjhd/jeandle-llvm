; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s
; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" -verify-each %s | FileCheck %s
;
; §1.4 deepest/nearest-dominating-def selection in resolveMaterializedUses.
;
; A loop-local object %X is allocated in the loop body and escapes on BOTH arms
; of an in-body diamond (@sink on each), so each arm materializes its OWN NewInv
; (pea.mat / pea.mat2) and the arm-merge synthesizes a materialized-ptr PHI
; (pea.materialized.phi, RAUWOrigToPHI). The object is then carried across the
; back-edge by the header PHI %px. Defs[%X] therefore holds THREE defs: the
; arm-merge pea.materialized.phi and the two per-arm NewInvs. The surviving
; in-body @use must resolve to the NEAREST dominating def — the arm-merge
; pea.materialized.phi — never to an arm NewInv that dominates only its own arm,
; and never to a stale earlier-in-RPO def.
;
; The §1.4 miscompile IS reachable today: tests 130/133/176 exercise the
; header-carried materializedValuePhi + a body/global NewInv co-dominating a
; surviving use, where the old "first dominating def" picked the wrong (earlier)
; def. This test adds a distinct shape — a multi-arm in-body diamond whose
; arm-merge materialized.phi is the nearest dominating def of the post-merge
; @use, with two per-arm NewInvs that each dominate only their own arm. The
; deepest-def selection binds @use to the arm-merge phi (the dominator-tree
; leaf), never to an arm NewInv, never to a stale earlier-in-RPO def. See
; docs/Jeandle-PEA-Review.md §1.4.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare void @use(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @test_414_deepest_def(i32 %n, i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  br label %ohdr
ohdr:
  %oi = phi i32 [ 0, %entry ], [ %oi1, %olatch ]
  %px = phi ptr addrspace(1) [ null, %entry ], [ %X, %olatch ]
  %oc = icmp slt i32 %oi, %n
  br i1 %oc, label %obody, label %oexit
obody:
  %X = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 5555 to ptr), i32 16)
       to label %arm unwind label %u
arm:
  br i1 %c, label %a1, label %a2
a1:
  call void @sink(ptr addrspace(1) %X) [ "deopt"(i32 0, i32 0) ]
  br label %amrg
a2:
  call void @sink(ptr addrspace(1) %X) [ "deopt"(i32 0, i32 0) ]
  br label %amrg
amrg:
  call void @use(ptr addrspace(1) %X)
  br label %olatch
olatch:
  %oi1 = add i32 %oi, 1
  br label %ohdr
oexit:
  call void @sink(ptr addrspace(1) %px) [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; The arm-merge builds a materialized-ptr PHI over the two per-arm NewInvs, and
; the post-merge in-body @use must bind to THAT phi (the nearest dominating def),
; not to either arm NewInv.
; CHECK-LABEL: define void @test_414_deepest_def
; CHECK: %[[MAT1:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: %[[MAT2:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}@jeandle.new_instance
; CHECK: %[[PHI:pea\.materialized\.phi[0-9]*]] = phi ptr addrspace(1)
; CHECK: call void @use(ptr addrspace(1) %[[PHI]])
; CHECK-NOT: poison

!java-method-compilation = !{}
