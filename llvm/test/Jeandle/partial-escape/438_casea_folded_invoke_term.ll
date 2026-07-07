; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Case-A (processBlockPhis, SkipGlobalRAUW=false, IsPerPred=false) materializes
; %o at `else`'s TERMINATOR, which is a folded JavaOp invoke (jeandle.arraylength
; on the virtual %o). The invoke's ReplaceCall erases it before the Case-A
; Materialize applies. The eager-update hook (relocateDependentMaterializes,
; called from ReplaceCallEffect::apply) re-aims the Materialize's InsertBefore
; to the `br` that replaced the invoke (the normal successor in the SAME block)
; before the erase nulls the WeakTrackingVH. The materialization must land at
; the end of `else` (before `br merge`), NOT at %o's allocation normal-dest
; (premature escape + SSA-dominance-unsound replay). Without the eager update,
; applyMaterialize's InsertBefore-null assert fires (the old unsound fallback
; hoisted to the normal-dest head).

declare hotspotcc ptr addrspace(1) @jeandle.new_array(ptr, i32, i32, i32, i32)
declare hotspotcc i32 @jeandle.arraylength(ptr addrspace(1) readonly)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @casea_folded_invoke_term(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_array(ptr inttoptr (i64 12345 to ptr), i32 7, i32 44, i32 16, i32 1048576)
         to label %n unwind label %u
n:
  br i1 %c, label %then, label %else
then:
  br label %merge
else:
  ; arraylength on the virtual %o folds -> ReplaceCall erases this invoke
  ; (else's terminator). else has two successors (merge, handler).
  %len = invoke hotspotcc i32 @jeandle.arraylength(ptr addrspace(1) %o)
              to label %merge unwind label %handler
merge:
  ; PHI mixing %o virtual (else arm) and null (then arm) -> Case-A materialize
  ; of %o at else's terminator (the erased arraylength invoke).
  %p = phi ptr addrspace(1) [ null, %then ], [ %o, %else ]
  call void @sink(ptr addrspace(1) %p)
  ret void
handler:
  %lp = landingpad i64 cleanup
  resume i64 %lp
u:
  %lpr = landingpad i64 cleanup
  resume i64 %lpr
}

!java-method-compilation = !{}

; CHECK-LABEL: define void @casea_folded_invoke_term
; The arraylength invoke is gone (folded).
; CHECK-NOT: @jeandle.arraylength
; The materialization lands at the end of `else` (re-aimed off the erased
; arraylength-invoke terminator onto the `br`), not at %o's alloc normal-dest.
; CHECK: else:
; CHECK-NEXT: %{{.*}} = invoke hotspotcc{{.*}}@jeandle.new_array(ptr inttoptr (i64 12345 to ptr), i32 7, i32 44, i32 16, i32 1048576)
; CHECK-NEXT: to label %mat.cont unwind label %u
; CHECK: mat.cont:
; CHECK-NEXT: br label %merge
; CHECK: merge:
; CHECK: %p = phi ptr addrspace(1) [ null, %{{.*}} ], [ %{{.*}}, %mat.cont ]
