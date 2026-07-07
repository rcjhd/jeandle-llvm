; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Case-A CASCADE: two objects %o1 (array) and %o2 (instance), both virtual on
; `else` and absent on `then`, both Case-A-materialized at `else`'s terminator
; — a folded jeandle.arraylength invoke on %o1. The eager-update hook re-aims
; BOTH cascade members' InsertBefore off the erased invoke onto the `br`
; replacement (re-indexing each into the `br`'s bucket). The two materialize
; invokes then chain at the `br` (the `br` is reparented by successive
; splitBasicBlock calls, never erased): else -> pea.mat -> mat.cont ->
; pea.mat2 -> mat.cont1 -> br merge. Without the eager update, both members'
; InsertBefore null and applyMaterialize's assert fires.

declare hotspotcc ptr addrspace(1) @jeandle.new_array(ptr, i32, i32, i32, i32)
declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc i32 @jeandle.arraylength(ptr addrspace(1) readonly)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @casea_folded_invoke_term_cascade(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o1 = invoke hotspotcc ptr addrspace(1) @jeandle.new_array(ptr inttoptr (i64 12345 to ptr), i32 7, i32 44, i32 16, i32 1048576) to label %n1 unwind label %u1
n1:
  %o2 = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16) to label %n2 unwind label %u2
n2:
  br i1 %c, label %then, label %else
then:
  br label %merge
else:
  ; arraylength reads %o1 (virtual) -> folds, erasing this invoke (else's
  ; terminator). Both %o1 and %o2 are virtual here, absent on `then` -> two
  ; Case-A materializes at the same erased terminator (cascade group).
  %len = invoke hotspotcc i32 @jeandle.arraylength(ptr addrspace(1) %o1) to label %merge unwind label %handler
merge:
  %p1 = phi ptr addrspace(1) [ null, %then ], [ %o1, %else ]
  %p2 = phi ptr addrspace(1) [ null, %then ], [ %o2, %else ]
  call void @sink(ptr addrspace(1) %p1)
  call void @sink(ptr addrspace(1) %p2)
  ret void
handler:
  %lp = landingpad i64 cleanup
  resume i64 %lp
u1:
  %lp1 = landingpad i64 cleanup
  resume i64 %lp1
u2:
  %lp2 = landingpad i64 cleanup
  resume i64 %lp2
}

!java-method-compilation = !{}

; CHECK-LABEL: define void @casea_folded_invoke_term_cascade
; CHECK-NOT: @jeandle.arraylength
; The two materializations chain at the `br` (re-aimed off the erased
; arraylength-invoke terminator): else -> pea.mat -> mat.cont -> pea.mat2.
; CHECK: else:
; CHECK-NEXT: %{{.*}} = invoke hotspotcc{{.*}}@jeandle.new_array(ptr inttoptr (i64 12345 to ptr), i32 7, i32 44, i32 16, i32 1048576)
; CHECK-NEXT: to label %mat.cont unwind label %u1
; CHECK: mat.cont:
; CHECK-NEXT: %{{.*}} = invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16)
; CHECK-NEXT: to label %mat.cont{{.*}} unwind label %u2
; CHECK: mat.cont{{.*}}:
; CHECK-NEXT: br label %merge
; CHECK: merge:
; CHECK: %p1 = phi ptr addrspace(1) [ null, %{{.*}} ], [ %{{.*}}, %mat.cont{{.*}} ]
; CHECK: %p2 = phi ptr addrspace(1) [ null, %{{.*}} ], [ %{{.*}}, %mat.cont{{.*}} ]
