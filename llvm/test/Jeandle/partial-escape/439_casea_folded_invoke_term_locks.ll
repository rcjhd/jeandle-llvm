; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Case-A + locks + a folded JavaOp-invoke terminator. The object %o has an
; elided monitorenter (depth 0); Case-A materializes %o at `else`'s
; terminator (the folded arraylength invoke). The eager-update hook re-aims
; the Materialize's InsertBefore to the `br` replacement; the lock re-emit
; then resolves the escape point via CascadeKeyOf (the ORIGINAL captured
; InsertBefore), NOT the re-aimed E.InsertBefore — so the merged lock list
; is found and the monitorenter is re-emitted at the materialize point.

declare hotspotcc ptr addrspace(1) @jeandle.new_array(ptr, i32, i32, i32, i32)
declare hotspotcc i32 @jeandle.arraylength(ptr addrspace(1) readonly)
declare hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1), ptr)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @casea_folded_invoke_term_locks(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lo = alloca i64, align 8
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_array(ptr inttoptr (i64 12345 to ptr), i32 7, i32 44, i32 16, i32 1048576)
         to label %n unwind label %u
n:
  call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) %o, ptr %lo)
  br i1 %c, label %then, label %else
then:
  br label %merge
else:
  %len = invoke hotspotcc i32 @jeandle.arraylength(ptr addrspace(1) %o) to label %merge unwind label %handler
merge:
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

; CHECK-LABEL: define void @casea_folded_invoke_term_locks
; CHECK-NOT: @jeandle.arraylength
; The materialization re-emits the monitorenter at the materialize point
; (after the pea.mat invoke, in mat.cont), receiver = pea.mat.
; CHECK: else:
; CHECK-NEXT: %{{.*}} = invoke hotspotcc{{.*}}@jeandle.new_array(ptr inttoptr (i64 12345 to ptr), i32 7, i32 44, i32 16, i32 1048576)
; CHECK-NEXT: to label %mat.cont unwind label %u
; CHECK: mat.cont:
; CHECK-NEXT: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) %{{.*}}, ptr %lo)
; CHECK-NEXT: br label %merge
