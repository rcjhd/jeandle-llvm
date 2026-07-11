; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Lock-stack identity mismatch at a diamond merge.
;
; A single virtual %o is allocated before the diamond. Each arm enters its
; OWN monitorenter call site on %o (lock_t and lock_e are distinct alloca
; slots used for two distinct enter call instances). At the merge, both
; preds report LockCount==1 on %o but the per-pred live stacks differ
; (CallSite_then != CallSite_else).
;
; Per-pred materialise routes through materializeAtPredFromExitInfo for
; each pred. The merged VO flips to Materialized; the analyzer emits a
; per-pred Materialize at each branch and a CreatePHI at the merge
; collecting both pred-side materialised pointers. The downstream
; monitorexit and use see the merged pointer.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1), ptr)
declare hotspotcc void @jeandle.monitorexit_with_lightweight_lock(ptr addrspace(1), ptr)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_lockdepth_stack_identity_per_pred(i1 %cond) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock_t = alloca i64, align 8
  %lock_e = alloca i64, align 8
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 9999 to ptr), i32 16)
       to label %dispatch unwind label %u
dispatch:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %cond, label %t, label %e
t:
  ; Then-arm enter on its own call site (proxy depth 0 — RPO visits %t first).
  call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(
                  ptr addrspace(1) %o, ptr %lock_t)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
e:
  ; Else-arm enter on a DIFFERENT call site (proxy depth 1 — %e is visited
  ; after %t). Call identity differs from the then-arm's enter, so locksEqual
  ; = false at the merge (the depth difference is incidental — the EnterCall
  ; mismatch alone routes to per-pred materialise).
  call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(
                  ptr addrspace(1) %o, ptr %lock_e)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  ; Both preds hold one lock on %o; downstream observes the merged pointer.
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  ; Note: there is no balanced monitorexit on the merged path on purpose;
  ; we are only checking the merge-time mat behaviour, and the per-pred
  ; enters survive in IR with their first operand RAUW'd onto the matching
  ; pred-side materialised pointer.
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; %o is materialised at each pred (per-pred Materialize
; effects); a CreatePHI at the merge collects both pred-side materialised
; pointers; the sink sees the phi. The original allocation is replaced by
; the per-pred materialised invokes (the entry-block alloc is removed
; entirely because every escape now happens at the per-pred mat sites).
;
; CHECK-LABEL: define void @test_lockdepth_stack_identity_per_pred
; Per-pred materialise: TWO new_instance invokes survive, one per arm.
; CHECK-DAG: %[[MATT:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 9999 to ptr), i32 16)
; CHECK-DAG: %[[MATE:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 9999 to ptr), i32 16)
; Both pred-side enter calls survive, each pointing at its own pred's
; materialised pointer.
; CHECK-DAG: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) {{.*}}, ptr %lock_t)
; CHECK-DAG: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) {{.*}}, ptr %lock_e)
; The merge block synthesises a phi over the two pred materialised pointers
; and the sink uses it.
; CHECK: phi ptr addrspace(1)
; CHECK: call void @sink(ptr addrspace(1)

!java-method-compilation = !{}
