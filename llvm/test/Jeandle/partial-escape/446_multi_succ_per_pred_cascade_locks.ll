; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Multi-successor per-pred cascade with merged lock re-emit (review §1.2 F2).
;
; 442 exercises a SINGLE-succ per-pred cascade (right -> merge only); the
; critical-edge pre-pass guard `getNumSuccessors() <= 1` then skips the split,
; so the split-edge re-aim + merged-emit interaction is NOT reached. This test
; makes `right` have TWO successor merges (merge1, merge2), both mixed (left
; arm virtual, right arm locked) so per-pred materialization of the `a.f=b`
; cascade fires on BOTH (right, merge1) and (right, merge2) edges — both
; critical -> the pre-pass splits each into its own `pea.crit.split` and re-aims
; the per-pred Materialize effects there. The unified merged-emit then fires
; once per split edge, depth-sorted (a@0, b@1), each lock's receiver resolved
; per-effect via NewInvOf[SourceEffect].
;
; Without the §1.2 fix (per-pred effects skipped by computeEscapePointLocks +
; per-effect re-emit), each split edge would re-emit in SeqNo order (b@1 then
; a@0 = 1,0) — mis-ordered.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1), ptr)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_multi_succ_per_pred_cascade_locks(i1 %c, i1 %c2, i1 %c3)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %la = alloca i64, align 8
  %lb = alloca i64, align 8
  %a = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
          ptr inttoptr (i64 11111 to ptr), i32 16)
       to label %na unwind label %u
na:
  %b = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
          ptr inttoptr (i64 22222 to ptr), i32 16)
       to label %fld unwind label %u
fld:
  ; a.f = b: materializing a per-pred cascades b (forward prereq).
  %af = getelementptr inbounds i8, ptr addrspace(1) %a, i64 0
  store atomic ptr addrspace(1) %b, ptr addrspace(1) %af unordered, align 8
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %left, label %right
left:
  ; No locks; branches to BOTH merges so each merge is mixed (left virtual,
  ; right locked).
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c3, label %merge1, label %merge2
right:
  ; Unbalanced enters on a (depth 0) and b (depth 1); two successors.
  call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(
      ptr addrspace(1) %a, ptr %la)
  call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(
      ptr addrspace(1) %b, ptr %lb)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c2, label %merge1, label %merge2
merge1:
  call void @sink(ptr addrspace(1) %a) [ "deopt"(i32 0, i32 0) ]
  call void @sink(ptr addrspace(1) %b) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
merge2:
  call void @sink(ptr addrspace(1) %a) [ "deopt"(i32 0, i32 0) ]
  call void @sink(ptr addrspace(1) %b) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

!java-method-compilation = !{}

; Four critical-edge split blocks: `left` (no locks) splits to both merges,
; `right` (locked) splits to both merges. Only the two `right` splits carry
; re-emitted monitorenters.
; CHECK-LABEL: define void @test_multi_succ_per_pred_cascade_locks
; CHECK-COUNT-4: pea.crit.split
; On each right split, the two re-emitted enters appear strictly depth-increasing
; (a@0 with %la, then b@1 with %lb) — the per-effect (SeqNo) order would be
; b@1 then a@0 = 1,0. Two right splits => four enters, in la,lb,la,lb order.
; CHECK: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) %{{.*}}, ptr %la)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) %{{.*}}, ptr %lb)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) %{{.*}}, ptr %la)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) %{{.*}}, ptr %lb)
; CHECK-NOT: poison
; CHECK: ret void
