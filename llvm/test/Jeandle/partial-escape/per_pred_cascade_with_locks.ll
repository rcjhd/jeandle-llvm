; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Per-pred materialization of a CYCLIC field graph. `o` is virtual on both merge
; preds with lock counts 0 (left) / 1 (right) — a lock mismatch — and `o` and
; `p` form a cycle: o.f = p, p.g = o. The lock mismatch drives per-pred
; materialization of `o` (and its prerequisite `p`) at BOTH preds (4 NewInvs),
; and at each pred the cycle's back edge (p.g = o) must resolve to THAT pred's
; own o-NewInv. The cascade tail at each pred replays both objects' field stores
; in its MatCont (dominated by both per-pred NewInvs), so the back edge resolves
; per-pred via the point-sensitive resolution sub-pass instead of lowering to
; poison. The unbalanced monitorenter is re-emitted at right's o-NewInv.
; See PartialEscapeTransform.cpp applyMaterialize (cyclic-field-materialize) and
; the matching 403_nested_per_pred_field_replay.ll (forward edge only).

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @per_pred_cascade_with_locks(i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock = alloca i64, align 8
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 11111 to ptr), i32 16) to label %oi unwind label %u
oi:
  %p = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 16) to label %fld unwind label %u
fld:
  %of = getelementptr inbounds i8, ptr addrspace(1) %o, i64 0
  store atomic ptr addrspace(1) %p, ptr addrspace(1) %of unordered, align 8
  %pg = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8
  store atomic ptr addrspace(1) %o, ptr addrspace(1) %pg unordered, align 8
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %left, label %right
left:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
right:
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %o, ptr %lock)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  call void @sink(ptr addrspace(1) %p) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

!java-method-compilation = !{}

; CHECK-LABEL: define void @per_pred_cascade_with_locks
; Both objects materialize per-pred (klass 11111 and 22222 each appear on both
; the left and right materialization paths).
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 11111 to ptr)
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 22222 to ptr)
; Exactly four per-pred replayed field stores (two per pred), each using a real
; per-pred NewInv — the back edge p.g = o resolves at each pred to that pred's
; own o-NewInv. (The later merge re-materialization of p stores the merge PHI,
; not a raw %pea.mat, so it is excluded by the value pattern.)
; CHECK-COUNT-4: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; The unbalanced monitorenter is re-emitted at the right pred's materialization.
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock
; CHECK-NOT: poison
; CHECK: ret void
