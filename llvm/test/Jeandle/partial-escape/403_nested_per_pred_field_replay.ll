; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Nested per-pred materialization. `outer` is virtual on both merge preds with
; lock counts 0 (left) / 1 (right) — a lock mismatch — AND outer.field (offset
; 8) holds a SECOND virtual object `inner`. At the merge, the lock mismatch
; drives materializeAndBuildPhi(outer), which per-pred-materializes outer at
; left and right; the recursive prerequisite inside each per-pred materialize
; then per-pred-materializes `inner` at the same pred.
;
; The field store replayed by outer's materialize at each pred must store THAT
; pred's OWN inner-NewInv into that pred's outer-NewInv. The field-replay value
; is inner's OrigAlloc on both the live and per-pred paths; the transform's
; point-sensitive resolution sub-pass (resolveMaterializedUses) rewrites each
; replayed store to the inner NewInv that dominates it — left's store to left's
; inner-NewInv, right's to right's — recovering per-pred distinctness via
; dominance (Jeandle's analog of Graal getAliasAndResolve). No eager per-effect
; substitution is used: it would be last-write-wins across the two per-pred
; inner-NewInvs and miscompile this case.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_nested_per_pred_field_replay(i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock = alloca i64, align 8
  %outer = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
              ptr inttoptr (i64 11111 to ptr), i32 16)
           to label %oi unwind label %u
oi:
  %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
              ptr inttoptr (i64 22222 to ptr), i32 16)
           to label %fld unwind label %u
fld:
  %ofs = getelementptr inbounds i8, ptr addrspace(1) %outer, i64 8
  store atomic ptr addrspace(1) %inner, ptr addrspace(1) %ofs unordered, align 8
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %branch
branch:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %left, label %right
left:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
right:
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
              ptr addrspace(1) %outer, ptr %lock)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  call void @sink(ptr addrspace(1) %outer) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; outer (11111) and inner (22222) each materialize per-pred; a materializedValuePhi
; selects outer's per-pred NewInvs; the un-elided enter on right snaps to right's
; outer-NewInv; sink consumes the PHI. Verifier-clean (no abort).
; CHECK-LABEL: define void @test_nested_per_pred_field_replay
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 11111 to ptr)
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 22222 to ptr)
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %{{[^,]+}}, ptr %lock)
; CHECK: %[[PHI:[A-Za-z0-9._]+]] = phi ptr addrspace(1)
; CHECK: call void @sink(ptr addrspace(1) %[[PHI]])

!java-method-compilation = !{}
