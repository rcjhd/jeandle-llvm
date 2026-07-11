; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Lock-stack identity mismatch at a diamond where each arm enters the SAME
; virtual %o via a DISTINCT call site, so the two arms carry DIFFERENT
; proxy depths (RPO visits %t before %e, so lock_t gets depth 0 and lock_e
; depth 1). The per-element compare
;   S[i].Call == RefStack[i].Call && S[i].BytecodeDepth == RefStack[i].BytecodeDepth
; sees a Call mismatch (lock_t vs lock_e) — and a BytecodeDepth mismatch
; (0 vs 1) — and routes to per-pred materialise. (The Call mismatch alone
; suffices; the depth difference is incidental.)

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1), ptr)
declare hotspotcc void @jeandle.monitorexit_with_lightweight_lock(ptr addrspace(1), ptr)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_lockdepth_per_pred_diff_depth(i1 %cond) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock_t = alloca i64, align 8
  %lock_e = alloca i64, align 8
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 8765 to ptr), i32 16)
       to label %dispatch unwind label %u
dispatch:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %cond, label %t, label %e
t:
  ; Then-arm: this enter is the only one on %o on this path. RPO visits %t
  ; first, so the proxy assigns it depth 0.
  call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(
                  ptr addrspace(1) %o, ptr %lock_t)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
e:
  ; Else-arm: a DISTINCT call site on %o. RPO visits %e after %t, so the
  ; proxy assigns it depth 1. Different call site, different depth.
  call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(
                  ptr addrspace(1) %o, ptr %lock_e)
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
merge:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; %o is materialised at each pred (depth-mismatch routes
; the merge through per-pred materialise instead of the legacy bail).
; The original allocation is replaced by the per-pred materialised invokes.
; CHECK-LABEL: define void @test_lockdepth_per_pred_diff_depth
; Per-pred materialise: TWO new_instance invokes survive, one per arm.
; CHECK-DAG: %[[MATT:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 8765 to ptr), i32 16)
; CHECK-DAG: %[[MATE:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 8765 to ptr), i32 16)
; Both pred-side enter calls survive on per-pred materialised pointers.
; CHECK-DAG: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) {{.*}}, ptr %lock_t)
; CHECK-DAG: call hotspotcc void @jeandle.monitorenter_with_lightweight_lock(ptr addrspace(1) {{.*}}, ptr %lock_e)
; The merge block synthesises a phi over the two pred materialised pointers
; and the sink uses it.
; CHECK: phi ptr addrspace(1)
; CHECK: call void @sink(ptr addrspace(1)

!java-method-compilation = !{}
