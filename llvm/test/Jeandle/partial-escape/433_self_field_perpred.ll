; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Self-referential field (o.f = o) materialized PER-PRED at a merge.
;
; `else` keeps o virtual and stores o into its own field (offset 8); `then`
; escapes o via sink. At `merge` (mixed: then materialized, else virtual) o is
; per-pred materialized at else's terminator. The replayed field store (o.f = o)
; must store the per-pred NewInv into ITS OWN field.
;
; Before the fix the field-replay value was the per-pred placeholder (an
; unparented PHINode); the transform emitted it unresolved as the store value
; (<badref>) and the verifier aborted. The fix records OrigAlloc as the
; field-replay value and resolveMaterializedUses rewrites the store to the
; per-pred NewInv that dominates it.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @self_field_perpred(i1 %c) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16) to label %n unwind label %u
n:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %then, label %else
then:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %merge
else:
  %slot = getelementptr inbounds i8, ptr addrspace(1) %o, i64 8
  store atomic ptr addrspace(1) %o, ptr addrspace(1) %slot unordered, align 8
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

!java-method-compilation = !{}

; CHECK-LABEL: define void @self_field_perpred
; Two per-pred materializations of o (then path + else path).
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr)
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr)
; The self-referential field store: value operand is a real per-pred NewInv
; (pea.mat), never <badref>.
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot unordered, align 8
; merge combines the two per-pred NewInvs in a PHI consumed by sink.
; CHECK: phi ptr addrspace(1)
; CHECK: ret void
