; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; Three-object cycle (o.f = p, p.g = q, q.h = o) on the per-pred path. The
; cycle's back edge q.h = o (o materialized last in the cascade) resolves to a
; real per-pred NewInv: the cascade tail replays the whole group's field stores
; in its MatCont (dominated by every NewInv), so every edge — forward and back —
; resolves via the point-sensitive resolution sub-pass. See
; PartialEscapeTransform.cpp applyMaterialize (cyclic-field-materialize) and the
; matching 435_mutual_field_perpred.ll.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @self_field_cycle3_perpred(i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 11111 to ptr), i32 32) to label %oi unwind label %u
oi:
  %p = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 22222 to ptr), i32 32) to label %pi unwind label %u
pi:
  %q = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 33333 to ptr), i32 32) to label %fld unwind label %u
fld:
  %of = getelementptr inbounds i8, ptr addrspace(1) %o, i64 0
  store atomic ptr addrspace(1) %p, ptr addrspace(1) %of unordered, align 8
  %pg = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8
  store atomic ptr addrspace(1) %q, ptr addrspace(1) %pg unordered, align 8
  %qh = getelementptr inbounds i8, ptr addrspace(1) %q, i64 16
  store atomic ptr addrspace(1) %o, ptr addrspace(1) %qh unordered, align 8
  br i1 %c, label %left, label %right
left:
  br label %merge
right:
  br label %merge
merge:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  call void @sink(ptr addrspace(1) %p) [ "deopt"(i32 0, i32 0) ]
  call void @sink(ptr addrspace(1) %q) [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

!java-method-compilation = !{}

; CHECK-LABEL: define void @self_field_cycle3_perpred
; All three objects materialize once at the escape point (a live-path cascade
; of 3 sharing one InsertBefore).
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 11111 to ptr)
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 22222 to ptr)
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 33333 to ptr)
; Replayed field stores use real NewInvs — the cycle's back edge q.h = o (o
; materialized last in the cascade) resolves instead of lowering to poison.
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK: store atomic ptr addrspace(1) %pea.mat{{[0-9]*}}, ptr addrspace(1) %pea.matslot{{[0-9]*}} unordered, align 8
; CHECK-NOT: poison
; CHECK: ret void
