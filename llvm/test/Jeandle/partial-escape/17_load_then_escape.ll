; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; PEA: alloc + store + load (folded to the stored constant) + use of
; the load + escape. The load fold survives the escape: %v becomes the
; constant 99 (so use_int gets a constant, consumed before the escape), and
; the object is materialized at the ESCAPE POINT (the @sink call) — Graal's
; materializeBefore=node — so the new invoke sits just before the sink. The
; field store is replayed at the top of the materialization continuation,
; ahead of the sink. (Escape-point placement: OrigAlloc uses are resolved
; per-point by the transform, so the materialize need not dominate earlier
; in-block references.)

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @use_int(i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @test_load_then_escape() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       to label %n unwind label %u
n:
  %s = getelementptr inbounds i8, ptr addrspace(1) %o, i64 8
  store atomic i32 99, ptr addrspace(1) %s unordered, align 4
  %v = load atomic i32, ptr addrspace(1) %s unordered, align 4
  call void @use_int(i32 %v)
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_load_then_escape
; CHECK: call void @use_int(i32 99)
; CHECK: %[[MAT:[A-Za-z0-9._]+]] = invoke hotspotcc{{.*}}ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16)
; CHECK-NEXT: to label %{{.*}} unwind label %{{.*}}
; CHECK: store atomic i32 99, ptr addrspace(1) %{{.*}} unordered, align {{[0-9]+}}
; CHECK: call void @sink(ptr addrspace(1) %[[MAT]])

!java-method-compilation = !{}
