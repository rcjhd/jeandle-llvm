; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s
; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s
; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" -jeandle-dump-pea-stats %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATS

; Allocation sinking is speculative by default: every materialization point must
; be a call or invoke carrying exact deopt state. Objects without such a point
; keep their original allocation and object/monitor operations. Objects that
; never escape still use scalar replacement, lock elimination, and lazy deopt
; reconstruction.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1), ptr)
declare hotspotcc void @jeandle.monitorexit_with_thin_lock(ptr addrspace(1), ptr)
declare hotspotcc void @jeandle.safepoint_poll()
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @escaping_without_deopt_point_keeps_original() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       [ "deopt"(i32 1, i32 1) ]
       to label %n unwind label %u
n:
  %field = getelementptr inbounds i8, ptr addrspace(1) %o, i64 12
  store i32 42, ptr addrspace(1) %field
  call void @sink(ptr addrspace(1) %o)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @escaping_without_deopt_point_keeps_original
; CHECK: %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16) [ "deopt"(i32 1, i32 1) ]
; CHECK-NOT: %pea.mat = invoke
; CHECK: store i32 42, ptr addrspace(1) %field
; CHECK: call void @sink(ptr addrspace(1) %o)

define void @escaping_locked_object_keeps_monitors(i1 %escape) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock = alloca i64, align 8
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 23456 to ptr), i32 16)
       to label %n unwind label %u
n:
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
      ptr addrspace(1) %o, ptr %lock)
  br i1 %escape, label %yes, label %no
yes:
  call void @sink(ptr addrspace(1) %o)
  br label %done
no:
  br label %done
done:
  call hotspotcc void @jeandle.monitorexit_with_thin_lock(
      ptr addrspace(1) %o, ptr %lock)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @escaping_locked_object_keeps_monitors
; CHECK: %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance
; CHECK-NOT: %pea.mat = invoke
; CHECK: call hotspotcc void @jeandle.monitorenter_with_thin_lock(ptr addrspace(1) %o
; CHECK: call void @sink(ptr addrspace(1) %o)
; CHECK: call hotspotcc void @jeandle.monitorexit_with_thin_lock(ptr addrspace(1) %o

define void @non_escaping_object_still_scalar_replaced() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %lock = alloca i64, align 8
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 34567 to ptr), i32 16)
       to label %n unwind label %u
n:
  call hotspotcc void @jeandle.monitorenter_with_thin_lock(
      ptr addrspace(1) %o, ptr %lock)
  %field = getelementptr inbounds i8, ptr addrspace(1) %o, i64 12
  store i32 7, ptr addrspace(1) %field
  call hotspotcc void @jeandle.monitorexit_with_thin_lock(
      ptr addrspace(1) %o, ptr %lock)
  call hotspotcc void @jeandle.safepoint_poll()
       [ "deopt"(i32 4, i32 4, i64 12, ptr addrspace(1) %o) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @non_escaping_object_still_scalar_replaced
; CHECK-NOT: jeandle.new_instance
; CHECK-NOT: jeandle.monitorenter_with_thin_lock
; CHECK-NOT: jeandle.monitorexit_with_thin_lock
; CHECK-NOT: store i32 7
; CHECK: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 4, i32 4, i64 4295491596, i64 34567, i64 1, i64 12, i64 589834, i32 7, i64 12, i64 1) ]

; STATS: ;; PEA stats @escaping_without_deopt_point_keeps_original: NeverEscapes=0 PartiallyEscapes=0 AlwaysEscapes=1
; STATS: ;; PEA stats @escaping_locked_object_keeps_monitors: NeverEscapes=0 PartiallyEscapes=0 AlwaysEscapes=1
; STATS: ;; PEA stats @non_escaping_object_still_scalar_replaced: NeverEscapes=1 PartiallyEscapes=0 AlwaysEscapes=0

!java-method-compilation = !{}
