; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s --check-prefixes=COMMON,DEFAULT
; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s --check-prefixes=COMMON,AGGRESSIVE

; Edge cases for live-path materialization deopt-anchor selection.
;
; Covered here:
;  1. escape point deopt state wins over stale allocation deopt state;
;  2. multiple safe prior anchors choose the nearest one;
;  3. a non-Call escape can still materialize at the nearest safe prior anchor;
;  4. a prior anchor is rejected after intervening virtual-object activity;
;  5. an invoke escape point carrying deopt state is also eligible.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc void @jeandle.safepoint_poll()
declare void @sink(ptr addrspace(1))
declare void @sink_may_throw(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @escape_deopt_overrides_allocation_deopt() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 12345 to ptr), i32 16)
       [ "deopt"(i32 11, i32 11) ]
       to label %n unwind label %u
n:
  call void @sink(ptr addrspace(1) %o) [ "deopt"(i32 22, i32 22) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; COMMON-LABEL: define void @escape_deopt_overrides_allocation_deopt
; COMMON: %pea.mat = invoke {{.*}}@jeandle.new_instance(ptr inttoptr (i64 12345 to ptr), i32 16) [ "deopt"(i32 22, i32 22) ]
; COMMON-NOT: "deopt"(i32 11, i32 11)
; COMMON: call void @sink(ptr addrspace(1) %pea.mat) [ "deopt"(i32 22, i32 22) ]

define void @nearest_prior_deopt_anchor_wins() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 23456 to ptr), i32 16)
       [ "deopt"(i32 51, i32 51) ]
       to label %n unwind label %u
n:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 52, i32 52) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 53, i32 53) ]
  call void @sink(ptr addrspace(1) %o)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; DEFAULT-LABEL: define void @nearest_prior_deopt_anchor_wins
; DEFAULT: %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance
; DEFAULT-NOT: %pea.mat = invoke
; DEFAULT: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 52, i32 52) ]
; DEFAULT: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 53, i32 53) ]
; DEFAULT: call void @sink(ptr addrspace(1) %o)
; AGGRESSIVE-LABEL: define void @nearest_prior_deopt_anchor_wins
; AGGRESSIVE: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 52, i32 52) ]
; AGGRESSIVE-NEXT: %pea.mat = invoke hotspotcc nonnull "java-klass"="23456" "java-klass-exact" ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 23456 to ptr), i32 16) [ "deopt"(i32 53, i32 53) ]
; AGGRESSIVE-NEXT: to label %mat.cont unwind label %u
; AGGRESSIVE: mat.cont:
; AGGRESSIVE-NEXT: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 53, i32 53) ]
; AGGRESSIVE-NEXT: call void @sink(ptr addrspace(1) %pea.mat)

define void @unsafe_prior_anchor_after_virtual_store() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 54321 to ptr), i32 16)
       [ "deopt"(i32 41, i32 41) ]
       to label %n unwind label %u
n:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 42, i32 42) ]
  %slot = getelementptr inbounds i8, ptr addrspace(1) %o, i64 8
  store atomic i32 7, ptr addrspace(1) %slot unordered, align 4
  call void @sink(ptr addrspace(1) %o)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; COMMON-LABEL: define void @unsafe_prior_anchor_after_virtual_store
; COMMON-NOT: %pea.mat = invoke
; COMMON: %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 54321 to ptr), i32 16) [ "deopt"(i32 41, i32 41) ]
; COMMON-NOT: %pea.mat = invoke
; COMMON: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 42, i32 42) ]
; COMMON-NOT: %pea.mat = invoke
; COMMON: store atomic i32 7
; COMMON-NOT: %pea.mat = invoke
; COMMON: call void @sink(ptr addrspace(1) %o)

define ptr addrspace(1) @return_escape_uses_prior_deopt_anchor() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 67890 to ptr), i32 16)
       [ "deopt"(i32 31, i32 31) ]
       to label %n unwind label %u
n:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 32, i32 32) ]
  ret ptr addrspace(1) %o
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; DEFAULT-LABEL: define ptr addrspace(1) @return_escape_uses_prior_deopt_anchor
; DEFAULT: %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance
; DEFAULT-NOT: %pea.mat = invoke
; DEFAULT: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 32, i32 32) ]
; DEFAULT-NEXT: ret ptr addrspace(1) %o
; AGGRESSIVE-LABEL: define ptr addrspace(1) @return_escape_uses_prior_deopt_anchor
; AGGRESSIVE: n:
; AGGRESSIVE-NEXT: %pea.mat = invoke hotspotcc nonnull "java-klass"="67890" "java-klass-exact" ptr addrspace(1) @jeandle.new_instance(ptr inttoptr (i64 67890 to ptr), i32 16) [ "deopt"(i32 32, i32 32) ]
; AGGRESSIVE-NEXT: to label %mat.cont unwind label %u
; AGGRESSIVE: mat.cont:
; AGGRESSIVE-NEXT: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 32, i32 32) ]
; AGGRESSIVE-NEXT: ret ptr addrspace(1) %pea.mat

define void @invoke_escape_uses_its_deopt_state() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %o = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 78901 to ptr), i32 16)
       to label %n unwind label %u
n:
  %field = getelementptr inbounds i8, ptr addrspace(1) %o, i64 12
  store i32 9, ptr addrspace(1) %field
  invoke void @sink_may_throw(ptr addrspace(1) %o)
      [ "deopt"(i32 61, i32 61) ]
      to label %done unwind label %u
done:
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; COMMON-LABEL: define void @invoke_escape_uses_its_deopt_state
; COMMON-NOT: %o = invoke
; COMMON: %pea.mat = invoke {{.*}}@jeandle.new_instance(ptr inttoptr (i64 78901 to ptr), i32 16) [ "deopt"(i32 61, i32 61) ]
; COMMON: store atomic i32 9
; COMMON: invoke void @sink_may_throw(ptr addrspace(1) %pea.mat) [ "deopt"(i32 61, i32 61) ]

define void @prior_anchor_rejected_after_nested_state_change() gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %outer = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
                 ptr inttoptr (i64 89012 to ptr), i32 16)
           to label %alloc_inner unwind label %u
alloc_inner:
  %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
                 ptr inttoptr (i64 89013 to ptr), i32 16)
           to label %n unwind label %u
n:
  %outer_slot = getelementptr inbounds i8, ptr addrspace(1) %outer, i64 8
  store atomic ptr addrspace(1) %inner, ptr addrspace(1) %outer_slot unordered, align 8
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 71, i32 71) ]
  %inner_slot = getelementptr inbounds i8, ptr addrspace(1) %inner, i64 12
  store atomic i32 5, ptr addrspace(1) %inner_slot unordered, align 4
  call void @sink(ptr addrspace(1) %outer)
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; COMMON-LABEL: define void @prior_anchor_rejected_after_nested_state_change
; COMMON-NOT: %pea.mat = invoke
; COMMON: %outer = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance
; COMMON: %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance
; COMMON: call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 71, i32 71) ]
; COMMON: store atomic i32 5
; COMMON: call void @sink(ptr addrspace(1) %outer)

!java-method-compilation = !{}
