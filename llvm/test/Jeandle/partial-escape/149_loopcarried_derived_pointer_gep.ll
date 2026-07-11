; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s
;
; A DERIVED pointer (a GEP of a loop-body alloc) carried across the back-edge by
; a header PHI. Here %sf = getelementptr %X, 8 is the carried value.
;
; Carrying a derived pointer is harder than carrying the OBJECT pointer itself
; (see 142-148): the GEP is defined inside the body (before the latch), so the
; back-edge materialization (which re-emits the object at the latch) cannot
; rewrite the GEP base — it does not dominate the materialization point. The
; fix mirrors Graal getAliasAndResolve + setPhiInput (re-derive the incoming
; from the per-predecessor materialized object state at the merge), extended to
; replay the byte offset over the freshly-materialized base: the carried PHI's
; back-edge incoming becomes a GEP of the materialized object at offset 8.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_149_carried_gep(i32 %n) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %hdr
hdr:
  %i = phi i32 [ 0, %entry ], [ %i1, %latch ]
  %psf = phi ptr addrspace(1) [ null, %entry ], [ %sf, %latch ]
  %c = icmp slt i32 %i, %n
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %body, label %exit
body:
  %X = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 5555 to ptr), i32 16)
          to label %bcont unwind label %u
bcont:
  %sf = getelementptr inbounds i8, ptr addrspace(1) %X, i64 8
  store atomic i32 %i, ptr addrspace(1) %sf unordered, align 4
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %latch
latch:
  %i1 = add i32 %i, 1
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %hdr
exit:
  %ec = icmp eq ptr addrspace(1) %psf, null
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %ec, label %done, label %obs
obs:
  call void @sink(ptr addrspace(1) %psf) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %done
done:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; The carried PHI's back-edge incoming is a GEP of the materialized object at
; offset 8 (re-derived at the back-edge), NOT poison and NOT the eliminated
; original body GEP. The body GEP %sf is dead-code swept away.
; CHECK-LABEL: define void @test_149_carried_gep
; CHECK: %psf = phi ptr addrspace(1) [ null, %entry ], [ %pea.matoff, %mat.cont ]
; CHECK: %pea.mat = invoke
; CHECK: %pea.matoff = getelementptr inbounds i8, ptr addrspace(1) %pea.mat, i64 8
; CHECK: call void @sink(ptr addrspace(1) %psf)
; CHECK-NOT: poison

!java-method-compilation = !{}
