; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s
;
; TWO header PHIs each carry a different DERIVED pointer of the same loop-body
; object: %sf1 = gep %X, 8 and %sf2 = gep %X, 16. Each carried incoming is
; re-derived independently at its own byte offset over the materialized base.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_152_carried_multiple(i32 %n) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %hdr
hdr:
  %i = phi i32 [ 0, %entry ], [ %i1, %latch ]
  %psf1 = phi ptr addrspace(1) [ null, %entry ], [ %sf1, %latch ]
  %psf2 = phi ptr addrspace(1) [ null, %entry ], [ %sf2, %latch ]
  %c = icmp slt i32 %i, %n
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %c, label %body, label %exit
body:
  %X = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
            ptr inttoptr (i64 5555 to ptr), i32 32)
          to label %bcont unwind label %u
bcont:
  %sf1 = getelementptr inbounds i8, ptr addrspace(1) %X, i64 8
  %sf2 = getelementptr inbounds i8, ptr addrspace(1) %X, i64 16
  store atomic i32 %i, ptr addrspace(1) %sf1 unordered, align 4
  store atomic i32 %i, ptr addrspace(1) %sf2 unordered, align 4
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %latch
latch:
  %i1 = add i32 %i, 1
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %hdr
exit:
  %ec = icmp eq ptr addrspace(1) %psf1, null
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br i1 %ec, label %done, label %obs
obs:
  call void @sink(ptr addrspace(1) %psf1) [ "deopt"(i32 0, i32 0) ]
  call void @sink(ptr addrspace(1) %psf2) [ "deopt"(i32 0, i32 0) ]
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  br label %done
done:
  call hotspotcc void @jeandle.safepoint_poll() [ "deopt"(i32 0, i32 0) ]
  ret void
u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @test_152_carried_multiple
; Both offsets are re-derived over the materialized base.
; CHECK: getelementptr inbounds i8, ptr addrspace(1) %pea.mat, i64 8
; CHECK: getelementptr inbounds i8, ptr addrspace(1) %pea.mat, i64 16
; CHECK: call void @sink(ptr addrspace(1) %psf1)
; CHECK: call void @sink(ptr addrspace(1) %psf2)
; CHECK-NOT: poison

!java-method-compilation = !{}
