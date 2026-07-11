; RUN: opt -jeandle-pea-enable-allocation-sinking -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s
;
; A DERIVED pointer carried across the back-edge where the derivation is a
; multi-step chain: %t = gep %X, 4 ; %sf = bitcast %t. stripPointerCastsAndOffsets
; accumulates the full byte offset (4), so the re-derivation emits
; gep %pea.mat, 4 — not just offset 0.

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

declare hotspotcc void @jeandle.safepoint_poll()

define void @test_151_carried_chain(i32 %n) gc "hotspotgc" personality ptr @__gxx_personality_v0 {
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
            ptr inttoptr (i64 5555 to ptr), i32 32)
          to label %bcont unwind label %u
bcont:
  %t = getelementptr inbounds i8, ptr addrspace(1) %X, i64 4
  %sf = bitcast ptr addrspace(1) %t to ptr addrspace(1)
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

; CHECK-LABEL: define void @test_151_carried_chain
; CHECK: %psf = phi ptr addrspace(1) [ null, %entry ], [ %pea.matoff, %mat.cont ]
; CHECK: %pea.matoff = getelementptr inbounds i8, ptr addrspace(1) %pea.mat, i64 4
; CHECK: call void @sink(ptr addrspace(1) %psf)
; CHECK-NOT: poison

!java-method-compilation = !{}
