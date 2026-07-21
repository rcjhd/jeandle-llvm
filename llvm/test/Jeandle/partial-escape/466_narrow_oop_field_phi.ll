; RUN: opt -S -passes="require<partial-escape-analysis>,partial-escape-transform" %s | FileCheck %s

; A virtual reference is tracked by its wide allocation value even when it was
; stored through a compressed-oop field. Field merges must materialize the
; nested object on the contributing edge, encode the resolved allocation, and
; feed that narrow value to the field PHI.

target datalayout = "e-p:64:64-p1:64:64-p3:32:32"

declare hotspotcc ptr addrspace(1) @jeandle.new_instance(ptr, i32)
declare hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(ptr addrspace(1))
declare hotspotcc ptr addrspace(1) @jeandle.decode_heap_oop(ptr addrspace(3))
declare void @sink(ptr addrspace(1))
declare i32 @__gxx_personality_v0(...)

define void @ordinary_narrow_field_phi(i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  %outer = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
               ptr inttoptr (i64 71001 to ptr), i32 24)
            to label %split unwind label %u

split:
  br i1 %c, label %left, label %right

left:
  %inner = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
               ptr inttoptr (i64 71002 to ptr), i32 16)
            to label %left.store unwind label %u

left.store:
  %encoded = call hotspotcc ptr addrspace(3)
      @jeandle.encode_heap_oop(ptr addrspace(1) %inner)
  %left.field = getelementptr inbounds i8, ptr addrspace(1) %outer, i64 16
  store atomic ptr addrspace(3) %encoded,
      ptr addrspace(1) %left.field unordered, align 4
  br label %merge

right:
  br label %merge

merge:
  %field = getelementptr inbounds i8, ptr addrspace(1) %outer, i64 16
  %narrow = load atomic ptr addrspace(3),
      ptr addrspace(1) %field unordered, align 4
  %wide = call hotspotcc ptr addrspace(1)
      @jeandle.decode_heap_oop(ptr addrspace(3) %narrow)
  call void @sink(ptr addrspace(1) %wide)
  ret void

u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @ordinary_narrow_field_phi(
; Outer is scalar-replaced; inner is materialized only on the left edge.
; CHECK-NOT: i64 71001
; CHECK: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 71002 to ptr)
; CHECK: call hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(ptr addrspace(1) {{%.*}})
; CHECK: = phi ptr addrspace(3)
; CHECK: call void @sink
; CHECK: ret void

define void @case_c_narrow_field_phi(i1 %c)
    gc "hotspotgc" personality ptr @__gxx_personality_v0 {
entry:
  br i1 %c, label %left, label %right

left:
  %outer.left = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
                    ptr inttoptr (i64 72001 to ptr), i32 24)
                 to label %left.inner unwind label %u

left.inner:
  %inner.left = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
                    ptr inttoptr (i64 72002 to ptr), i32 16)
                 to label %left.store unwind label %u

left.store:
  %encoded.left = call hotspotcc ptr addrspace(3)
      @jeandle.encode_heap_oop(ptr addrspace(1) %inner.left)
  %field.left = getelementptr inbounds i8,
      ptr addrspace(1) %outer.left, i64 16
  store atomic ptr addrspace(3) %encoded.left,
      ptr addrspace(1) %field.left unordered, align 4
  br label %merge

right:
  %outer.right = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
                     ptr inttoptr (i64 72001 to ptr), i32 24)
                  to label %right.inner unwind label %u

right.inner:
  %inner.right = invoke hotspotcc ptr addrspace(1) @jeandle.new_instance(
                     ptr inttoptr (i64 72002 to ptr), i32 16)
                  to label %right.store unwind label %u

right.store:
  %encoded.right = call hotspotcc ptr addrspace(3)
      @jeandle.encode_heap_oop(ptr addrspace(1) %inner.right)
  %field.right = getelementptr inbounds i8,
      ptr addrspace(1) %outer.right, i64 16
  store atomic ptr addrspace(3) %encoded.right,
      ptr addrspace(1) %field.right unordered, align 4
  br label %merge

merge:
  %outer = phi ptr addrspace(1)
      [ %outer.left, %left.store ], [ %outer.right, %right.store ]
  %field = getelementptr inbounds i8, ptr addrspace(1) %outer, i64 16
  %narrow = load atomic ptr addrspace(3),
      ptr addrspace(1) %field unordered, align 4
  %wide = call hotspotcc ptr addrspace(1)
      @jeandle.decode_heap_oop(ptr addrspace(3) %narrow)
  call void @sink(ptr addrspace(1) %wide)
  ret void

u:
  %lp = landingpad i64 cleanup
  resume i64 %lp
}

; CHECK-LABEL: define void @case_c_narrow_field_phi(
; CHECK-NOT: i64 72001
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 72002 to ptr)
; CHECK-DAG: invoke hotspotcc{{.*}}@jeandle.new_instance(ptr inttoptr (i64 72002 to ptr)
; CHECK-DAG: call hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(ptr addrspace(1) {{%.*}})
; CHECK-DAG: call hotspotcc ptr addrspace(3) @jeandle.encode_heap_oop(ptr addrspace(1) {{%.*}})
; CHECK: = phi ptr addrspace(3)
; CHECK: call void @sink
; CHECK: ret void

!java-method-compilation = !{}
