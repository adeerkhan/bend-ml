# Draft issue for bendlang/bend: the error poll in every device loop slows tight loops 3.5-20x on CUDA

Status: draft (2026-10-18), not posted; Renan posts it. Checked for duplicates on 2026-10-18 (searched "err_spun",
"WL_SPIN", "YIELD", "error poll", "GPU slow loop"): none; the closest is #1143 (Metal, heap_free's error check, closed).
Still in upstream main at e5527d5b (`err_spun` and `WL_SPIN` unchanged). Metal uses the same poll on the device; I have
no Metal machine to measure it. The times were taken with the CPU in use and the GPU idle, on driver 610.57.04 (first
tables) and 615.78.08 (the patch); Renan posts it after every table here is rerun on an idle machine. The tested
patch is below in full.

---

**Title:** CUDA: the error poll in every flat loop (`err_spun`) makes ptxas yield on each iteration; tight loops run
3.5-20x slower

## Summary

Every flat loop Bend compiles for the device (`spin_N`, opened by `WL_SPIN`) starts with
`err_spun(e.mem, &wpoll)`: a counter and, every 4,096 steps, a load of the heap's error word. In the SASS, that loop gets
a `YIELD` and a convergence barrier (`BSSY`/`BSYNC`) on every iteration, and it is not unrolled. On a loop with little
work per step this dominates: a one-leaf loop of `acc * 0.999 + x` takes ~70 ns a step on an RTX 4050 (2.5 GHz), against
~4 ns for the same loop handwritten in CUDA. With the poll removed from the emitted C, Bend's loop runs as fast as the
handwritten one.

## Repro

```python
import Base

def flat_l(k: Nat, +x: F32, acc: F32) -> F32:
  match k:
    case 0n: acc
    case 1n+q: flat_l(q, x, ((acc * 0.999 : F32) + x : F32))

def flat_tree(d: Nat, +k: Nat, +i: U32) -> F32:
  match d:
    case 0n: flat_l(k, U32.to_f32(U32.and(i, 7)), 0.0)
    case 1n++q:
      a b = flat_tree(q, k, (i * 2 : U32)) flat_tree(q, k, ((i * 2 : U32) + 1 : U32))
      (a + b : F32)

def main() -> IO(Unit):
  IO.print(F32.show(flat_tree!(14n, 2000000n, 0)))
```

```sh
bend flat.bend -o flat.c
sed 's/^#define err_spun(H, n) ((++\*(n) & 4095) == 0 \&\& err_seen(H))$/#define err_spun(H, n) (0)/' flat.c > flat_np.c
# build both the way `bend -o` does (clang -DBEND_CUDA=1 ... -std=c11 -O3 ... -lcuda -lnvrtc), then ./x --gpu-build
```

Both print 57342550; `/usr/bin/time` gives 0.36-0.40 s and 0.15-0.17 s, of which ~0.13 s is startup. The table below
times the bang alone (`IO.now` around it, after a warm-up bang).

Bend 2.0.37, RTX 4050 Laptop (sm_89), driver 610.57.04, CUDA 12.9; the same sums in every case:

| program | as built | poll removed | ratio |
|---|---:|---:|---:|
| the loop above, 1 leaf, 2M steps | 139-172 ms | 7 ms | ~20x |
| the loop above, 16,384 leaves, 2M steps | 218 ms | 15 ms | 14.5x |
| 16,384 leaves, 2M reads each of a 32-number Array | 2.0 s | 1.07 s | 1.9x |
| 16,384 leaves, interleaved reads of one shared Array | 169 ms | 114 ms | 1.48x |
| tiled matmul 512x512x4096, 8x4 block a leaf | 174 ms | 150 ms | 1.16x |
| tiled matmul 512x512x4096, 4x4 block a leaf | 217 ms | 242 ms | 0.90x |

The same loop handwritten in CUDA takes 14.2 ms for 16,384 threads, so without the poll Bend matches it. The last row
is a loss: one matmul got 10% slower with the poll gone (I have not looked at why).

## The cost is the poll's shape, not its load

In a handwritten CUDA kernel with the same loop (one thread / 16,384 threads, ns per step):

| poll | 1 thread | 16,384 |
|---|---:|---:|
| none | 4.1 | 7.2 |
| `if ((++n & 4095) == 0 && *err != 0) return;` in the loop (Bend's shape), `volatile` load | 16.6 | 30.7 |
| the same with `__ldcg` or `ld.relaxed.gpu` | 16.6 | 30.8 |
| split loop: 4,096 steps with no check, then one check | 3.4 | 7.3 |

So the check can stay: emitting a spin as an inner loop of up to 4,096 steps with no poll, inside an outer loop that
polls once, costs nothing measurable here. On the host `err_seen` is `DEVICE && ...`, so CPU builds do not pay.

## A fix that keeps the poll

I tried the split shape in `bend2/comp.ts` (against main e5527d5b): on the device, each `spin_N`, each spin segment and the
work loop run their body in an inner loop of up to 4,096 steps with no poll, inside an outer loop that polls once. A
`break` leaves the inner loop with steps left, which also ends the outer one; `continue` stays in the inner loop. The
host keeps its macros, so CPU builds do not change.

```diff
@@ emit_native (spin_N)
-    ...seg_take(seg), "WL_SPIN"], 1),
-    ...seg_text(seg.lines, 2), "  break;", "  }",
+    ...seg_take(seg), "WL_SPIN2"], 1),
+    ...seg_text(seg.lines, 2), "  break;", "  }", "  WL_SPIN2_END",
@@ compile_book (spin segments)
-      "WL_OPEN", ...seg.spin ? ["WL_SPIN", ...seg.lines, "WL_SPUN"]
+      "WL_OPEN", ...seg.spin ? ["WL_SPIN2", ...seg.lines, "WL_SPUN2"]
@@ after WL_AGAIN
+#if DEVICE
+#define WL_SPIN2     for (u32 wfirst = 1;; wfirst = 0) { \
+  if (!wfirst && err_seen(e.mem)) { return 0; } \
+  u32 wleft = 4096; for (; wleft != 0; wleft -= 1) {
+#define WL_SPIN2_END if (wleft != 0) { break; } }
+#define WL_SPUN2     } if (wleft != 0) { break; } } break;
+#else
+#define WL_SPIN2     WL_SPIN
+#define WL_SPIN2_END
+#define WL_SPUN2     WL_SPUN
+#endif
@@ work_loop (device)
-  WL_SPIN
+  WL_SPIN2
   switch (fid) {
@@ end of work_loop
   }
+  WL_SPIN2_END
+  return 0;
 }
```

Built from main with `bun bend2/main.ts`, every program below prints the same results as main's build, on the GPU and
with `--gpu off`, and a lane-stack overflow inside a bang is still reported. Three alternated rounds (driver 615.78.08):

| program | main | with the split | poll removed (2.0.37) |
|---|---:|---:|---:|
| flat loop, 1 leaf, 2M steps | 140-172 ms | 19 ms | 7 ms |
| flat loop, 16,384 leaves, 2M steps | 219-223 ms | 42-43 ms | 15 ms |
| interleaved reads of one shared Array | 169-171 ms | 131-133 ms | 114 ms |
| matrix-vector 16,384 x 4,096, steady state | 3.1-3.3 ms | 2.67-2.72 ms | ~2.5 ms |
| tiled matmul 512², 8x4 a leaf, 50 products | 173-175 ms | 157-160 ms | 150 ms |
| tiled matmul 512², 4x4 a leaf, 50 products | 217-221 ms | 216-219 ms | 242 ms |
| 2M reads a leaf of a 32-number Array | 2.0 s | 2.0 s | 1.07 s |

It keeps most of the gain and loses nowhere. The last row does not move, and Nsight Compute shows why: the three builds
make the same 102.5 M load requests, but without the poll that loop's only exit is its `Nat` fuel, so nvcc knows the
trip count and issues 16 loads back to back (546 M warp instructions, 93 ms); with the split the inner loop has two exits
(the budget and the fuel), a test sits between consecutive loads and one load is in flight at a time (1,000 M, 190 ms;
main: 1,845 M, 184 ms). For a spin counted by a `Nat`, the inner budget could be the smaller of 4,096 and the steps
left, which would give the compiler one exit again; I have not tried it.

Nsight Compute on main's matmul shows the poll in the
hot loop's SASS (`YIELD`, `BSSY`/`BSYNC` every K step) among 83 instructions a step, against ~45 in the same loop
handwritten in CUDA.

The change adds 17 lines to `comp.ts`, which is at its token cap; a shorter form is surely possible. I have not opened a
pull request; this is to show the shape works.
