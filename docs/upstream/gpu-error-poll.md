# Draft issue for bendlang/bend: the error poll in every device loop slows tight loops on CUDA

Status: posted as https://github.com/bendlang/bend/issues/1776 (2026-10-19, by Claude Code with Renan's go-ahead), in the Bug form's fields. Checked for duplicates on 2026-10-18 (searched "err_spun",
"WL_SPIN", "YIELD", "error poll", "GPU slow loop") and again on 2026-10-19 ("poll", "err_spun", "gpu error"): none;
the closest is #1143 (Metal, heap_free's error check, closed). Still in upstream main at 881cb2e5 (2026-10-19; `err_spun`
and `WL_SPIN` unchanged, and `split-poll.patch` applies as is). Every table was rerun on an idle machine on 2026-10-19,
against main 881cb2e5, which has PR #1602 (bend2-lab `experiments/2026-10-19-poll-idle/`; the busy-machine study it
replaces is `2026-10-18-device-code/`). #1602 changed the picture: on main the poll no longer costs anything on loops
that read memory, so the earlier sections on the probe's loop and the counted shape are gone. Metal uses the same poll
on the device; I have no Metal machine to measure it.

---

**Title:** CUDA: the error poll in every flat loop (`err_spun`) makes ptxas yield on each iteration; tight loops run up
to 20x slower

## Summary

Every flat loop Bend compiles for the device (`spin_N`, opened by `WL_SPIN`) starts with
`err_spun(e.mem, &wpoll)`: a counter and, every 4,096 steps, a load of the heap's error word. In the SASS (read from 2.0.37's build; main emits the same
C for it), that loop gets a `YIELD` and a convergence barrier (`BSSY`/`BSYNC`) on every iteration, and it is not unrolled. On a loop with little
work per step this dominates: a one-leaf loop of `acc * 0.999 + x` takes ~70 ns a step on an RTX 4050, against ~3.4 ns
for the same loop handwritten in CUDA. With the poll removed from the emitted C, Bend's loop runs as fast as the
handwritten one. Splitting the loop keeps the check and most of the speed; a patch is below.

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
bun bend2/main.ts flat.bend -o flat.c      # or `bend flat.bend -o flat.c` with 2.0.37
sed 's/^#define err_spun(H, n) ((++\*(n) & 4095) == 0 \&\& err_seen(H))$/#define err_spun(H, n) (0)/' flat.c > flat_np.c
# build both the way `bend -o` does (clang -DBEND_CUDA=1 ... -std=c11 -O3 ... -lcuda -lnvrtc), then ./x --gpu-build
```

Both print 57342550. The tables time the bang alone (`IO.now` around it, after a warm-up bang), five alternated rounds
on an idle machine, min-max. Main 881cb2e5 (with #1602), RTX 4050 Laptop (sm_89), driver 615.78.08, CUDA 12.9; every
build prints the same results, on the GPU and with `--gpu off`:

| program | main | poll removed | ratio |
|---|---:|---:|---:|
| the loop above, 1 leaf, 2M steps | 136-138 ms | 7 ms | ~20x |
| the loop above, 16,384 leaves, 2M steps | 129-132 ms | 15-16 ms | 8.5x |
| tiled matmul 512x512x4096, 8x4 block a leaf, 50 products | 172-177 ms | 149-152 ms | 1.16x |
| tiled matmul 512x512x4096, 4x4 block a leaf, 50 products | 230-237 ms | 210-218 ms | 1.09x |
| 16,384 leaves, interleaved reads of one shared Array | 112-114 ms | 112-115 ms | 1.0x |
| 16,384 leaves, 2M reads each of a 32-number Array (whole program) | 0.92-0.93 s | 0.88-0.91 s | ~1.0x |

The same loop handwritten in CUDA takes 14.1-14.3 ms for 16,384 threads, so without the poll Bend matches it. Loops that
wait on memory hide the poll; loops of arithmetic pay for it.

## The cost is the poll's shape, not its load

In a handwritten CUDA kernel with the same loop (one thread / 16,384 threads, ns per step):

| poll | 1 thread | 16,384 |
|---|---:|---:|
| none | 3.4 | 7.1 |
| `if ((++n & 4095) == 0 && *err != 0) return;` in the loop (Bend's shape), `volatile` load | 16.7 | 30.5 |
| the same with `__ldcg` or `ld.relaxed.gpu` | 16.7 | 30.4-30.9 |
| split loop: 4,096 steps with no check, then one check | 3.4 | 7.2 |

So the check can stay: emitting a spin as an inner loop of up to 4,096 steps with no poll, inside an outer loop that
polls once, costs nothing measurable here. On the host `err_seen` is `DEVICE && ...`, so CPU builds do not pay.

## A fix that keeps the poll

I tried the split shape in `bend2/comp.ts` (written against e5527d5b, applies to 881cb2e5 unchanged): on the device,
each `spin_N`, each spin segment and the work loop run their body in an inner loop of up to 4,096 steps with no poll,
inside an outer loop that polls once. A `break` leaves the inner loop with steps left, which also ends the outer one;
`continue` stays in the inner loop. The host keeps its macros, so CPU builds do not change.

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

Built with `bun bend2/main.ts`, every program below prints the same results as main's build, on the GPU and with
`--gpu off`, and a lane-stack overflow inside a bang is still reported. Same machine and rounds:

| program | main | with the split | poll removed |
|---|---:|---:|---:|
| flat loop, 1 leaf, 2M steps | 136-138 ms | 19-20 ms | 7 ms |
| flat loop, 16,384 leaves, 2M steps | 129-132 ms | 36-37 ms | 15-16 ms |
| tiled matmul 512², 8x4 a leaf, 50 products | 172-177 ms | 157-161 ms | 149-152 ms |
| tiled matmul 512², 4x4 a leaf, 50 products | 230-237 ms | 222-227 ms | 210-218 ms |
| interleaved reads of one shared Array | 112-114 ms | 106-107 ms | 112-115 ms |
| 2M reads a leaf of a 32-number Array (whole program) | 0.92-0.93 s | 0.89-0.90 s | 0.88-0.91 s |

It keeps most of the gain on the arithmetic loops and is slower than main nowhere. The flat loops do not reach the
no-poll time: the split still runs the outer loop and its test every 4,096 steps, and I have not looked further at what
nvcc makes of it.

The change adds 17 lines to `comp.ts`, which is at its token cap; a shorter form is surely possible. I have not opened a
pull request; this is to show the shape works.
