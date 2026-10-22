# Draft issue for bendlang/bend: the C backend never emits fused multiply-add

Status: draft (2026-10-17), not posted, and not to be posted as a speed issue: timed on an idle machine on 2026-10-18,
the native target gains little and fusion loses (below). Checked for duplicates on 2026-10-18 (searched "FMA", "fused
multiply", "fp-contract", "march", "f32_rewrap"): none. Reproduced on Bend 2.0.37, clang 22.1.8, Core Ultra 7 155H.

Update 2026-10-20 (idle machine, bend2-lab `experiments/2026-10-20-block-b/`): on today's BendTorch code the native
target gains much more than below: one SmolLM2-135M `train 3`, 3 alternated rounds, clang 23.1.3, `-march=native`
-12.5% on 1 thread (41.50 -> 36.33 s) and -6.8% on 16 (15.07 -> 14.05 s), `-mavx2` -9.7% and -3.4%, both well above the
loop-alignment noise (`-falign-loops=64` alone: -2.2% and -1.3%), same bits, still no FMA. The flags question is closed
upstream: PR #1461 (`$CFLAGS` for `bend -o`) was declined on 2026-10-14 (keep the build CLI small; `-o f.c` and clang by
hand). What remains is a small bug, drafted in `cc-silent-fallback.md`. This draft stays unposted; the code-generation
fact below is still true.

Timing (idle machine, the same emitted C built three ways; one SmolLM2-135M AdamW step in pure Bend, 1 thread pinned to a
P-core and 16 threads, 3 alternated rounds; each build gives the same bits on 1 and 16 threads):

| clang flags | 1 thread | 16 threads | `fma_dot` repeated 2,000 times |
|---|---:|---:|---:|
| `-std=c11 -O3` (what `bend -o` does) | 17.27 s | 5.34 s | 947 ms |
| `-std=c11 -O3 -march=native` | 17.05 s | 5.07 s | 934 ms |
| `-std=c11 -O3 -march=native -ffp-contract=fast` | 23.96 s | 6.31 s | 1,790 ms |

The native target is 1-5% faster; the fused build is 18-39% slower on the model and 1.89x slower on the dot product. I
have not found why. What stays true below is the code-generation fact, worth at most a line in a broader issue.

---

**Title:** C backend: `a + x * y` on F32 never becomes a fused multiply-add, even on a CPU that has FMA

## Summary

Every F32 product-and-sum in a Bend program compiles to a separate multiply and add. Two things stop clang from fusing
them:

1. `bend x.bend -o x` calls `clang -std=c11 -O3` with no `-march`, so it targets baseline x86-64, which has no FMA.
2. Even with `-march=native`, the emitted C converts every F32 result to its 32-bit word and back between operations:

   ```c
   o[0] = f32_rewrap(f32_unbox(_acc_0) + f32_unbox(f32_rewrap(f32_unbox(_ra_1) * f32_unbox(_rb_1))));
   ```

   `f32_rewrap`/`f32_unbox` are `union` puns. clang's default `-ffp-contract=on` fuses only within one expression, and
   the pun splits it, so nothing fuses. On top of that, the emitted C starts with `#pragma clang fp contract(off)`, so
   the file itself asks clang not to fuse; only `-ffp-contract=fast` overrides the pragma.

`fma_dot.bend` (a dot product of two 1,048,576-number arrays, one running sum):

| clang flags on the emitted C | `vfmadd` instructions in the binary |
|---|---:|
| `-std=c11 -O3` (what `bend -o` does) | 0 |
| `-std=c11 -O3 -march=native` | 0 |
| `-std=c11 -O3 -march=native -ffp-contract=fast` | 6 |

In a larger program (a 30-layer transformer's training step, built on bend-ml's BendHub packages), the last line fuses 218 sites,
the hot product loop among them (`vfmadd213ps`).

## Why it may matter

- Speed: an FMA does a multiply and an add in one instruction, but measured here it made Bend's code slower (above).
- Accuracy: one rounding per term instead of two. In my measurement it did not change much: the gradients of that
  transformer had the same error against float64 with and without fusion. So this is mostly about speed.
- Determinism: fusion changes bits (the loss's last bits moved), but a fused build was still identical on 1 and 16
  threads, since the order of operations is fixed either way. Projects that need bit-exact results across machines
  (with or without FMA) would want it to be opt-in; some build with `-ffp-contract=off` today for that reason.

## Possible directions

- An opt-in flag for `bend -o` that passes `-march=native -ffp-contract=fast` (or lets the user pass clang flags).
- Emit F32 arithmetic on `f32` values inside an expression and pun only at the boundaries, so `-ffp-contract=on` can
  fuse where the target allows it.
- A Base `F32.fma(a, b, c)` primitive for code that wants it explicitly.

Repro (`fma_dot.bend`, also at https://github.com/nuxyel/bend-ml/blob/3d13885/docs/upstream/fma_dot.bend):

```python
import Base

def dot(k: Nat, +i: U32, acc: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> F32:
  match k ra rb:
    case 0n Tuple{a, +x} Tuple{b, +y}: (acc + (x * y : F32) : F32)
    case 1n+q Tuple{a, +x} Tuple{b, +y}:
      dot(q, (i + 1 : U32), (acc + (x * y : F32) : F32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def main() -> IO(Unit):
  IO.print(F32.show(dot(1048575n, 0, 0.0, Array.get(F32, Array.new(F32, 20n, 0.5), 0), Array.get(F32, Array.new(F32, 20n, 0.25), 0))))
```

```sh
bend fma_dot.bend -o fma_dot.c
clang -std=c11 -O3 -march=native -ffp-contract=fast fma_dot.c -lpthread -lm -o fma_dot && objdump -d fma_dot | grep -c vfmadd
```
