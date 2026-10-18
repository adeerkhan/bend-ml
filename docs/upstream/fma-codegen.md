# Draft issue for bendlang/bend: the C backend never emits fused multiply-add

Status: draft (2026-10-17), not posted; Renan posts it. Not checked for duplicates yet (search "FMA", "fp-contract",
"fused multiply" before posting). Reproduced on Bend 2.0.37, clang 22.1.8, Core Ultra 7 155H. Speed not measured yet:
measure the products with and without fusion on an idle machine before posting, or post it as a question.

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
   the pun splits it, so nothing fuses.

`fma_dot.bend` (a dot product of two 1,048,576-number arrays, one running sum):

| clang flags on the emitted C | `vfmadd` instructions in the binary |
|---|---:|
| `-std=c11 -O3` (what `bend -o` does) | 0 |
| `-std=c11 -O3 -march=native` | 0 |
| `-std=c11 -O3 -march=native -ffp-contract=fast` | 6 |

In a larger program (a 30-layer transformer's training step built with BendHub packages), the last line fuses 218 sites,
the hot product loop among them (`vfmadd213ps`).

## Why it may matter

- Speed: an FMA does a multiply and an add in one instruction. I have not measured the gain on Bend's products yet.
- Accuracy: one rounding per term instead of two. In my measurement it did not change much: the gradients of that
  transformer had the same error against float64 with and without fusion. So this is mostly about speed.
- Determinism: fusion changes bits (the loss's last bits moved), but a fused build was still identical on 1 and 16
  threads, since the order of operations is fixed either way. Projects that need bit-exact results across machines
  (with or without FMA) would want it to be opt-in, which is how Stelliferous builds today (`-ffp-contract=off`).

## Possible directions

- An opt-in flag for `bend -o` that passes `-march=native -ffp-contract=fast` (or lets the user pass clang flags).
- Emit F32 arithmetic on `f32` values inside an expression and pun only at the boundaries, so `-ffp-contract=on` can
  fuse where the target allows it.
- A Base `F32.fma(a, b, c)` primitive for code that wants it explicitly.

Files: `fma_dot.bend` (here).
