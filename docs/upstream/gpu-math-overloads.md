# Draft issue for bendlang/bend: on CUDA, F32's math primitives give other bits than on the host

Status: not posted (2026-10-19): a duplicate of #1305/#1306 (F32.sin on CPU and GPU), closed as a known limit ("one shared software implementation ... would be slower on the GPU"). Rechecked on main 881cb2e5: the same counts. Checked on 2026-10-18 (searched "exp CUDA", "F32.exp",
"transcendental", "math bits", "expf"): none. Bend 2.0.37, RTX 4050 Laptop, driver 615.78.08, NVRTC 12.9, clang 22.1.8.
Repro: `gpu_math_bits.bend` (next to this file).

---

**Title:** CUDA: `F32.exp`, `log`, `tanh`, `sin`, `cos` and `pow` give other bits on the GPU than on the host (C++ picks
the float overloads)

## Summary

The C that `bend -o` emits computes `F32.exp(x)` as `(f32)exp(f32_unbox(x))`. On the host (C) that is libm's double
`exp`, rounded to F32. In the device program (CUDA C++, NVRTC) the argument is a `float`, so overload resolution picks
`exp(float)`, CUDA's single-precision `expf` (up to 2 ulp). The same Bend program then gives other numbers in a bang on
the GPU than with `--gpu off` or on the JS lane.

`gpu_math_bits.bend`: a bang over 2^14 leaves computes f(x) for 2^20 inputs (-128..128 in steps of 1/4096, with large
and tiny values mixed in) into one block, and compares each word with the host's f(x):

| f | words that differ, GPU | `--gpu off` | GPU, with the argument cast to `double` in the emitted C |
|---|---:|---:|---:|
| `F32.exp` | 217,244 | 0 | 0 |
| `F32.log` | 19,870 | 0 | 0 |
| `F32.sqrt` | 0 | 0 | 0 |
| `F32.tanh` | 4,959 | 0 | 0 |
| `F32.sin` | 204,758 | 0 | 0 |
| `F32.cos` | 202,825 | 0 | 0 |
| `F32.pow(x, 1.5)` | 32,620 | 0 | 4 |

(of 1,048,576 each). Casting the argument to `double` on the device (`(f32)exp((double)f32_unbox(x))`) makes the GPU give
the host's bits for every function but `pow`, where CUDA's and glibc's double `pow` still disagree on 4 inputs. The cost
of the double functions on a consumer GPU (FP64 at 1/64 rate) is the trade-off; I have not timed it.

## Why it matters

Programs that compare a GPU run with a CPU run bit for bit (bend-ml-tensor-gpu does, for matrix products, which use only
+ and *) cannot do so for anything with an exponential: a softmax, SiLU, a cross-entropy. For a training run it also
means the GPU and the CPU drift apart from the first step.

## Possible directions

- Emit the double call on the device too (as the host does), so every lane gives the same bits.
- A float call on the host too (`expf`) would not be enough: glibc's `expf` and CUDA's are different implementations
  and would still disagree.
- Either way, document what each lane computes.
