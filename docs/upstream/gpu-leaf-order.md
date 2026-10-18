# Draft issue for bendlang/bend: on CUDA, a balanced tree's leaves are dealt to threads in a run-time order

Status: draft (2026-10-18), not posted; Renan posts it. Checked for duplicates on 2026-10-18 (searched "lane order",
"thread index", "leaf order", "coalesce", "lane index"): none. Bend 2.0.37, RTX 4050 Laptop, drivers 610.57.04 and
615.78.08 (the map below on 615.78.08). Pairs with `gpu-matmul-gap.md` and `gpu-error-poll.md`.

---

**Title:** CUDA: the leaves of a balanced fork tree land on threads in an order that changes from run to run, so a
layout cannot be coalesced

`SHADERS.md` asks for one leaf per lane and flat loops; the speed of such a loop on a GPU then depends on neighbouring
threads reading neighbouring addresses. To lay data out that way I need to know which thread runs which leaf. I
measured it:

```python
import Base

@unsafe
def launch(+d: Nat, +t: U32, +y: Array<U32>) -> Array<U32>:
  match d:
    case 0n: Array.set(U32, y, t, t)
    case 1n+p:
      l r = launch(p, (t * 2 : U32), y) launch(p, ((t * 2 : U32) + 1 : U32), y)
      Array.join(U32, l, r)
# main: launch!(14n, 0, Array.new(U32, 14n, 0)), then print every cell
```

In the C from `bend x.bend -o x.c` I patched that write, on the device only, to store `blockIdx.x * 128 +
threadIdx.x` instead of `t`, built it as `bend -o` does, and ran it three times:

- every leaf runs on its own thread, all on the GPU;
- every block of 128 threads gets exactly one aligned range of 128 leaves, in every run;
- inside the block the order changes: 6,900 and 5,312 of the 16,384 leaves were on the same thread as in the first run;
- a warp's 32 leaves spread over the block's whole range, so a read of element j of leaf t at `j * 16384 + t` touches 16
  sectors of 32 bytes per warp in every run (Nsight Compute: 15.9 sectors a request; 4 would be coalesced).

What it costs: a tiled matmul whose layout assumes neighbouring leaves on neighbouring threads reads 16.45 sectors a
request against 8.5 for the same kernel handwritten in CUDA. Renumbering the leaves with a permutation of the low 7 bits
that fits the most frequent order gets it to 9.39 and makes the product 1.35x faster (4.3 -> 3.16 ms), but that is a
guess about a run-time order: for a tree of depth 13 the same permutation is slower.

Would it be possible to deal a balanced tree's leaves in a fixed order (leaf k on thread k, or any documented order), or
to give a bang's code its thread index (like `threadIdx`/`blockIdx`, or CubeCL's `ABSOLUTE_POS`) so that a program can
choose its layout? Either would let flat loops over shared blocks read coalesced.
