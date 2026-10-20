# Draft issue for bendlang/bend: on CUDA, a balanced tree's leaves are dealt to threads in a run-time order

Status: draft (2026-10-18), not posted; Renan posts it. Rerun on main 881cb2e5 (with #1602) on 2026-10-19: the map
three more times, the layouts and the matmul on an idle machine (bend2-lab `experiments/2026-10-19-leaves-mm-idle/`). Checked for duplicates on 2026-10-18 (searched "lane order",
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

In the C that main (881cb2e5) emits for it I patched that write, on the device only, to store `blockIdx.x * 128 +
threadIdx.x` instead of `t`, built it as `bend -o` does, and ran it three times (Bend 2.0.37 gave the same picture):

- every leaf runs on its own thread, all on the GPU;
- every block of 128 threads gets exactly one aligned range of 128 leaves, in every run;
- inside the block the order changes: 6,048 and 6,660 of the 16,384 leaves were on the same thread as in the first run;
- a warp's 32 leaves spread over the block's whole range, so a read of element j of leaf t at `j * 16384 + t` touches 16
  sectors of 32 bytes per warp in every run (Nsight Compute on main: 15.9 sectors a request; 4 would be coalesced).

The layout matters even with this order: 200,000 reads a leaf over one shared 2^26-number block take 112-113 ms
interleaved (element j of leaf t at `j * 16384 + t`), 230-235 ms with each leaf's cells contiguous and 175-196 ms
interleaved by the bit-reversed leaf index (main 881cb2e5, idle machine, driver 615.78.08, 5 alternated rounds); an empty
bang over the same tree takes 1.35 ms.

What it costs: a tiled matmul (512x512x4096, a 4x4 block of C a leaf) whose layout assumes neighbouring leaves on
neighbouring threads reads 16.45 sectors a request (Nsight Compute) against 8.5 for the same kernel handwritten in CUDA.
Renumbering the leaves with a permutation of the low 7 bits that fits the observed order (in all six maps, Bend 2.0.37's
and main's, a warp then touches ~5 sectors) gets it to 9.59 and makes the product 1.58x faster (4.62-4.72 -> 2.82-2.96
ms, the same results), but that is a guess about a run-time order: for the 8x4 version, a tree of depth 13, the same
permutation is slower (3.44-3.54 -> 3.58-3.70 ms).

Would it be possible to deal a balanced tree's leaves in a fixed order (leaf k on thread k, or any documented order), or
to give a bang's code its thread index (like `threadIdx`/`blockIdx`, or CubeCL's `ABSOLUTE_POS`) so that a program can
choose its layout? Either would let flat loops over shared blocks read coalesced.
