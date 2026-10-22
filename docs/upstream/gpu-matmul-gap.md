# Draft issue for bendlang/bend: where a tiled matmul's time goes on CUDA, against the same kernel handwritten

Status: posted as https://github.com/bendlang/bend/issues/1781 (2026-10-19, by Claude Code with Renan's go-ahead), in the Feature form's fields. Rewritten on 2026-10-19 against main
881cb2e5, which has PR #1602 (no dynamic shared memory at launch): every time below is from that day's idle round
(bend2-lab `experiments/2026-10-19-leaves-mm-idle/`), the counters from Nsight Compute on main, and the handwritten-twin
breakdown from 2026-10-18 (busy CPU, idle GPU), as marked. The text below the line is the current one; the older numbers
(2.0.37) are in git history. The headline times were rerun on an
idle machine on 2026-10-18 (driver 615.78.08, medians of 5 alternated rounds); the breakdowns below come from a busy
CPU and an idle GPU, except where marked idle. It is a
measurement report with questions, not a bug. Checked for duplicates on 2026-10-18 (searched "matmul CUDA", "handwritten
CUDA", "occupancy", "registers", "page fault"): none. Pairs with the poll issue (`gpu-error-poll.md`).

---

**Title:** A tiled matmul in Bend runs about 4x slower than the same kernel handwritten in CUDA: where the time goes

I wrote a tiled matmul with the one-block pattern of `tests/run/stencil3d.bend` (`Array.fork`/`Array.join`, one block per
array, one 4x4 block of C a leaf over 2^14 leaves, A stored transposed so lanes read neighbouring cells), and the same
kernel in CUDA (same thread mapping, layouts and data, compiled with NVRTC, `--fmad=false` as Bend's device build). RTX
4050 Laptop, driver 615.78.08, main 881cb2e5 (with #1602), idle machine, 5 alternated rounds; the results are the same
bits in both:

| ms a product | Bend | handwritten | ratio |
|---|---:|---:|---:|
| 512x512x4096, 4x4 a leaf | 4.62-4.72 | 1.08-1.23 | 4.3x |
| 512x512x4096, 8x4 a leaf | 3.44-3.54 | 0.78-0.82 | 4.4x |
| 256x1536x576, 4x4 a leaf | 2.10-2.17 | 0.224-0.226 | 9.4x |

From K = 1024 to 4096 (512², 4x4): ~0.97 µs a K step in Bend against ~0.27 µs, plus ~0.7 ms a bang, which is why the
smaller product does worse. This is where the time goes, with Nsight Systems and Nsight Compute.

## A bang's fixed cost (~1.3 ms empty)

Each bang is five `bend_dev` launches back to back, then the host's turn. Nsight Systems, medians of 19 bangs:

| ms | grow, 1 group | grow, 128 groups | main pass | final | host, between bangs |
|---|---:|---:|---:|---:|---:|
| an empty bang over a 2^14-leaf tree | 0.55 | 0.10 | 0.11 | 0.007 | 0.58 |
| the 512² matmul (4x4) | 0.32 | 0.15 | 5.06 | 0.009 | 0.30 |

(the fifth launch, between the grows and the main pass, takes ~1 µs). The first launch grows the fork tree on one group of
128 lanes, so a balanced 2^14 tree costs up to 0.55 ms before any leaf runs, and the host's turn adds about as much. On
2.0.37 I counted hundreds of CPU page faults a bang in that turn (Bend's heap is managed memory), while the host only
reads a result.

## The main pass

Nsight Compute on main, application replay (kernel replay restores device memory, and with Bend's managed heap that makes
every replay fault its pages back in):

| | Bend | handwritten |
|---|---:|---:|
| warp instructions | 175.7 M | 95.5 M |
| SASS instructions a K step | 83 | ~45 |
| registers; blocks of 128 per SM | 80; 6 (registers) | 39-48; 10-12 |
| warps active a cycle | 21.6 | 23.6 |
| sectors a load request | 16.45 | 8.5 |

The step carries, besides the 8 loads and 32 float operations:

- the error poll (`YIELD`, `BSSY`/`BSYNC`, a counter and a branch every step; #1776);
- ~30 address instructions: every `Array.get` masks its index (`i & (2^cls - 1)`) and the mask is rebuilt from the
  block's class bits each step (`blk_at` reads `blk_cls(a)` of an array that is a loop variable), so the compiler cannot
  turn `base + i` into a pointer plus a fixed offset;
- a 64-bit counter for the `Nat` fuel.

And the whole runtime is one kernel, so the matmul's loop runs with the register count of the largest path in it (80),
which caps the resident lanes at 6 blocks per SM.

The largest single part is coalescing: Bend's loads cost 16.45 sectors a request against the twin's 8.5, because the
warps do not hold the leaves the layout assumes. Each block of 128 threads runs one aligned range of 128 leaves, but
inside the block the leaves are dealt in an order that changes from run to run (a separate issue). Renumbering the
leaves to fit the order I observed (a permutation of the low 7 bits) brings the loads to 9.59 sectors a request, with
the same instructions:

| 512², 4x4, ms a product | |
|---|---:|
| as built | 4.62-4.72 |
| leaves renumbered | 2.82-2.96 |
| renumbered, poll split (#1776's patch) | 3.14-3.30 |
| renumbered, poll removed | 2.16-2.22 |
| handwritten | 1.08-1.23 |

The last Bend row minus the ~0.7 ms a bang costs (~1.5 ms) is where the handwritten kernel lands once it copies Bend's
index mask, poll and occupancy: 1.36-1.47 ms at 6 blocks per SM (2026-10-18, busy CPU; the mask alone +20%). Copying
Bend's addressing further (both masks rebuilt every step, a 64-bit counter) costs the twin nothing measurable. Ruled out:
the leaf count (4,096 to 16,384 leaves give the same ratio), unrolling (the twin without unrolling is not slower), managed
memory (the twin in one managed heap runs as fast) and the data layout (the same in both).

One thing I did not expect: with the leaves renumbered, #1776's split poll is slower than the plain poll (3.14-3.30
against 2.82-2.96 ms), while on the matmul as built it is faster (222-227 against 230-237 ms for 50 products). I have not
looked at why.

## Questions

- Could a balanced tree's leaves be dealt to threads in a fixed order, or could a bang expose the thread's index? (the
  separate issue)
- Could the tree be grown on more than one group, or could a bang over a known balanced tree skip the grow?
- Is there a way to keep a bang's blocks on the device between bangs (no host touches until the result is read)?
- Could a spin's array masks be computed once per spin when its arrays are passed unchanged (the common case in a flat
  loop over one block)?
- Would a separate, smaller kernel for leaf-only spins be possible, so a flat loop does not inherit the runtime's 80
  registers?

Programs and Nsight reports available on request; I can rerun anything on this GPU.
