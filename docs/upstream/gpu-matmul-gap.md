# Draft issue for bendlang/bend: where a tiled matmul's time goes on CUDA, against the same kernel handwritten

Status: draft (2026-10-18), not posted; Renan decides whether and when to post it, after the tables are rerun on an
idle machine (the ratios below come from a busy CPU and an idle GPU, driver 615.78.08, except where marked idle). It is a
measurement report with questions, not a bug. Checked for duplicates on 2026-10-18 (searched "matmul CUDA", "handwritten
CUDA", "occupancy", "registers", "page fault"): none. Pairs with the poll issue (`gpu-error-poll.md`).

---

**Title:** A tiled matmul in Bend runs 4.3x slower than the same kernel handwritten in CUDA: where the time goes

I wrote a tiled matmul with the one-block pattern of `tests/run/stencil3d.bend` (`Array.fork`/`Array.join`, one block per
array, one 4x4 block of C a leaf over 2^14 leaves, A stored transposed so lanes read neighbouring cells), and the same
kernel in CUDA (same thread mapping, layouts and data, compiled with NVRTC, `--fmad=false` as Bend's device build). On an
RTX 4050 Laptop, idle machine, Bend 2.0.37: 512x512x4096 takes 4.78 ms a product in Bend and 1.10 ms handwritten; a K
sweep gives ~1.0 µs a K step in Bend against ~0.27 µs, plus ~0.7 ms a bang. The best tile in each (8x4) gives 3.48
against 0.81 ms. Results are the same bits in both. This is where the time goes, with Nsight Systems and Nsight Compute.

## A bang's fixed cost (~0.8-1.4 ms)

Each bang launches `bend_dev` four times (`gpu_run`). Nsight Systems, medians:

| ms | grow, 1 group | grow, 128 groups | main pass | final | host, between bangs |
|---|---:|---:|---:|---:|---:|
| an empty bang over a 2^14-leaf tree | 0.58 | 0.16 | 0.17 | 0.006 | 0.49 |
| the 512² matmul | 0.33 | 0.15 | 4.50 | 0.009 | 0.27 |

- The first launch grows the fork tree on one group of 128 lanes; for a balanced 2^14 tree that is up to 0.6 ms before
  any leaf runs.
- Between bangs the host and the device trade managed-memory pages: ~335 CPU and ~42 GPU page faults a bang for the
  empty bang, ~228 and ~10 for the matmul, while the host only reads a result.

## The main pass (4.5 ms against 1.04 ms)

Nsight Compute, application replay (kernel replay restores device memory, and with Bend's managed heap that makes every
replay fault its pages back in):

| | Bend | handwritten |
|---|---:|---:|
| warp instructions | 175.6 M | 95.5 M |
| SASS instructions a K step | 83 | ~45 |
| registers; blocks of 128 per SM | 80; 5 (limited by the 18,432 dynamic shared bytes, see #1602), 6 by registers | 39-48; 10-12 |
| warps active / eligible a cycle | 15-16 / 0.35 | 23.6 / 1.44 |
| issue active | 27% | 50% |
| L1 hit rate | 86.5% | 87.4% |

Both read the same bytes from the same caches; Bend waits longer for them. In the K loop's SASS, 69% of the stall samples
sit on the first multiply after the 8 loads, and the step carries, besides the 8 loads and 32 float operations:

- the error poll (`YIELD`, `BSSY`/`BSYNC`, a counter and a branch every step; the separate issue);
- ~30 address instructions: every `Array.get` masks its index (`i & (2^cls - 1)`) and the mask is rebuilt from the
  block's class bits each step (`blk_at` reads `blk_cls(a)` of an array that is a loop variable), so the compiler cannot
  turn `base + i` into a pointer plus a fixed offset;
- a 64-bit counter for the `Nat` fuel.

And the whole runtime is one kernel, so the matmul's loop runs with the register count of the largest path in it (80),
which caps the resident lanes.

What I could reproduce: adding the index mask, the poll and Bend's occupancy to the handwritten kernel makes it 1.25-1.35x
slower (1.45-1.7 ms). The rest is coalescing. Bend's loads cost 16.45 sectors a request against 8.5 handwritten, and
40.5 M sectors come from L2 against 5.7 M: the warps do not hold the leaves the layout assumes. Each block of 128
threads runs one aligned range of 128 leaves, but inside the block the leaves are dealt to threads in an order that
changes from run to run (`gpu-leaf-order.md`, a separate issue). Renumbering the leaves to fit the order I observed most
often (a permutation of the low 7 bits) brings the matmul to 9.39 sectors a request, 12.7 M from L2, and 3.16 ms a
product (from 4.3, the same results). Ruled out: the leaf count (4,096 to 16,384 leaves give the same ratio), unrolling
(the handwritten kernel without unrolling is not slower), managed memory (the handwritten kernel in one managed heap
runs as fast) and the data layout (the same in both).

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
