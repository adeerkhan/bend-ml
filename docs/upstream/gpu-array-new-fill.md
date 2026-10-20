# Draft issue for bendlang/bend: on CUDA, `Array.new` inside a bang fills its block on one lane

Status: posted as https://github.com/bendlang/bend/issues/1778 (2026-10-19, by Claude Code with Renan's go-ahead), in the Bug form's fields. Checked on 2026-10-18 (searched "Array.new GPU", "Array.new
fill", "blk_new"): none open; #1267 is about `Array.new` of boxed values at depth 24, not this. Bend 2.0.37, RTX 4050
Laptop, driver 615.78.08, NVRTC 12.9. Rechecked on main 881cb2e5 (2026-10-19): `blk_new` is the same loop, and the
sweep gives the same ~4.6 ns a cell. Repro: `gpu_array_new_fill.bend` (next to this file).

---

**Title:** CUDA: `Array.new` inside a bang fills the new block in one loop on one lane (~4.6 ns a cell)

## What I see

A kernel that writes its output into an array made inside the bang (the pattern `tests/run/stencil3d.bend` uses, and
the one that keeps the output's pages on the device) pays for the allocation before any leaf runs. A bang whose root
only allocates 2^c F32 cells and forks 16 leaves that read one cell (`gpu_array_new_fill.bend`), Nsight Systems, the
first `bend_dev` launch (grid 1), 5 bangs each, Bend 2.0.37 and main 881cb2e5 (2026-10-19):

| cells | first launch, 2.0.37 | first launch, main |
|---:|---:|---:|
| 2^16 | 0.54-0.85 ms | 0.52-1.02 ms |
| 2^18 | 1.60-1.82 ms | 1.35-1.40 ms |
| 2^20 | 5.33-5.52 ms | 4.93-5.13 ms |
| 2^22 | 19.05-20.29 ms | 19.23-19.88 ms |

About 4.6 ns a cell, linear. In the emitted C, `blk_new` writes every cell in one loop on the lane that runs it:

```c
for (u64 i = 0; i < (1ull << c); i += 1) {
  blk_write(H, arr, l, (u32)i, i % (1u << lgs) < n ? v[i % (1u << lgs)] : 0);
}
```

and the root of a bang runs on one group while the tree grows. `blk_fill` (used by `Array.clone`, matching `ANode`, and
`blk_node`) is the same kind of loop.

## Why it matters

With Bend 2.0.37 I moved a whole training step of a transformer block (8 x 64 rows, width 576, feed-forward 1536) onto the GPU: every
layer as bangs that make their outputs inside the bang, so nothing moves to the host between them. The leaves' work
takes ~83 ms a step; the first launches, where these fills run, take 115-130 ms. Making the outputs on the host instead
is worse here: a new block often reuses pages that sit on the device, and filling it on the host pulls them back (3
steps: 1,118 ms against 760).

## Questions

- Could `blk_new` on the device fill with every lane of the group (or of the grid) instead of one, or skip the fill when
  the value is zero and the heap block is known to be zeroed?
- Would an `Array.new` that leaves the cells to the leaves (each leaf writes its own part) fit the runtime's model?
