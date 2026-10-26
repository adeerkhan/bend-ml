# Draft PR for bendlang/bend: CUDA picks the round after a sync on the host, skipping #1447's entry check

Status: draft (2026-10-25), for Renan to open. Asked on the Bend Discord first; lorenzo (2026-10-25): "either is fine,
you can implement it and we drift the direction if needed". The branch is `cuda-entry-check` in `~/Repos/Bend2/bend`
(one commit, 6dcc1878, on main 295e9965), not pushed. Duplicates checked on 2026-10-25 (searched "1447", "command
buffers", "rounds CUDA", "CUDA slower 2.0.38"): none. Evidence: bend2-lab `experiments/2026-10-25-cuda-rounds/`
(private; the numbers below are all the PR needs). Shaped by the maintainers' reviews: Metal stays as it is, the PR
goes in over the cap rather than trimming tokens (#1820), and it edits no CHANGELOG, GUIDE or gates.

Before opening: rerun the bench table on an idle machine (the one below ran at load ~2).

The text below the line is the PR.

---

**Title:** CUDA: the round after a sync is picked on the host, so its dispatches skip the device's frontier check

## Why

Since 2.0.38 a GPU training loop of mine is about 3% slower on CUDA. I bisected it to a97a7c46 (#1447, "GPU bangs chain their rounds on the device"). #1447 notes that its CUDA path wasn't compiled or tested, so here is what it does on a CUDA machine.

The program is a two-layer Llama trained for 200 AdamW steps, with its linear layers on the GPU (bend-ml-tensor-gpu): many small bangs per step, each one round long. RTX 4050 Laptop, driver 615.78.08, CUDA 12.9, Linux.

- 2.0.37: 16.29 s
- 2.0.38 and 2.0.39: 16.71 s
- bisection (each commit's `bend2/main.ts` run with bun): good commits 16.28-16.32 s, bad ones 16.82-16.93 s, first bad a97a7c46.

To find the part that costs, I edited 2.0.38's emitted C and built it with the clang line `bend -o` uses (3 rounds each):

| build | train 200 |
|---|---|
| 2.0.37 | 16.24-16.46 s |
| 2.0.38 as emitted | 16.74-16.94 s |
| `cuCtxSynchronize` after every round, instead of after 1, 2, 4 .. 64 | 16.77-16.92 s |
| ... and the host launching only the dispatches `turn_groups(H[H_FRONT], at)` allows, after that sync | 16.76-16.82 s |
| ... and without the entry check `if (!turn_groups(H[H_FRONT], at)) return;` | 16.22-16.26 s |

So the rounds queued past the end of the work and the empty dispatches don't cost anything measurable. The entry check does, even in dispatches that have work. The kernel uses 255 registers with and without it, so it isn't occupancy. I don't know the mechanism. Under nsys the kernels take the same total time with and without the check, and the host's CPU time is the same too, so the gap shows only in an untraced run. Unified-memory page faults are the same in both.

## Change

`bend2/comp.ts` only.

- After each sync the host can read the frontier, so the first round after it launches only the dispatches that frontier needs (`turn_groups(f, at)`, the same rule the device applies), and those launches skip the check.
- The rounds after it, between that sync and the next, are chained on the device as in #1447, and keep the check.
- The kernel takes a fourth argument, `chk`, and checks only when it is set. `gpu_run(f)` takes the frontier the host read, or 0 for "launch all five, checked".
- Metal: two lines change (`gpu_kernel` takes the unused `chk`, and its one call is `gpu_run(0)`), so it launches the same dispatches as before. The host CPU path doesn't change.

## Measurements

The training loop above, current code, main (295e9965) against this branch, 5 alternated rounds, the same output bits:

| | main | this PR |
|---|---|---|
| train 200, median | 16.74 s | 16.33 s |
| range | 16.73-17.05 s | 16.21-16.65 s |

The 17 runtime benches in PAR-GPU mode (`+RTS --gpu <the gate's MEMORY size>`), built by each compiler with `CUDA_HOME` set, 3 alternated rounds, medians in ms, every output identical between the two:

<!-- BENCH TABLE: rerun idle before posting -->

| bench | main | this PR | ratio |
|---|---:|---:|---:|
| bfs | 595 | 570 | 0.96x |
| editdist | 765 | 769 | 1.01x |
| gameoflife | 219 | 232 | 1.06x |
| hashmap | 787 | 784 | 1.00x |
| histogram | 1026 | 1015 | 0.99x |
| kmeans | 490 | 492 | 1.00x |
| lexer | 1320 | 1342 | 1.02x |
| mandelbrot | 216 | 200 | 0.93x |
| merkle | 219 | 204 | 0.93x |
| nbody | 189 | 198 | 1.05x |
| queens | 1615 | 1591 | 0.99x |
| raytrace | 978 | 964 | 0.99x |
| symreg | 515 | 517 | 1.00x |
| terrain | 588 | 574 | 0.98x |
| tree-bitonic | 1874 | 1872 | 1.00x |
| tree-matmul | 548 | 528 | 0.96x |
| tree-radix | 761 | 755 | 0.99x |

Geometric mean 0.99x. The benches under 250 ms move by up to 7% either way between sessions, for both builds.

## Tests

The 42 tests that call a bang, built for CUDA by main and by this branch and run on the GPU: the 40 that are programs print their `#|` lines on both; the two in `tests/printer/` aren't runnable programs and fail to build the same way on both.

## Gates

`bun gates/repo.ts` fails only on the cap: `comp.ts` is 66,096 ttok, 96 over 66,000 (main is under it). I didn't trim anything to fit.

## Not tested

- Metal: I have no Mac, so I couldn't build it. The Metal diff is the two lines above.
- `gates/test.ts`, `gates/perf.ts` and `gates/safe.ts` need the cluster.
