# Draft issue and PR for bendlang/bend: #1447's CUDA entry check costs ~3%; the round after a sync picked on the host

Status: draft (2026-10-25), for Renan to open. Asked on the Bend Discord first; lorenzo (2026-10-25): "either is fine,
you can implement it and we drift the direction if needed". The branch is `cuda-entry-check` in `~/Repos/Bend2/bend`
(one commit, 6dcc1878, on main 295e9965), pushed to nuxyel/bend. Duplicates checked on 2026-10-25 (searched "1447", "command
buffers", "rounds CUDA", "CUDA slower 2.0.38"): none. Evidence: bend2-lab `experiments/2026-10-25-cuda-rounds/`
(private; the numbers below are all the PR needs). Shaped by the maintainers' reviews: Metal stays as it is, the PR
goes in over the cap rather than trimming tokens (#1820), and it edits no CHANGELOG, GUIDE or gates.

The tables below were rerun on an idle machine on 2026-10-25.

Open the issue first, then the PR with its number in the `Fixes` line (replace `#ISSUE`), then
comment on the issue pointing at the PR. The two texts are below.

---

## Issue

**Title:** CUDA: #1447's entry check in `bend_dev` makes a small training loop ~3% slower (bisected)

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

I have a fix that keeps #1447's chaining (the first round after each sync picked on the host, without the check) and brings the time back; I'll open it as a PR and link it here.

## Comment on the issue, after the PR is open

The fix is in #PR: the first round after each sync launches only the dispatches the frontier the host read needs, without the check, and the rounds chained after it keep it. On this loop main takes 16.72 s and the PR 16.27 s; the 17 runtime benches on the GPU are unchanged (geometric mean 1.00x).

---

## PR

**Title:** CUDA: the round after a sync is picked on the host, so its dispatches skip the device's frontier check

Fixes #ISSUE

Since 2.0.38 a GPU training loop of mine is about 3% slower on CUDA, and the cost is #1447's entry check in `bend_dev` (#ISSUE has the bisection and the variants that isolate it). This keeps #1447's device-chained rounds and takes the check off the round the host can pick itself.

## Change

`bend2/comp.ts` only.

- After each sync the host can read the frontier, so the first round after it launches only the dispatches that frontier needs (`turn_groups(f, at)`, the same rule the device applies), and those launches skip the check.
- The rounds after it, between that sync and the next, are chained on the device as in #1447, and keep the check.
- The kernel takes a fourth argument, `chk`, and checks only when it is set. `gpu_run(f)` takes the frontier the host read, or 0 for "launch all five, checked".
- Metal: two lines change (`gpu_kernel` takes the unused `chk`, and its one call is `gpu_run(0)`), so it launches the same dispatches as before. The host CPU path doesn't change.

## Measurements

The training loop from #ISSUE, current code, main (295e9965) against this branch, 5 alternated rounds on an idle machine, the same output bits:

| | main | this PR |
|---|---|---|
| train 200, median | 16.72 s | 16.27 s |
| range | 16.60-16.87 s | 16.15-16.31 s |

The 17 runtime benches in PAR-GPU mode (`+RTS --gpu <the gate's MEMORY size>`), built by each compiler with `CUDA_HOME` set, 5 alternated rounds on an idle machine, medians in ms, every output identical between the two:

| bench | main | this PR | ratio |
|---|---:|---:|---:|
| bfs | 563 | 564 | 1.00x |
| editdist | 768 | 773 | 1.01x |
| gameoflife | 215 | 210 | 0.98x |
| hashmap | 775 | 780 | 1.01x |
| histogram | 1022 | 1020 | 1.00x |
| kmeans | 493 | 499 | 1.01x |
| lexer | 1334 | 1327 | 0.99x |
| mandelbrot | 200 | 201 | 1.00x |
| merkle | 209 | 209 | 1.00x |
| nbody | 196 | 195 | 0.99x |
| queens | 1584 | 1622 | 1.02x |
| raytrace | 954 | 965 | 1.01x |
| symreg | 503 | 504 | 1.00x |
| terrain | 572 | 568 | 0.99x |
| tree-bitonic | 1851 | 1847 | 1.00x |
| tree-matmul | 536 | 535 | 1.00x |
| tree-radix | 755 | 757 | 1.00x |

Geometric mean 1.00x: the benches neither gain nor lose.

## Tests

The 42 tests that call a bang, built for CUDA by main and by this branch and run on the GPU: the 40 that are programs print their `#|` lines on both; the two in `tests/printer/` aren't runnable programs and fail to build the same way on both.

## Gates

`bun gates/repo.ts` fails only on the cap: `comp.ts` is 66,096 ttok, 96 over 66,000 (main is under it). I didn't trim anything to fit.

## Not tested

- Metal: I have no Mac, so I couldn't build it. The Metal diff is the two lines above.
- `gates/test.ts`, `gates/perf.ts` and `gates/safe.ts` need the cluster.
