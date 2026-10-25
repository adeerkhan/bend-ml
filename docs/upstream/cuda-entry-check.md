# Draft issue for bendlang/bend: the CUDA kernel's entry check from #1447 costs ~3% on a small training loop

Status: draft (2026-10-25), for Renan to post. Checked for duplicates on 2026-10-25 (searched "1447", "command buffers",
"rounds CUDA", "CUDA slower 2.0.38"): none. Evidence: bend2-lab `experiments/2026-10-25-cuda-rounds/` (private; the
numbers below are all that the issue needs). The text below the line is the issue.

---

**Title:** CUDA: #1447's entry check in `bend_dev` makes a small training loop ~3% slower (bisected)

Since 2.0.38 a training loop of mine on the GPU is about 3% slower, and I bisected it to a97a7c46 ("GPU bangs chain their rounds on the device, three command buffers in flight", #1447). The PR notes that the CUDA path wasn't tested, so here is a measurement from a CUDA machine.

**Setup.** RTX 4050 Laptop (driver 615.78.08, CUDA 12.9), Linux. The program is a two-layer Llama trained for 200 AdamW steps, with its linear layers on the GPU through bend-ml-tensor-gpu: many small bangs per step. Every build gives the same output bits.

- 2.0.37: 16.29 s (idle machine, 5 rounds)
- 2.0.38 and 2.0.39: 16.71 s

**Bisection** (each commit's `bend2/main.ts` run with bun, `train 200` twice, the best kept): the good commits take 16.28-16.32 s and the bad ones 16.82-16.93 s; the first bad one is a97a7c46.

**What in it costs.** I edited 2.0.38's emitted C and built it with the same clang line `bend -o` uses (3 rounds each):

| build | train 200 |
|---|---|
| 2.0.37 | 16.24-16.46 s |
| 2.0.38 as emitted | 16.74-16.94 s |
| 2.0.38, `cuCtxSynchronize` after every round instead of after 1, 2, 4 .. 64 | 16.77-16.92 s |
| ... and the host launching only the dispatches `turn_groups(H[H_FRONT], at)` allows, read after that sync (as 2.0.37 chose them) | 16.76-16.82 s |
| ... and without the entry check `if (!turn_groups(H[H_FRONT], at)) return;` in `bend_dev` | 16.22-16.26 s |

So the rounds queued past the end of the work and the dispatches that find no work are not what costs; the entry check is. Once the host picks the dispatches, the check is redundant, and removing it brings the time back to 2.0.37's (slightly below). The kernel uses 255 registers in all three builds, so it isn't occupancy.

I don't know why one read of `H[H_FRONT]` at entry would cost ~2.6 ms on a ~80 ms step. My guess is that the passes are short and launched many times per step, so whatever the check does to the start of each launch adds up. The diffs of the three variants are small; I can attach them or open a PR for whichever direction you prefer (keeping the device-chained rounds and making the check cheaper, or picking the dispatches on the host on CUDA as before).
