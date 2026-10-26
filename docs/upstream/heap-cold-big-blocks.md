# Draft issue for bendlang/bend: every lane parks a big freed block, so peak memory grows with the thread count

Status: draft (2026-10-26), for Renan to post. Checked for duplicates on 2026-10-26 (searched "heap_hand", "COLD
memory", "free list threads memory", "peak memory threads", "RSS threads"): none; #1007 is about the same allocator's
address order, not its size. Evidence: bend2-lab `experiments/2026-10-26-1b7-memory/` (private; the numbers below are
all the issue needs).

---

**Title:** C runtime: each lane parks a freed big block as its COLD generation, so a training step's peak memory grows with the thread count

### What I ran

BendTorch's SmolLM2 training step (AdamW, F32 moments) with SmolLM2-1.7B's shapes cut to 2 and 4 decoder blocks, `train 2`, peak RSS from `/usr/bin/time %M`, Bend 2.0.39, Linux x86-64, Core Ultra 7 155H, 32 GB.

| model | `+RTS --threads 1` | `--threads 4` | `--threads 16` |
|---|---|---|---|
| 2 blocks | 4,988 MB | 5,684 MB | 5,510 MB |
| 4 blocks | 7,145 MB | 7,897 MB | 7,867-8,461 MB |

At one thread each block adds 1,078 MB, which is exactly its state (parameters, gradient and two moments, 4 x 268 MB). At 16 threads a block adds about 0.4 GB more, and the full 24-block model's step peaks at 24.1 GB where the state and the rest account for about 21.6 GB.

### Why, as far as I can tell

`heap_free` pushes onto the lane's HOT chain and, once HOT reaches `KEEP_WORDS` (32,768 words on the host), `heap_hand` banks COLD and parks HOT as COLD. For a class of 32,768 words or more, `KEEP(cls)` is 1, so a single block is a whole generation: every lane that frees a big block keeps it as its COLD, where no other lane can take it until that lane frees another block of the same class. A program whose big temporaries (here, packed copies of 64 MB weight matrices) are freed on whichever lane ran them ends up with up to one parked block per big class per lane.

### A change that removes it

Sending a big block to the bank at once, in `heap_free` before the push:

```c
  if (!DEVICE && KEEP(cls) == 1) {
    e.mem[loc] = 0;
    bank_push(e.mem, cls, loc);
    return;
  }
```

On the 4-block model at 16 threads this gives 7,181 and 7,182 MB (unpatched 8,434 and 7,867), the one-thread level, with the same output bits. The step took 13.86 and 13.84 s against 13.43 and 13.72 s unpatched, on a busy machine, so I can't tell whether the bank's lock costs anything; your perf gate would. Small classes are untouched, so the two-generation reuse that commit 94afd425 measured stays where it matters.

I can open a PR with this and a test, or try another shape if you'd rather keep big blocks lane-local (for example a cap on the parked words per lane).
