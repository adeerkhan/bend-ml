# Draft issue for bendlang/bend: the CUDA lane is not covered by the test and perf gates

Status: draft (2026-10-18), not posted; Renan decides whether to post it (it is a request, not a bug) and whether to
offer runs on his RTX 4050. Checked on 2026-10-18 (`gates/test.ts`, `gates/perf.ts`, `.github/workflows/`): the gates
that run `!` programs run on the cluster of Apple M4 minis (Metal); GitHub Actions runs only `repo-gate`. Searched
"CUDA CI", "CUDA test": nothing.

---

**Title:** The test and perf gates run `!` on Metal only; CUDA-specific behaviour goes unnoticed

The test gate builds every runnable test with Metal and runs its `!` on the minis, and the perf gate pins its GPU cells
on `apple_m4`. Nothing runs the CUDA lane, and it differs from Metal in ways the gates cannot see: managed memory over
PCIe instead of unified memory, the shared-memory reservation that PR #1602 changes, register counts and occupancy, and
NVRTC's code generation. Three things I found on an RTX 4050 Laptop with Bend 2.0.37 that no gate would report:

- the device loops' error poll costs 4-20x on tight loops on CUDA (a separate issue: [link once posted]);
- #1393's lane-stack fault reproduces on CUDA at the same depths (comment on #1393: [link]);
- PR #1602 changes CUDA kernels by 0.94-2.24x (comment on #1602: [link]).

Would a CUDA machine in the gates be possible, even a small one running the `!` tests? If it helps meanwhile, I can run
the test suite's `!` programs on CUDA before releases and report what differs from Metal.
