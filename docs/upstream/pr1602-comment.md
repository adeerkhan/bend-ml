# Draft comment for bendlang/bend PR #1602 (static `vote[3]`, 0 dynamic shared bytes on CUDA)

Status: discarded (2026-10-19, Renan): PR #1602 was merged on 2026-10-19, before it was posted. The timings are from an idle machine (alternated rounds, driver
610.57.04); the Nsight Compute lines from a busy CPU (driver 615.78.08), counts and occupancy only.

---

Numbers from another CUDA GPU, in case they help: RTX 4050 Laptop (sm_89, 20 SMs), driver 610.57.04, CUDA 12.9, Bend
2.0.37. I applied this PR's two lines to the C that `bend -o x.c` emits and built both versions from the same file
(`clang ... -O3`, then `--gpu-build`); same results in every case, idle machine, medians of 5 alternated rounds.

| program (16,384 leaves, one bang) | base | this PR | base / PR |
|---|---:|---:|---:|
| 2M reads a leaf of a 32-number Array | 2.02 s | 0.90 s | 2.24x |
| 2M steps a leaf of `acc * 0.999 + x`, no memory | 0.40 s | 0.28 s | 1.43x |
| 2M reads a leaf of a 4,096-number Array | 3.21 s | 2.48 s | 1.29x |
| 200,000 interleaved reads a leaf of one shared Array | 171 ms | 114 ms | 1.50x |
| the same, each leaf's cells contiguous | 309 ms | 231 ms | 1.34x |
| matrix-vector 16,384 x 4,096, steady state | 3 ms | 2 ms | 1.5x |
| tiled matmul 512x512x4096 (4x4 a leaf), 50 products | 442 ms | 470 ms | 0.94x |
| tiled matmul 256x1536x576 (4x4 a leaf), 100 products | 200 ms | 196 ms | 1.02x |

Two of them again on driver 615.78.08, another idle machine run (5 alternated rounds, the same binaries): the
interleaved reads 170 -> 115 ms (1.48x), the 512x512x4096 matmul 216 -> 231 ms (0.94x). Both builds of the matmul take
half the time they took on 610.57.04; the ratio is the same.

One more effect besides L1: occupancy. The device program of the matmul uses 80 registers, so
`cuOccupancyMaxActiveBlocksPerMultiprocessor` gives 6 blocks of 128 lanes per SM with 0 dynamic shared bytes and 5 with
18,432 (`TG_HOLD * 8`): 12,800 lanes at once instead of 15,360, while the launch asks for 128 groups (16,384 lanes). The
matmul is the one case that got slower here (6%); I have not found why.

Nsight Compute on that matmul's main pass (application replay, since kernel replay restores device memory and Bend's
heap is managed memory): with 18,432 dynamic bytes the shared-memory limit is 5 blocks per SM and the launch needs 1.28
waves, with 16.5 warps active a cycle; with this PR the limit moves to 14 (registers then cap at 6), 1.07 waves, 21.8
warps active, the same 175.6 M instructions, and the pass was not faster in my runs. So more resident lanes did not help
this kernel; the memory-bound ones above are where it pays.
