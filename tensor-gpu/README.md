# bend-ml-tensor-gpu

The six products of [`bend-ml-tensor-array`](../tensor-array) (`matmul`, `matmul_nt`, `matmul_tn` and their accumulating
forms), with the same arguments, result types and **bits**, each run as one `!` call (a bang) so that it runs on the GPU
when there is one. It is **not proved**, and says so: like [`bend-ml-tensor-fork`](../tensor-fork), it shares arrays
between the leaves of a fork tree through Base's `@unsafe` `Array.fork`.

- Bend: **2.0.37** · License: MIT · Depends on `bend-ml-tensor-array@0.1.12.0`. Not published to BendHub yet.
- `bend tensor-gpu/main.bend` → `SOME PROOFS FAIL`, naming the 18 defs that rely on `@unsafe` code. All of it is Bend:
  no foreign code, no CUDA written by hand.
- Tests: `tests/bits.bend`, against tensor-array, bit for bit, on the GPU, on 1 and 16 CPU threads and on the JS lane.
- Written by [`scripts/gen_tensor_gpu.py`](../scripts/gen_tensor_gpu.py) (4 x 4 tiles): a tile carries 64 sums through
  its loop, which is tedious to write by hand.

```python
import Base
import bend-ml-tensor-array@0.1.12.0/main.bend as TA
import bend-ml-tensor-gpu@0.1.0.0/main.bend as TG

def prod(x: TA.Mat<256n, 576n>, w: TA.Mat<1536n, 576n>) -> TA.MMulNT<256n, 576n, 1536n>:
  TG.matmul_nt(256n, 576n, 1536n, 0n, x, w)   # instead of TA.Mat.matmul_nt(...); par is ignored
```

Build with CUDA to get the GPU: `CUDA_HOME=... bend x.bend -o x` writes `x` and `x.gpu`. Without CUDA, or with
`./x --gpu off`, the same bangs run on the CPU threads, with the same bits.

## How it works

Each product is one bang over a balanced fork tree. Leaf t computes a 4 x 4 tile of C (tiles in row-major order; leaves
past the last tile write nothing). Every node shares A, B and C (no copy), as upstream's one-block pattern does
(bendlang/bend `tests/run/stencil3d.bend`). A plain product makes its C inside the bang: made on the host, a new C often
reused a block whose pages were on the device, and zeroing it there pulled them back over PCIe (9.2 → 6.0 ms on a
256 x 576 x 1536 product).

The bits: tensor-array's dot product keeps four partial sums. Product p goes to sum p mod 4, in increasing p, each sum
starting at +0.0, and the four are combined at the end in an order that depends on k mod 4 (`TA.t8_comb`). Each leaf does
the same for each of its 16 cells, four steps of k a turn; the steps past k in the last turn add +0.0, which leaves a sum
unchanged, since a sum that starts at +0.0 is never -0.0. An accumulating product adds a cell to C once, with
tensor-array's own `cwr`. Bend builds the device code without fused multiply-add. So every cell has tensor-array's bits,
on any tile size and any number of leaves; `tests/bits.bend` checks 2,958 products (n, m = 1..9 with k = 0, 1, 3, 8, 9,
17, and bigger shapes, in the three layouts and both modes), the six public products and three big ones, with -0, ±inf,
NaN and a subnormal mixed into the inputs.

## Speed (first measurement)

RTX 4050 Laptop, driver 615.78.08, CUDA 12.9, against tensor-fork on 16 CPU threads (Core Ultra 7 155H), ms a product,
medians of 3 alternated rounds, 2026-10-18 (an idle machine; the final numbers come with bend-ml's next benchmark run):

| product (BendTorch's linear layers at SmolLM2-135M's sizes) | tensor-fork, CPU | tensor-gpu, GPU |
|---|---:|---:|
| `x·Wᵀ`, 256 x 576 → 1536 (`matmul_nt`) | 9.6 | 6.05 |
| `dy·W`, 256 x 1536 → 576 (`matmul`) | 10.0 | 4.05 |
| `dW += dyᵀ·x`, 1536 x 256 → 576 (`matmul_tn_acc`) | 8.0 | 2.85 |
| 512 x 4096 → 512 (`matmul`) | 36.6 | 8.2 |

What it costs and what is left:

- Keeping tensor-array's four partial sums costs a lot of registers: the device program uses 241 a lane with 4 x 4
  tiles (2 blocks of 128 lanes per SM). A kernel that keeps one sum a cell ran the 512 x 4096 x 512 product in 3.5 ms
  instead of 8.2, with different bits.
- Each bang has a fixed cost of about 0.8-1.4 ms (the runtime grows the tree on one group of lanes first), and arrays
  the host touched between bangs move over PCIe. Small products are faster on the CPU.
- The leaves' order on the lanes is dealt at run time inside each group of 128, so the loads are not fully coalesced
  (16 sectors a request for `matmul_nt`). Upstream issues are drafted with the measurements.
