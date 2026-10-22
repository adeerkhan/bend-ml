# Draft comment for bendlang/bend #1393 (and PR #1439): the same lane-stack limit on CUDA

Status: posted on PR #1439 (2026-10-19, https://github.com/bendlang/bend/pull/1439#issuecomment-6279729706, by Claude Code with Renan's go-ahead), after checking that main 881cb2e5 still faults.
#1393 and #1439 show Metal only; I built #1439 (6764c339) and its base (3f9c40e8) from source and ran them on CUDA
(below).

---

The same limit holds on CUDA (Bend 2.0.37, RTX 4050 Laptop, driver 610.57.04, CUDA 12.9): a non-tail recursion inside a
`!` faults with `bend: memory fault (machine stack overflow?)` once it needs more than the lane's 2,048 stack words, and
the same binary with `--gpu off` finishes. The smallest program I found:

```python
import Base

def deep(n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      (deep(p) + 1 : U32)

def main() -> IO(Unit):
  IO.print(U32.show(deep!(5000n)))
```

GPU: the fault, exit 1. `--gpu off`: 5000. Without the `!`: 5000. The thresholds match `STAK_LEN = 1 << 11`, at 3 +
w x depth > 2048 words, w the words a level keeps:

| recursion | words a level | last depth that runs | first that faults |
|---|---:|---:|---:|
| `deep(p) + 1` | 1 | 2045 | 2046 |
| `List.append` | 2 | 1022 | 1023 |
| `deep3(p, a, b) + a + b` | 3 | 681 | 682 |

Every run was deterministic, and `--gpu 4GB` does not change it (the stack is a fixed block outside the heap). I hit it
with a matrix product that appended row lists at each join of its fork tree (fine on the CPU, a fault on CUDA); replacing
`List.append` with two tail loops fixed it. Two small things that would have saved me time: the message says "machine
stack", but it is the device lane's stack, and neither `bend guide` nor `SHADERS.md` gives the 2,048-word figure.

I built this PR (6764c339) and its base (3f9c40e8) from source and ran the programs above on CUDA. The base faults at
every depth past the limit; the PR runs all of them with the CPU's results, my matrix product included (9031405), and a
depth of 100,000 (`deep!(100000n)`). It did not change the speed of two loop kernels I time (a flat loop over 16,384
leaves: 213-219 ms against 213-259; a tiled matmul: 172-175 ms both), measured with the CPU busy.
