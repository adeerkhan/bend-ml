# Draft issue for bendlang/bend: a safe way to give a fork tree's leaves disjoint parts of one Array to write

Status: posted as https://github.com/bendlang/bend/issues/1777 (2026-10-19, by Claude Code with Renan's go-ahead), in the Feature form's fields; no duplicate on a recheck the same day. Checked on 2026-10-18 (searched "Array.fork", "Array.join",
"borrow", "disjoint", "shared array", "split array"): no issue asks for it. Related: #1376 (a read-only borrow, opened by
Renan; this is its counterpart for writes), #885 (where `Array.fork`/`Array.join` were introduced for an `@unsafe` def),
PR #1278 (the histogram bench, whose description says "Bend has no safe way to give lanes disjoint slices of one array").
Taelin on read-only sharing (2026-10-16): the unsafe route "is fine for read-only for now", and they want to study
"whether borrow inference should be used instead of explicit references like Rust". Writes are not covered by that, so
this asks whether they are meant to be, and whether it belongs in Base or in a library (Renan's rule: we built the
packages ourselves, now we ask).

---

**Title:** A safe way to give a fork tree's leaves disjoint parts of one Array to write (the counterpart of #1376)

## The pattern

The one-block pattern of `tests/run/stencil3d.bend` (and of bend-parallel's histogram, PR #1278): one `Array`, a fork tree
whose every node holds a handle to it (`Array.fork`), each leaf writing its own part, the halves joined back
(`Array.join`). It is how a GPU kernel is written in Bend today, and the fast one: in bend-ml I use it for matrix
products on the CPU threads (bend-ml-tensor-fork: each leaf writes its rows of C) and on the GPU (bend-ml-tensor-gpu:
each leaf writes its tile of C), and both give tensor-array's bits.

Both packages are `@unsafe`, and so is every program that uses them. That part is expected: Base models `Array.fork(a)`
as `(a, a)` and `Array.join(a, b)` as `a`, so a write through the second handle is not in the model at all, and no law
about the result can be stated. The alternatives cost what the pattern saves:

| way to write disjoint parts in parallel | what it costs |
|---|---|
| each leaf returns its part as a list, written into the array after the join | 2-3.6x slower than not splitting, for products with a short inner dimension (bend-ml-tensor-array's NOTES); on CUDA, the lists' appends hit the lane stack (#1393) |
| each leaf gets its own array, concatenated after the join | a copy of the whole output per call |
| `Array.fork` / `Array.join` | O(1), the speed of the pattern, but `@unsafe` |

## What could make it safe

Base already has the right shape in its model: an `Array<T>` is `ALeaf{value}` or `ANode{xs, ys}`, two disjoint
halves. In the compiled program, though, matching an array copies each half into its own block (BendRT, "Arrays"), so a
fork tree cannot use it. Something like

```python
# split a into its two halves, both views of a's block: O(1), no copy
Array.halves(-T: Type, a: Array<T>) -> Array<T> & Array<T>      # model: match a { ANode{xs, ys}: (xs, ys) }
# put two halves of one block back together: O(1) when they came from Array.halves
Array.node(-T: Type, xs: Array<T>, ys: Array<T>) -> Array<T>     # model: ANode{xs, ys}
```

would have a model in which each leaf owns its half (writes included), so laws about the joined result could be stated
and proved, and the runtime would do what the unsafe pattern does today. A leaf at depth d owns a contiguous range of
2^(capacity - d) cells. That fits tensor-fork (blocks of rows of C), stencil3d (slabs) and the histogram's rows. It does
not fit a 2-D tile directly (tensor-gpu's leaves write 4 x 4 tiles of a row-major C): such a kernel would store C
tile-major and permute it afterwards, or need a split by stride, which is a further question.

Questions:

1. Is a safe split for writes on the roadmap, alongside the read-only borrow of #1376 (or the borrow inference Taelin
   mentioned)?
2. Would it belong in Base (as above), or is a library the intended place, with `@unsafe` contained in it as bend-ml does
   now?
3. Is the halves-of-a-node shape the right one for the runtime (a view into one block, freed once), or would ranges
   (`Array.split_at(a, i)`) fit it better?

I can share the packages' tests (every product against the sequential code, bit for bit, on the CPU threads and on the
GPU) as a test bed for whatever shape it takes.
