"""Writes tensor-gpu/main.bend, bend-ml-tensor-gpu: tensor-array's six products on the GPU with tensor-array's bits.

  python scripts/gen_tensor_gpu.py [TM TN] [out]      # tile of C a leaf computes (default 4 4)

Each leaf of one bang computes a TM x TN tile of C with four partial sums a cell, in tensor-array's order: product p of
a dot goes to sum p mod 4, in increasing p, each sum starting at +0.0, and the four are combined by TA.t8_comb with
rot = k mod 4 (tensor-array's dotr and t8_dot). The loop takes four steps of k a turn; the last turn's steps past k add
+0.0, which leaves a sum unchanged (a sum that starts at +0.0 is never -0.0). An accumulating product adds the cell to C
once with TA.cwr. So every cell has the bits tensor-array computes, whatever the tile or the number of leaves.
The code is generated because a tile of TM x TN cells carries 4 * TM * TN sums through the loop.
"""
import sys

TM, TN = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (4, 4)
OUT = sys.argv[3] if len(sys.argv) > 3 else "tensor-gpu/main.bend"
G = 4  # steps of k a turn
cells = [(r, c) for r in range(TM) for c in range(TN)]
S = [f"s{q}_{r}_{c}" for (r, c) in cells for q in range(4)]  # sums, cell-major
X = [f"x{g}_{r}" for g in range(G) for r in range(TM)]  # A values: step g, row r
Y = [f"y{g}_{c}" for g in range(G) for c in range(TN)]  # B values: step g, column c
V = X + Y


def sum_name(q, r, c):
    return f"s{q}_{r}_{c}"


def j(xs, sep=", "):
    return sep.join(xs)


o = []
p = o.append
p(f"""# bend-ml-tensor-gpu: tensor-array's products on the GPU, with tensor-array's bits.
#
#   import bend-ml-tensor-array@0.1.12.0/main.bend as TA
#   import bend-ml-tensor-gpu@0.1.0.0/main.bend as TG
#   TG.matmul(n, k, m, par, a, b)      # the same arguments, result type and bits as TA.Mat.matmul
#
# The six products (matmul, matmul_nt, matmul_tn and their accumulating forms) take tensor-array's Mat values and return
# its result types; par is ignored (the GPU decides the split). Each product is one `!` call (a bang): a balanced fork
# tree whose leaves each compute a {TM} x {TN} tile of C, every node sharing A, B and C through Base's Array.fork (two
# handles to one block, no copy), as upstream's one-block pattern does (bendlang/bend tests/run/stencil3d.bend). On a
# machine without a GPU, or with `--gpu off`, the same bang runs on the CPU threads.
#
# The bits: a cell of C is computed in tensor-array's order. Product p of a dot goes to partial sum p mod 4, in
# increasing p, each sum starting at +0.0, and the four sums are combined by TA.t8_comb with rot = k mod 4, as
# tensor-array's dotr and t8_dot do; an accumulating product adds the cell to C once with TA.cwr. The loop takes four
# steps of k a turn; the steps past k in the last turn add +0.0, which leaves a sum unchanged (a sum that starts at +0.0
# is never -0.0). Bend builds the device code with --fmad=false, so no multiply-add is fused. tests/bits.bend checks the
# bits against tensor-array on many shapes, on the GPU and the CPU.
#
# Not proved, and marked so: Array.fork and Array.join are @unsafe in Base, so the defs below that call them, or take
# a shared array, are @unsafe, and `bend main.bend` names every def that relies on them. Nothing checks that a leaf
# writes only its own tile of C, or never writes A or B; the tiles are disjoint by construction (leaf t writes tile t).
# All of it is Bend: no foreign code. Written by bend-ml's scripts/gen_tensor_gpu.py ({TM} x {TN} tiles).
import Base
import bend-ml-tensor-array@0.1.12.0/main.bend as TA

# the sizes and strides of one product: C[i][j] = sum_p A[i*arow + p*sa] * B[p*sb + j*bcol], n x m, k steps
type E is Data:
  E{{n: U32, m: U32, k: U32, tcols: U32, tiles: U32, arow: U32, sa: U32, sb: U32, bcol: U32, kg: Nat, rot: Nat, rem: U32, ac: Bool}}

# a product that counts only when its step is inside k (the last turn's steps past k add +0.0)
def pk(ok: Bool, +x: F32) -> F32:
  match ok:
    case True{{}}: x
    case False{{}}: 0.0

# the values of one turn: A's {TM} rows and B's {TN} columns at four consecutive steps, read through the arrays in order
type GR is Type:
  GR{{a: Array<F32>, b: Array<F32>, {j(v + ': F32' for v in V)}}}

""")

# getter chain: gx_i reads value i (of V), carrying values 0..i-1; the A values come from a, the B values from b
n_v = len(V)
for i in reversed(range(n_v)):
    known = [f"+{V[t]}: F32" for t in range(i)]
    head = f"def gr_{i}(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, {j(known) + ', ' if known else ''}"
    is_a = i < len(X)
    if is_a:
        head += f"rv: Array<F32> & F32, b: Array<F32>) -> GR:\n"
        body = "  match rv:\n    case Tuple{a, +" + V[i] + "}:\n"
    else:
        head += f"a: Array<F32>, rv: Array<F32> & F32) -> GR:\n"
        body = "  match rv:\n    case Tuple{b, +" + V[i] + "}:\n"
    vals = j(V[: i + 1])
    if i == n_v - 1:
        body += f"      GR{{a, b, {vals}}}\n"
    else:
        nxt = V[i + 1]
        args = f"ia, ib, arow, sa, sb, bcol, {vals}"
        if i + 1 < len(X):  # next is an A value: step g, row r
            g, r = divmod(i + 1, TM)
            body += f"      gr_{i + 1}({args}, Array.get(F32, a, (ia + ((sa * {g} : U32) + (arow * {r} : U32) : U32) : U32)), b)\n"
        else:
            g, c = divmod(i + 1 - len(X), TN)
            src = "a" if is_a else "a"
            if is_a:  # switching from a to b: a is the matched array, b the parameter
                body += f"      gr_{i + 1}({args}, a, Array.get(F32, b, (ib + ((sb * {g} : U32) + (bcol * {c} : U32) : U32) : U32)))\n"
            else:
                body += f"      gr_{i + 1}({args}, a, Array.get(F32, b, (ib + ((sb * {g} : U32) + (bcol * {c} : U32) : U32) : U32)))\n"
    o.append(head + body)
p(f"""# the turn whose A values start at ia and B values at ib
def turn(+ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>) -> GR:
  gr_0(ia, ib, arow, sa, sb, bcol, Array.get(F32, a, ia), b)

# the tile's cells after the four sums are combined, row by row
type T is Type:
  T{{a: Array<F32>, b: Array<F32>, {j(f'c{r}_{c}: F32' for (r, c) in cells)}}}

""")


def upd(guard):
    out = []
    for (r, c) in cells:
        for q in range(4):
            prod = f"(x{q}_{r} * y{q}_{c} : F32)"
            if guard:
                prod = f"pk(U32.is_lt({q}, rem), {prod})"
            out.append(f"({sum_name(q, r, c)} + {prod} : F32)")
    return out


sp = j("+" + s + ": F32" for s in S)
vp = j("+" + v for v in V)
fin = []
for (r, c) in cells:
    a = upd(True)[(r * TN + c) * 4: (r * TN + c) * 4 + 4]
    fin.append(f"TA.t8_comb(rot, {j(a)})")
p(f"""# kg more full turns; g holds the current turn's values. The last call (kg = 0) adds the turn that holds the steps
# past the last full turn (rem = k mod 4 of them), counting only those, and combines the sums.
@unsafe
def mm(kg: Nat, +ia: U32, +ib: U32, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +rem: U32, +rot: Nat, {sp}, g: GR) -> T:
  match kg g:
    case 0n GR{{a, b, {vp}}}:
      T{{a, b, {j(fin)}}}
    case 1n+q GR{{a, b, {vp}}}:
      mm(q, (ia + (sa * 4 : U32) : U32), (ib + (sb * 4 : U32) : U32), arow, sa, sb, bcol, rem, rot, {j(upd(False))}, turn((ia + (sa * 4 : U32) : U32), (ib + (sb * 4 : U32) : U32), arow, sa, sb, bcol, a, b))

# writes cell (i, j) of C when it is inside n x m
def wr(ok: Bool, +ac: Bool, +idx: U32, +x: F32, c: Array<F32>) -> Array<F32>:
  match ok:
    case True{{}}: TA.cwr(ac, idx, x, c)
    case False{{}}: c

""")
expr = "c"
for (r, c) in cells:
    ok = f"Bool.and(U32.is_lt((i0 + {r} : U32), n), U32.is_lt((j0 + {c} : U32), m))"
    idx = f"(((i0 + {r} : U32) * m : U32) + (j0 + {c} : U32) : U32)"
    expr = f"wr({ok}, ac, {idx}, c{r}_{c}, {expr})"
p(f"""# the tile at rows i0.., columns j0.. of C, from the loop's result; A and B are dropped (the caller holds them)
@unsafe
def put(+ac: Bool, +n: U32, +m: U32, +i0: U32, +j0: U32, c: Array<F32>, t: T) -> Array<F32>:
  match t:
    case T{{a, b, {j('+' + f'c{r}_{c}' for (r, c) in cells)}}}:
      {expr}

@unsafe
def tile(+e: E, +i0: U32, +j0: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match e:
    case E{{+n, +m, +k, +tcols, +tiles, +arow, +sa, +sb, +bcol, +kg, +rot, +rem, +ac}}:
      put(ac, n, m, i0, j0, c, mm(kg, (i0 * arow : U32), (j0 * bcol : U32), arow, sa, sb, bcol, rem, rot, {j('0.0' for _ in S)}, turn((i0 * arow : U32), (j0 * bcol : U32), arow, sa, sb, bcol, a, b)))

# leaf t computes tile t (row-major over the tiles); leaves past the last tile write nothing
@unsafe
def leaf_if(ok: Bool, +e: E, +t: U32, +tcols: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match ok:
    case True{{}}: tile(e, (U32.div(t, tcols) * {TM} : U32), (U32.mod(t, tcols) * {TN} : U32), a, b, c)
    case False{{}}: c

@unsafe
def leaf(+e: E, +t: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  match e:
    case E{{+n, +m, +k, +tcols, +tiles, +arow, +sa, +sb, +bcol, +kg, +rot, +rem, +ac}}:
      leaf_if(U32.is_lt(t, tiles), e, t, tcols, a, b, c)

@unsafe
def tree(+e: E, d: Nat, +t: U32, +a: Array<F32>, +b: Array<F32>, +c: Array<F32>) -> Array<F32>:
  match d:
    case 0n: leaf(e, t, a, b, c)
    case 1n++q:
      l r = tree(e, q, (t * 2 : U32), a, b, c) tree(e, q, ((t * 2 : U32) + 1 : U32), a, b, c)
      Array.join(F32, l, r)

# C made inside the bang: the root of the ! call allocates it on the device. Made on the host instead, a new C often
# reuses a block whose pages sit on the device, so zeroing it on the host pulls them back over PCIe and the bang moves
# them again (a 256 x 1536 product: 9.2 ms against 6.0, measured 2026-10-18). The leaves write every cell of C.
@unsafe
def tree_new(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, +cd: Nat) -> Array<F32>:
  tree(e, d, 0, a, b, Array.new(F32, cd, 0.0))

@unsafe
def bang_new(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, +cd: Nat) -> Array<F32>:
  tree_new!(e, d, a, b, cd)

# one bang over the tiles; A and B are shared with every leaf and come back unchanged
@unsafe
def bang(+e: E, d: Nat, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> Array<F32>:
  tree!(e, d, 0, a, b, c)

type ABC is Type:
  ABC{{a: Array<F32>, b: Array<F32>, c: Array<F32>}}

@unsafe
def run_e(+e: E, +tiles: U32, +a: Array<F32>, +b: Array<F32>, c: Array<F32>) -> ABC:
  ABC{{a, b, bang(e, TA.cap_depth(U32.to_nat(tiles)), a, b, c)}}

# the product's sizes and strides
def e_of(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32) -> E:
  E{{U32.from_nat(n), U32.from_nat(m), U32.from_nat(k), U32.from_nat(Nat.div(Nat.add(m, {TN - 1}n), {TN}n)), U32.from_nat(Nat.mul(Nat.div(Nat.add(n, {TM - 1}n), {TM}n), Nat.div(Nat.add(m, {TN - 1}n), {TN}n))), arow, sa, sb, bcol, Nat.div(k, 4n), Nat.mod(k, 4n), U32.from_nat(Nat.mod(k, 4n)), ac}}

def tiles(+n: Nat, +m: Nat) -> Nat:
  Nat.mul(Nat.div(Nat.add(n, {TM - 1}n), {TM}n), Nat.div(Nat.add(m, {TN - 1}n), {TN}n))

# C (n x m) = A · B with k steps into a new C made on the device
@unsafe
def gemm_new(+n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, +a: Array<F32>, +b: Array<F32>) -> ABC:
  ABC{{a, b, bang_new(e_of(False{{}}, n, m, k, arow, sa, sb, bcol), TA.cap_depth(tiles(n, m)), a, b, TA.cap_depth(Nat.mul(n, m)))}}

# C (n x m) = A · B with k steps, or C + A · B when ac, C[i][j] = sum_p A[i*arow + p*sa] * B[p*sb + j*bcol]
@unsafe
def gemm(+ac: Bool, +n: Nat, +m: Nat, +k: Nat, +arow: U32, +sa: U32, +sb: U32, +bcol: U32, a: Array<F32>, b: Array<F32>, c: Array<F32>) -> ABC:
  run_e(e_of(ac, n, m, k, arow, sa, sb, bcol), U32.from_nat(tiles(n, m)), a, b, c)

def mm_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMul<n, k, m>:
  match g:
    case ABC{{a, b, c}}: TA.MMul{{TA.Mat{{a}}, TA.Mat{{b}}, TA.Mat{{c}}}}

def mm_nt_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMulNT<n, k, m>:
  match g:
    case ABC{{a, b, c}}: TA.MMulNT{{TA.Mat{{a}}, TA.Mat{{b}}, TA.Mat{{c}}}}

def mm_tn_fin(-n: Nat, -k: Nat, -m: Nat, g: ABC) -> TA.MMulTN<n, k, m>:
  match g:
    case ABC{{a, b, c}}: TA.MMulTN{{TA.Mat{{a}}, TA.Mat{{b}}, TA.Mat{{c}}}}

# C = A · B   (n x k) · (k x m), tensor-array's TA.Mat.matmul bits
@unsafe
def matmul(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>) -> TA.MMul<n, k, m>:
  match a b:
    case TA.Mat{{x}} TA.Mat{{y}}:
      mm_fin(n, k, m, gemm_new(n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y))

# C + A · B, into C (n x m)
@unsafe
def matmul_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMul<n, k, m>:
  match a b c:
    case TA.Mat{{x}} TA.Mat{{y}} TA.Mat{{z}}:
      mm_fin(n, k, m, gemm(True{{}}, n, m, k, U32.from_nat(k), 1, U32.from_nat(m), 1, x, y, z))

# C = A · Bᵀ   (n x k) · (m x k)ᵀ: for weights stored with one row per output
@unsafe
def matmul_nt(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>) -> TA.MMulNT<n, k, m>:
  match a b:
    case TA.Mat{{x}} TA.Mat{{y}}:
      mm_nt_fin(n, k, m, gemm_new(n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y))

# C + A · Bᵀ, into C (n x m)
@unsafe
def matmul_nt_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<n, k>, b: TA.Mat<m, k>, c: TA.Mat<n, m>) -> TA.MMulNT<n, k, m>:
  match a b c:
    case TA.Mat{{x}} TA.Mat{{y}} TA.Mat{{z}}:
      mm_nt_fin(n, k, m, gemm(True{{}}, n, m, k, U32.from_nat(k), 1, 1, U32.from_nat(k), x, y, z))

# C = Aᵀ · B   (k x n)ᵀ · (k x m): weight gradient, Xᵀ · dY
@unsafe
def matmul_tn(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>) -> TA.MMulTN<n, k, m>:
  match a b:
    case TA.Mat{{x}} TA.Mat{{y}}:
      mm_tn_fin(n, k, m, gemm_new(n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y))

# C + Aᵀ · B, into C (n x m): dW += Xᵀ · dY
@unsafe
def matmul_tn_acc(+n: Nat, +k: Nat, +m: Nat, par: Nat, a: TA.Mat<k, n>, b: TA.Mat<k, m>, c: TA.Mat<n, m>) -> TA.MMulTN<n, k, m>:
  match a b c:
    case TA.Mat{{x}} TA.Mat{{y}} TA.Mat{{z}}:
      mm_tn_fin(n, k, m, gemm(True{{}}, n, m, k, 1, U32.from_nat(n), U32.from_nat(m), 1, x, y, z))
""")
open(OUT, "w").write("\n".join(o) if False else "".join(x if x.endswith("\n") else x + "\n" for x in o))
print(f"{OUT}: {TM}x{TN} tiles, {len(S)} sums, {len(V)} values a turn")
