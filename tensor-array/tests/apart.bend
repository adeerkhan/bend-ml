# The lemma the round trip needs, checked where it can be checked today: on a
# perfect tree of 2^d leaves, a write at j and a read at k reach different leaves
# whenever j and k differ, for every pair of indices.
#
#   bend tensor-array/tests/apart.bend
#
# The law `to_list (from_list xs) == xs` needs that lemma symbolically, and the
# lemma needs four U32 facts Base does not state. Until they are proved, this is
# the oracle: the semantic property, computed on every pair up to depth 7.
#
# n = 2^d and every index used is below n, so nothing wraps; the checks are
#   set at j, read at j:  v[j] is 1.0
#   set at j, read at k:  v[k] is still 0.0 when j != k
import Base
import ../main.bend as TA

def bump(ok: Bool, +bad: Nat) -> Nat:
  match ok:
    case True{}:
      bad
    case False{}:
      Nat.add(bad, 1n)

def second_read(r: Array<F32> & F32, +ok_j: Bool, +bad: Nat) -> Nat:
  match r:
    case Tuple{_, +vk}:
      bump(Bool.and(ok_j, F32.is_eq(vk, 0.0)), bad)

def first_read(r: Array<F32> & F32, +k: U32, +bad: Nat) -> Nat:
  match r:
    case Tuple{b, +vj}:
      second_read(Array.get(F32, b, k), F32.is_eq(vj, 1.0), bad)

def pair_bad(+d: Nat, +j: U32, +k: U32, +bad: Nat) -> Nat:
  a = Array.new(F32, d, 0.0)
  b = Array.set(F32, a, j, 1.0)
  first_read(Array.get(F32, b, j), k, bad)

def pair_of(eq: Bool, +d: Nat, +j: U32, +k: U32, +bad: Nat) -> Nat:
  match eq:
    case True{}:
      bad
    case False{}:
      pair_bad(d, j, k, bad)

# one row: k from 0 to n-1, every k != j
def row(+fuel: Nat, +d: Nat, +j: Nat, +k: Nat, +n: Nat, +bad: Nat) -> Nat:
  match fuel:
    case 0n:
      bad
    case 1n+rest:
      row(rest, d, j, Nat.add(k, 1n), n,
        pair_of(Nat.is_eq(j, k), d, U32.from_nat(j), U32.from_nat(k), bad))

# j from 0 to n-1
def rows(+fuel: Nat, +d: Nat, +j: Nat, +n: Nat, +bad: Nat) -> Nat:
  match fuel:
    case 0n:
      bad
    case 1n+rest:
      rows(rest, d, Nat.add(j, 1n), n, row(n, d, j, 0n, n, bad))

def depth(+d: Nat, +n: Nat) -> Nat:
  rows(n, d, 0n, n, 0n)

def upto(+left: Nat, +d: Nat, +n: Nat, +bad: Nat) -> Nat:
  match left:
    case 0n:
      bad
    case 1n+rest:
      upto(rest, Nat.add(d, 1n), Nat.mul(n, 2n), Nat.add(bad, depth(d, n)))

def report(+bad: Nat) -> IO(Unit):
  do IO<Unit>:
    IO.print(String.append("apart_failures: ", Nat.show(bad)))
    return Unit{}

def main() -> IO(Unit):
  report(upto(8n, 0n, 1n, 0n))
