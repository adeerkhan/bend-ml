# The cost of Array.new inside a bang: a root that allocates 2^c cells and runs a 16-leaf tree that writes nothing, for
# c = 16, 18, 20, 22, 5 bangs each, in that order; time each bang's first bend_dev launch (grid 1) with Nsight Systems.
import Base

def at0(r: Array<F32> & F32) -> F32:
  match r:
    case Tuple{a, +x}: x

@unsafe
def tr(d: Nat, +o: Array<F32>) -> F32:
  match d:
    case 0n: at0(Array.get(F32, o, 0))
    case 1n++p:
      a b = tr(p, o) tr(p, o)
      (a + b : F32)

@unsafe
def root(+c: Nat) -> F32:
  tr(4n, Array.new(F32, c, 0.0))

@unsafe
def go(+c: Nat) -> F32:
  root!(c)

@unsafe
def loop(k: Nat, +c: Nat, acc: F32) -> F32:
  match k:
    case 0n: acc
    case 1n+q: loop(q, c, (acc + go(c) : F32))

@unsafe
def main() -> IO(Unit):
  IO.print(F32.show((((loop(5n, 16n, 0.0) + loop(5n, 18n, 0.0) : F32) + loop(5n, 20n, 0.0) : F32) + loop(5n, 22n, 0.0) : F32)))
