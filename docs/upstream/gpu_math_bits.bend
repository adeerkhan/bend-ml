# Do F32's transcendental primitives give the same bits on the GPU as on the host? A bang over 2^14 leaves, each leaf
# computing f(x) for 64 inputs into one shared block (the one-block pattern), against the host's own f(x) on the same
# inputs. Inputs: x = (i - 2^19) / 4096 for i < 2^20 (-128 .. 128, step 1/4096) mixed with large and tiny values.
# `./mathbits <f>`: f = exp, log, sqrt, tanh, sin, cos, pow (x^1.5): prints the number of differing words of 2^20.
import Base

def inp(+i: U32) -> F32:
  Bool.pick(F32, U32.is_eq(U32.mod(i, 97), 0), (U32.to_f32(i) * 37.5 : F32), Bool.pick(F32, U32.is_eq(U32.mod(i, 89), 0), (U32.to_f32(i) / 1000000000.0 : F32), ((U32.to_f32(i) - 524288.0 : F32) / 4096.0 : F32)))

def f(+w: U32, +x: F32) -> F32:
  Bool.pick(F32, U32.is_eq(w, 0), F32.exp(x), Bool.pick(F32, U32.is_eq(w, 1), F32.log(F32.max(x, 0.0000001)), Bool.pick(F32, U32.is_eq(w, 2), F32.sqrt(F32.max(x, 0.0)), Bool.pick(F32, U32.is_eq(w, 3), F32.tanh(x), Bool.pick(F32, U32.is_eq(w, 4), F32.sin(x), Bool.pick(F32, U32.is_eq(w, 5), F32.cos(x), F32.pow(F32.max(x, 0.0), 1.5)))))))

def bits(x: F32) -> U32:
  match x:
    case F32{v}: U32{v}

# 64 cells of leaf t: i = t*64 + j
def cells(k: Nat, +w: U32, +i: U32, a: Array<U32>) -> Array<U32>:
  match k:
    case 0n: a
    case 1n+q: cells(q, w, (i + 1 : U32), Array.set(U32, a, i, bits(f(w, inp(i)))))

@unsafe
def tree(+d: Nat, +w: U32, +t: U32, +a: Array<U32>) -> Array<U32>:
  match d:
    case 0n: cells(64n, w, (t * 64 : U32), a)
    case 1n+p:
      l r = tree(p, w, (t * 2 : U32), a) tree(p, w, ((t * 2 : U32) + 1 : U32), a)
      Array.join(U32, l, r)

@unsafe
def dev(+w: U32) -> Array<U32>:
  tree!(14n, w, 0, Array.new(U32, 20n, 0))

def cmp(k: Nat, +w: U32, +i: U32, acc: U32, r: Array<U32> & U32) -> U32:
  match k r:
    case 0n Tuple{a, +v}: (acc + Bool.pick(U32, U32.is_eq(v, bits(f(w, inp(i)))), 0, 1) : U32)
    case 1n+q Tuple{a, +v}: cmp(q, w, (i + 1 : U32), (acc + Bool.pick(U32, U32.is_eq(v, bits(f(w, inp(i)))), 0, 1) : U32), Array.get(U32, a, (i + 1 : U32)))

@unsafe
def run(+w: U32) -> IO(Unit):
  IO.print(U32.show(w) ++ " differing " ++ U32.show(cmp(1048575n, w, 0, 0, Array.get(U32, dev(w), 0))) ++ " of 1048576")

def code(+s: String) -> U32:
  Bool.pick(U32, String.eq(s, "exp"), 0, Bool.pick(U32, String.eq(s, "log"), 1, Bool.pick(U32, String.eq(s, "sqrt"), 2, Bool.pick(U32, String.eq(s, "tanh"), 3, Bool.pick(U32, String.eq(s, "sin"), 4, Bool.pick(U32, String.eq(s, "cos"), 5, 6))))))

@unsafe
def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{s, Nil{}}}: run(code(s))
    case _: IO.die(Unit, 2, "usage: mathbits exp|log|sqrt|tanh|sin|cos|pow")

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
