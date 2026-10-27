# The words of the random fills, for reference/test_rng.py's NumPy twin: per line, the Gaussian's and the uniform's
# 32 bits of one cell of counter 77, first cells 0 .. 4095 from fill_gauss and fill_uniform (one bang each), then cells
# 2^32 - 4096 .. 2^32 - 1 from rng_gauss and rng_uniform (the indexes wrap there). The text is built from its last line
# back, each line put in front (a String appended at its end is copied).
import Base
import ../main.bend as TG

def bits(x: F32) -> U32:
  match x:
    case F32{w}: U32{w}

def row(+g: F32, +u: F32) -> String:
  U32.show(bits(g)) ++ " " ++ U32.show(bits(u)) ++ "\n"

def from_arrays(k: Nat, +i: U32, acc: String, rg: Array<F32> & F32, ru: Array<F32> & F32) -> String:
  match k rg ru:
    case 0n Tuple{a, g} Tuple{b, u}: acc
    case 1n+p Tuple{a, +g} Tuple{b, +u}:
      from_arrays(p, (i - 1 : U32), row(g, u) ++ acc, Array.get(F32, a, (i - 1 : U32)), Array.get(F32, b, (i - 1 : U32)))

def from_cells(k: Nat, +i: U32, acc: String) -> String:
  match k:
    case 0n: acc
    case 1n+p: from_cells(p, (i - 1 : U32), row(TG.rng_gauss(77, i), TG.rng_uniform(77, i)) ++ acc)

@unsafe
def words(u: Unit) -> String:
  from_arrays(4096n, 4095, from_cells(4096n, 4294967295, ""), Array.get(F32, TG.fill_gauss(77, 4096n, 12n), 4095), Array.get(F32, TG.fill_uniform(77, 4096n, 12n), 4095))

# This program takes no arguments (runtime options go between +RTS and -RTS): any argument is an error.
def alone(args: List<String>, go: IO(Unit)) -> IO(Unit):
  match args:
    case Nil{}:
      go
    case Con{p, Nil{}}:
      go
    case Con{p, Con{a, t}}:
      do IO<Unit>:
        IO.print_err("unexpected argument " ++ a ++ ": this program takes none (runtime options go between +RTS and -RTS)")
        IO.exit(2)

@unsafe
def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    alone(args, IO.print(words(Unit{})))
