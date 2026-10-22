# A dot product of two 1,048,576-number arrays: acc + a[i] * b[i], one running sum, as a model's products do.
# bend fma_dot.bend -o fma_dot.c; then count `vfmadd` in the binary clang makes from it (fma-codegen.md).
import Base

def dot(k: Nat, +i: U32, acc: F32, ra: Array<F32> & F32, rb: Array<F32> & F32) -> F32:
  match k ra rb:
    case 0n Tuple{a, +x} Tuple{b, +y}: (acc + (x * y : F32) : F32)
    case 1n+q Tuple{a, +x} Tuple{b, +y}:
      dot(q, (i + 1 : U32), (acc + (x * y : F32) : F32), Array.get(F32, a, (i + 1 : U32)), Array.get(F32, b, (i + 1 : U32)))

def main() -> IO(Unit):
  IO.print(F32.show(dot(1048575n, 0, 0.0, Array.get(F32, Array.new(F32, 20n, 0.5), 0), Array.get(F32, Array.new(F32, 20n, 0.25), 0))))
