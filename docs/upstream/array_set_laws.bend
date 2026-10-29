# Two laws about Base's Array, proved outside Base with Bend 2.0.39 (`bend array_set_laws.bend --verdict`):
# - size_set: Array.set keeps Array.size;
# - get_after_set: what Array.set writes at i, Array.get reads back at i.
# Both hold for every i (the two descents wrap i the same way) and for every tree Base can build, balanced
# or not, since a write replaces one leaf and leaves the tree's shape alone. "The size" is the second
# component of the pair Array.size returns, read with Base's Pair.snd (sz below); the value is Pair.snd
# of the pair Array.get returns.
#
# Why the proof is shaped this way: an Array is affine, and so is it inside a proof, so a proof may use a
# given array once. The obvious lemmas each need the same subtree again (one to say Array.size of it is
# (it, its size), one for the induction step, one to open the pair the descent returns), so they cannot
# be combined. Instead, one pass over the tree (facts) proves all six facts the step needs at once; each
# fact is an equation, and equations can be reused (+). A second pass (copy) hands back the same array
# and its size as a plain value, which the top-level proof needs to pick the first branch.
#
# The trailing Unit in Facts and Copy works around bendlang/bend#1077: a + binder on the last field of
# a nested pair fails to check, so no fact sits in the last field.

import Base

# sz(T, a): the size of a as a plain U32, the second component of the pair Array.size returns
def sz(-T: Data, a: Array<T>) -> U32:
  Pair.snd(Array<T>, U32, Array.size(T, a))

# the three results the descents build: the swapped pair, the written array, and the read pair
def G(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Array<T> & T:
  Array.swap.go(T, a, n, j, v, z)

def NA(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Array<T>:
  Array.set.fin(T, G(T, a, n, j, v, z))

def K(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Array<T> & T:
  Array.get.go(T, NA(T, a, n, j, v, z), n, j, z)

# Facts: what one pass over the tree a proves, for any n, j, v, z (n need not be a's real size):
# 1. Array.size(a) is the pair (a, sz(a));
# 2. the pair the write's descent returns is (the written array, the old value);
# 3. Array.size of the written array is the pair (written array, its sz);
# 4. the written array has a's sz;
# 5. reading the written array along the same path gives v;
# 6. the pair the read returns is (its first component, its second).
def Facts(-T: Data, -a: Array<T>, -n: U32, -j: U32, -v: T, -z: Bool) -> Type:
  {(a, sz(T, a)) == Array.size(T, a) : Array<T> & U32} & {(NA(T, a, n, j, v, z), Pair.snd(Array<T>, T, G(T, a, n, j, v, z))) == G(T, a, n, j, v, z) : Array<T> & T} & {(NA(T, a, n, j, v, z), sz(T, NA(T, a, n, j, v, z))) == Array.size(T, NA(T, a, n, j, v, z)) : Array<T> & U32} & {sz(T, NA(T, a, n, j, v, z)) == sz(T, a) : U32} & {Pair.snd(Array<T>, T, K(T, a, n, j, v, z)) == v : T} & {(Pair.fst(Array<T>, T, K(T, a, n, j, v, z)), Pair.snd(Array<T>, T, K(T, a, n, j, v, z))) == K(T, a, n, j, v, z) : Array<T> & T} & Unit

# Array.size hands back the array it measured, next to sz(T, a)
def size_pair(-T: Data, a: Array<T>) -> {(a, sz(T, a)) == Array.size(T, a) : Array<T> & U32}:
  match a:
    case ALeaf{x}:
      {==}
    case ANode{xs, ys}:
      %size_pair(T, xs) : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}
      {==}

def node_t(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, f: Facts(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))) -> Facts(T, ANode{xs, ys}, n, j, v, True{}):
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  match r4:
                    case Tuple{+c5, r5}:
                      match r5:
                        case Tuple{+c6, u}:
                          ({(%c1 : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{xs, ys}, sz(T, ANode{xs, ys})) == Array.size(T, ANode{xs, ys}) : Array<T> & U32}}, {(%c2 : {(Array.set.fin(T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, True{}), Pair.snd(Array<T>, T, G(T, ANode{xs, ys}, n, j, v, True{}))) == G(T, ANode{xs, ys}, n, j, v, True{}) : Array<T> & T}}, {(%c2 : {(Array.set.fin(T, Array.swap.lo(T, ys, _)), sz(T, Array.set.fin(T, Array.swap.lo(T, ys, _)))) == Array.size(T, Array.set.fin(T, Array.swap.lo(T, ys, _))) : Array<T> & U32}; %c3 : {(ANode{Array.set.fin(T, G(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))), ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, True{}), sz(T, NA(T, ANode{xs, ys}, n, j, v, True{}))) == Array.size(T, NA(T, ANode{xs, ys}, n, j, v, True{})) : Array<T> & U32}}, {(%c2 : {sz(T, Array.set.fin(T, Array.swap.lo(T, ys, _))) == sz(T, ANode{xs, ys}) : U32}; %c3 : {Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) == sz(T, ANode{xs, ys}) : U32}; %c1 : {U32.shl(sz(T, Array.set.fin(T, G(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))))) == Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) : U32}; %c4 : {U32.shl(sz(T, Array.set.fin(T, G(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n))))))) == U32.shl(_) : U32}; {==}) : {sz(T, NA(T, ANode{xs, ys}, n, j, v, True{})) == sz(T, ANode{xs, ys}) : U32}}, {(%c2 : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{})) == v : T}; %c6 : {Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _)) == v : T}; c5) : {Pair.snd(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, True{})) == v : T}}, {(%c2 : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.lo(T, ys, _)), n, j, True{}) : Array<T> & T}; %c6 : {(Pair.fst(Array<T>, T, Array.swap.lo(T, ys, _)), Pair.snd(Array<T>, T, Array.swap.lo(T, ys, _))) == Array.swap.lo(T, ys, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, True{})), Pair.snd(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, True{}))) == K(T, ANode{xs, ys}, n, j, v, True{}) : Array<T> & T}}, Unit{})

def node_f(-T: Data, -xs: Array<T>, -ys: Array<T>, +n: U32, +j: U32, +v: T, +s: {(xs, sz(T, xs)) == Array.size(T, xs) : Array<T> & U32}, f: Facts(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))) -> Facts(T, ANode{xs, ys}, n, j, v, False{}):
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  match r4:
                    case Tuple{+c5, r5}:
                      match r5:
                        case Tuple{+c6, u}:
                          ({(%s : {(ANode{xs, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{xs, ys}, sz(T, ANode{xs, ys})) == Array.size(T, ANode{xs, ys}) : Array<T> & U32}}, {(%c2 : {(Array.set.fin(T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, False{}), Pair.snd(Array<T>, T, G(T, ANode{xs, ys}, n, j, v, False{}))) == G(T, ANode{xs, ys}, n, j, v, False{}) : Array<T> & T}}, {(%c2 : {(Array.set.fin(T, Array.swap.hi(T, xs, _)), sz(T, Array.set.fin(T, Array.swap.hi(T, xs, _)))) == Array.size(T, Array.set.fin(T, Array.swap.hi(T, xs, _))) : Array<T> & U32}; %s : {(ANode{xs, Array.set.fin(T, G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n)))))}, Pair.snd(Array<T>, U32, Array.size.node(T, Array.set.fin(T, G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))), _))) == Array.size.node(T, Array.set.fin(T, G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))), _) : Array<T> & U32}; {==}) : {(NA(T, ANode{xs, ys}, n, j, v, False{}), sz(T, NA(T, ANode{xs, ys}, n, j, v, False{}))) == Array.size(T, NA(T, ANode{xs, ys}, n, j, v, False{})) : Array<T> & U32}}, {(%c2 : {sz(T, Array.set.fin(T, Array.swap.hi(T, xs, _))) == sz(T, ANode{xs, ys}) : U32}; %s : {Pair.snd(Array<T>, U32, Array.size.node(T, Array.set.fin(T, G(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n))))), _)) == Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) : U32}; {==}) : {sz(T, NA(T, ANode{xs, ys}, n, j, v, False{})) == sz(T, ANode{xs, ys}) : U32}}, {(%c2 : {Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{})) == v : T}; %c6 : {Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _)) == v : T}; c5) : {Pair.snd(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, False{})) == v : T}}, {(%c2 : {(Pair.fst(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{})), Pair.snd(Array<T>, T, Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{}))) == Array.get.go(T, Array.set.fin(T, Array.swap.hi(T, xs, _)), n, j, False{}) : Array<T> & T}; %c6 : {(Pair.fst(Array<T>, T, Array.swap.hi(T, xs, _)), Pair.snd(Array<T>, T, Array.swap.hi(T, xs, _))) == Array.swap.hi(T, xs, _) : Array<T> & T}; {==}) : {(Pair.fst(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, False{})), Pair.snd(Array<T>, T, K(T, ANode{xs, ys}, n, j, v, False{}))) == K(T, ANode{xs, ys}, n, j, v, False{}) : Array<T> & T}}, Unit{})

def facts(-T: Data, a: Array<T>, +n: U32, +j: U32, +v: T, +z: Bool) -> Facts(T, a, n, j, v, z):
  match a z:
    case ALeaf{x} _:
      ({==}, {==}, {==}, {==}, {==}, {==}, Unit{})
    case ANode{xs, ys} True{}:
      node_t(T, xs, ys, n, j, v, facts(T, xs, U32.shr(n), j, v, U32.is_lt(j, U32.shr(U32.shr(n)))))
    case ANode{xs, ys} False{}:
      node_f(T, xs, ys, n, j, v, size_pair(T, xs), facts(T, ys, U32.shr(n), U32.sub(j, U32.shr(n)), v, U32.is_lt(U32.sub(j, U32.shr(n)), U32.shr(U32.shr(n)))))

# copy(T, a): one pass over a hands back the same array b, its size n as a value, and the facts
# that tie them to a: b is a, n is sz(T, b), and Array.size(b) is the pair (b, sz(T, b))
def Copy(-T: Data, -a: Array<T>) -> Type:
  &b: Array<T> -> &n: U32 -> {b == a : Array<T>} & {n == sz(T, b) : U32} & {(b, sz(T, b)) == Array.size(T, b) : Array<T> & U32} & Unit

def copy_node(-T: Data, -xs: Array<T>, ys: Array<T>, r: Copy(T, xs)) -> Copy(T, ANode{xs, ys}):
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+m, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  match r4:
                    case Tuple{+s, u}:
                      (ANode{b, ys}, U32.shl(m),
                        {(%p : {ANode{b, ys} == ANode{_, ys} : Array<T>}; {==}) : {ANode{b, ys} == ANode{xs, ys} : Array<T>}},
                        {(%s : {U32.shl(m) == Pair.snd(Array<T>, U32, Array.size.node(T, ys, _)) : U32}; %q : {U32.shl(m) == U32.shl(_) : U32}; {==}) : {U32.shl(m) == sz(T, ANode{b, ys}) : U32}},
                        {(%s : {(ANode{b, ys}, Pair.snd(Array<T>, U32, Array.size.node(T, ys, _))) == Array.size.node(T, ys, _) : Array<T> & U32}; {==}) : {(ANode{b, ys}, sz(T, ANode{b, ys})) == Array.size(T, ANode{b, ys}) : Array<T> & U32}},
                        Unit{})

def copy(-T: Data, a: Array<T>) -> Copy(T, a):
  match a:
    case ALeaf{x}:
      (ALeaf{x}, 1, {==}, {==}, {==}, Unit{})
    case ANode{xs, ys}:
      copy_node(T, xs, ys, copy(T, xs))

# the masked index and the first branch, as Array.swap.at and Array.get.at compute them from a size n
def J(+i: U32, +n: U32) -> U32:
  U32.and(i, U32.sub(n, 1))

def Z(+i: U32, +n: U32) -> Bool:
  U32.is_lt(J(i, n), U32.shr(n))

def close_size(-T: Data, -b: Array<T>, +n: U32, +i: U32, +v: T, +q: {n == sz(T, b) : U32}, f: Facts(T, b, n, J(i, n), v, Z(i, n))) -> {sz(T, Array.set(T, b, i, v)) == sz(T, b) : U32}:
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  %c1 : {sz(T, Array.set.fin(T, Array.swap.at(T, i, v, _))) == sz(T, b) : U32}
                  %q : {sz(T, Array.set.fin(T, Array.swap.go(T, b, _, J(i, _), v, Z(i, _)))) == _ : U32}
                  Equal.trans(U32, sz(T, NA(T, b, n, J(i, n), v, Z(i, n))), sz(T, b), n, c4, Equal.sym(U32, n, sz(T, b), q))

def close_get(-T: Data, -b: Array<T>, +n: U32, +i: U32, +v: T, +q: {n == sz(T, b) : U32}, f: Facts(T, b, n, J(i, n), v, Z(i, n))) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, b, i, v), i)) == v : T}:
  match f:
    case Tuple{+c1, r1}:
      match r1:
        case Tuple{+c2, r2}:
          match r2:
            case Tuple{+c3, r3}:
              match r3:
                case Tuple{+c4, r4}:
                  match r4:
                    case Tuple{+c5, r5}:
                      %c1 : {Pair.snd(Array<T>, T, Array.get.at(T, i, Array.size(T, Array.set.fin(T, Array.swap.at(T, i, v, _))))) == v : T}
                      %q : {Pair.snd(Array<T>, T, Array.get.at(T, i, Array.size(T, Array.set.fin(T, Array.swap.go(T, b, _, J(i, _), v, Z(i, _)))))) == v : T}
                      %c3 : {Pair.snd(Array<T>, T, Array.get.at(T, i, _)) == v : T}
                      %Equal.trans(U32, n, sz(T, b), sz(T, NA(T, b, n, J(i, n), v, Z(i, n))), q, Equal.sym(U32, sz(T, NA(T, b, n, J(i, n), v, Z(i, n))), sz(T, b), c4)) : {Pair.snd(Array<T>, T, Array.get.go(T, NA(T, b, n, J(i, n), v, Z(i, n)), _, J(i, _), Z(i, _))) == v : T}
                      c5

def top_size(-T: Data, -a: Array<T>, +i: U32, +v: T, r: Copy(T, a)) -> {sz(T, Array.set(T, a, i, v)) == sz(T, a) : U32}:
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+n, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  %p : {sz(T, Array.set(T, _, i, v)) == sz(T, _) : U32}
                  close_size(T, b, n, i, v, q, facts(T, b, n, J(i, n), v, Z(i, n)))

def top_get(-T: Data, -a: Array<T>, +i: U32, +v: T, r: Copy(T, a)) -> {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, a, i, v), i)) == v : T}:
  match r:
    case Tuple{b, r1}:
      match r1:
        case Tuple{+n, r2}:
          match r2:
            case Tuple{+p, r3}:
              match r3:
                case Tuple{+q, r4}:
                  %p : {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, _, i, v), i)) == v : T}
                  close_get(T, b, n, i, v, q, facts(T, b, n, J(i, n), v, Z(i, n)))

# LAW: writing v at i keeps the array's size (any i: the index wraps around)
law size_set:
  for -T: Data
  for a: Array<T>
  for +i: U32
  for +v: T
  {sz(T, Array.set(T, a, i, v)) == sz(T, a) : U32}

def size_set(T, a, i, v):
  top_size(T, a, i, v, copy(T, a))

# LAW: what Array.set writes at i, Array.get reads at i (any i: both wrap it the same way)
law get_after_set:
  for -T: Data
  for a: Array<T>
  for +i: U32
  for +v: T
  {Pair.snd(Array<T>, T, Array.get(T, Array.set(T, a, i, v), i)) == v : T}

def get_after_set(T, a, i, v):
  top_get(T, a, i, v, copy(T, a))

# the laws, used on an array of 8 F32s (index 13 wraps to 5)
def use_it(a: Array<F32>) -> {Pair.snd(Array<F32>, F32, Array.get(F32, Array.set(F32, a, 13, 1.5), 13)) == 1.5 : F32}:
  get_after_set(F32, a, 13, 1.5)

def second(r: Array<F32> & F32) -> F32:
  (b, x) = r
  x

def main() -> F32:
  second(Array.get(F32, Array.set(F32, [0.0 : F32*8n], 13, 1.5), 13))
