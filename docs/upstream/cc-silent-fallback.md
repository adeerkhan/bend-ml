# Draft issue for bendlang/bend: a `$CC` that cannot run is skipped without a word

Status: posted as https://github.com/bendlang/bend/issues/1802 (2026-10-20, by Claude Code with Renan's go-ahead), in the Bug
form's fields, with the three cases below run on a one-line program (all exit 0, silent, built by clang). Checked for duplicates on 2026-10-20 (searched "CC environment", "cc_find",
"compiler flags", "cflags", "clang flags", "march", "native target"): none for this bug. The closest is PR #1461 (read
`$CFLAGS` for `bend -o`, closed 2026-10-14): the maintainers want to keep the build CLI small and point to `bend f.bend
-o f.c` plus clang by hand for custom flags. This draft does not ask for flags again; it asks only that a `$CC` the user
set is not dropped silently. Reproduced on Bend 2.0.37; `cc_find` is unchanged on upstream main at 40402b42
(2026-10-20).

Evidence: bend2-lab `bendtorch/runs/2026-10-20-block-a/cc-flags.txt` (the three builds below, on a BendTorch demo).

---

**Title:** `bend -o`: a `$CC` that is not a runnable clang is skipped silently, and the build uses plain `clang`

## What happens

`cc_find` (`bend2/main.ts`) tries `$CC`, then `clang`, then every `clang-NN` on `PATH`, and takes the first whose
`--version` says clang 14 or newer. It runs each name with `spawnSync` and no shell, so `$CC` must be one executable.
When it is not (a common way to write it is with flags, `CC="clang -mavx2"`, as `make` and most build systems
accept), the name is not found, `cc_find` moves on to `clang`, and the build succeeds with no message: the user believes
the binary was built with their compiler and it was not.

```sh
CC="clang -mavx2" bend prog.bend -o prog    # exit 0, no output
objdump -d prog | grep -c ymm               # 0: built by plain clang
```

The same happens (run on 2026-10-20 before posting) for a typo in a path (`CC=/usr/bin/clang-1` builds with `clang`) or a `$CC` that is gcc (it is skipped
as "not clang", but only an error that lists every candidate says so, and there is none when a later one works).

## Expected

When `$CC` is set and cannot be used, say so. Either of:

- fail, as the error after the last candidate already does (`Error: bend needs clang 14 or newer ... (found <CC>, which
  is not clang, ...)`), since the user asked for that compiler; or
- go on with the next candidate but print one line first, e.g. `bend: $CC="clang -mavx2" is not a runnable clang 14+;
  building with clang`.

The second keeps today's builds working; the first matches what an explicit `$CC` means elsewhere.

## Why it matters

Measured here: the same emitted C of a SmolLM2-135M training step built with plain `clang -O3` against `-march=native`
is 12.5% faster on one thread (same bits, no FMA, the emitted C's `fp contract(off)` holds). Following #1461's advice
(`-o f.c`, then clang by hand) works; a wrapper script set as `$CC` (`exec clang -march=native "$@"`) works too. The
trap is the first thing people try, which looks like it worked.
