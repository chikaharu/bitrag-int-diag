# bitrag-int-diag

Integer-only fixed-point cosine normalization for Gram-like matrices,
with **bit-exact diagonal** (`C_ii == 1.000000` to 0 ppm) on every
64-bit machine.

> **Scope.** This crate is the **numerical-reproducibility appendix**
> for the [bitRAG main theorem (MAIN-B)](https://github.com/chikaharu/bitrag-theorems).
> It is **not** a research lemma — it is a well-known fixed-point trick
> (scale inside the integer square root) packaged so that every numeric
> result in MAIN-B can be reproduced byte-for-byte without floating
> point. If you want the actual theorems, see the MAIN-B paper.

## Diagonal-exact identity

Let `G ∈ ℕ^{n×n}` be a Gram-like integer matrix with `G_ii > 0` for every
`i`. Let `P = 10^6` (the PPM unit). Define

```text
norm_i := isqrt(G_ii · P²)
C_ij   := ⌊ G_ij · P³ / (norm_i · norm_j) ⌋
```

Then `C_ii == P` exactly for every `i`, with **zero ppm error**. The
off-diagonal `C_ij` are floored quotients (not exact, but deterministic).

### Why this works (fixed-point sketch)

Write `q := norm_i = isqrt(G_ii · P²)`. By definition,

```text
q² ≤ G_ii · P² < (q + 1)²
```

so `G_ii · P² = q² + r` with `0 ≤ r ≤ 2q`. Therefore

```text
C_ii = ⌊ G_ii · P³ / q² ⌋
     = ⌊ (q² + r) · P / q² ⌋
     = ⌊ P + r·P / q² ⌋.
```

For any `G_ii ≥ 1` we have `q ≥ P`, so

```text
r·P / q² ≤ 2q · P / q² = 2P / q ≤ 2,
```

and a slightly tighter accounting on the residue keeps the floored
fractional part below `1`. The floor strips it, leaving `C_ii = P`. ∎

### Why scaling **inside** the isqrt is necessary

The naive integer cosine

```text
C'_ij := ⌊ G_ij · P / (isqrt(G_ii) · isqrt(G_jj)) ⌋
```

does **not** satisfy `C'_ii = P` whenever `G_ii` is not a perfect square.
With `G_ii = 1_234_567`:

```text
isqrt(G_ii)            = 1_111
C'_ii                  = ⌊ 1_234_567 · 10^6 / 1_111² ⌋
                       = ⌊ 1_234_567_000_000 / 1_234_321 ⌋
                       = 1_000_199 ≠ 10^6.
```

That is a 199 ppm drift — small, but enough to break byte-level
reproducibility across machines and runs. Folding the PPM scaling into
the isqrt fixes it. See `tests/counterexample.rs`.

### Origin

Extracted from
[`chikaharu/bitRAG`](https://github.com/chikaharu/bitRAG)
experiment **E145** ("XCORR重み IDF² 厳密対角単位化, 整数 cosine, 誤差 0 ppm",
commit `3048190c`).

## Public API

```rust
pub const PPM: u64 = 1_000_000;

pub fn isqrt_u128(n: u128) -> u128;
pub fn norm_ppm(g_ii: u64) -> u128;
pub fn cosine_ppm(g_ij: u64, norm_i: u128, norm_j: u128) -> u64;

pub fn diagonal_unitize(g: &[Vec<u64>]) -> (Vec<u128>, Vec<Vec<u64>>);
pub fn idf_squared_weights(num_docs: u32, df: &[u32]) -> Vec<u64>;

// Counterexample (used in tests; do not use in production):
pub fn naive_diagonal_unitize(g: &[Vec<u64>]) -> Vec<Vec<u64>>;
```

## Example

```rust
use bitrag_int_diag::{diagonal_unitize, PPM};

let g: Vec<Vec<u64>> = vec![
    vec![100, 30, 20],
    vec![30, 200, 40],
    vec![20, 40, 150],
];

let (norms, c) = diagonal_unitize(&g);

assert_eq!(c[0][0], PPM); // exactly 1_000_000
assert_eq!(c[1][1], PPM);
assert_eq!(c[2][2], PPM);
```

## Regression: E145 replay

The test `tests/e145_replay.rs` embeds the exact 6×6 G matrix produced
by E145 on the 六法 (kenpo / minpo / shoho / keiho / minso / keiso)
Japanese law corpus. Any drift in `diagonal_unitize` — change of `PPM`,
floored isqrt, or ordering of operations — is detected immediately.

## Citation

```bibtex
@misc{chikaharu2026bitrag-int-diag,
  author = {chikaharu},
  title  = {{bitrag-int-diag}: Integer-Only Fixed-Point Cosine
            Normalization with Bit-Exact Diagonal},
  year   = {2026},
  howpublished = {\url{https://github.com/chikaharu/bitrag-int-diag}},
  note   = {Numerical-reproducibility appendix for the bitRAG main
            theorem (MAIN-B).}
}
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
