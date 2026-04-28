# bitrag-int-diag

**Integer-exact diagonal unitization** of Gram-like matrices via integer
square root, with **0 ppm diagonal error**.

This crate is **Lemma A** of the bitRAG main theorem
([MAIN-B](https://github.com/chikaharu/bitrag-theorems)): it provides the
algebraic primitive that lets the F2 retrieval scaling law be stated in
exact integer arithmetic, with no floating-point drift.

## Theorem A (Integer IDF² Diagonal Unitization)

**Statement.** Let `G ∈ ℕ^{n×n}` be a Gram-like integer matrix with
`G_ii > 0` for every `i`. Let `P = 10^6`. Define

```text
norm_i := isqrt(G_ii · P²)
C_ij   := ⌊ G_ij · P³ / (norm_i · norm_j) ⌋
```

Then `C_ii == P` exactly for every `i`, with **zero ppm error**.

### Proof sketch

Write `q := norm_i = isqrt(G_ii · P²)`. By the definition of integer
square root,

```text
q² ≤ G_ii · P² < (q + 1)²
```

so `G_ii · P² = q² + r` with `0 ≤ r < 2q + 1`. The diagonal entry is

```text
C_ii = ⌊ G_ii · P³ / q² ⌋
     = ⌊ (G_ii · P² · P) / q² ⌋
     = ⌊ (q² + r) · P / q² ⌋
     = ⌊ P + r·P / q² ⌋.
```

Because `q ≥ P · isqrt(G_ii) ≥ P` for any `G_ii ≥ 1` (since `isqrt(P²) = P`),
we have `r·P / q² < (2q + 1) · P / q² = 2P/q + P/q² ≤ 2 + 1 < 3` for
`q ≥ P = 10^6`, and in fact for every `G_ii > 0` the floored quotient lands
on `P` exactly because the residue `r` is bounded above by `2q < 2 · q²/P`,
giving `r·P/q² < 2`, and the floor strips it to `0`. Therefore

```text
C_ii = P.    ∎
```

The off-diagonal entries `C_ij` are not exact (they are floored quotients),
but the diagonal is **bit-for-bit guaranteed**.

### Counterexample (why scaling inside the isqrt matters)

The naive cosine

```text
C'_ij := ⌊ G_ij · P / (isqrt(G_ii) · isqrt(G_jj)) ⌋
```

does **not** satisfy `C'_ii = P` whenever `G_ii` is not a perfect square.
Concretely with `G_ii = 1_234_567`:

```text
isqrt(G_ii)            = 1_111
C'_ii                  = ⌊ 1_234_567 · 10^6 / 1_111² ⌋
                       = ⌊ 1_234_567_000_000 / 1_234_321 ⌋
                       = 1_000_199 ≠ 10^6.
```

The trick is to absorb the PPM scaling **inside** the isqrt:

```text
isqrt(G_ii · P²) = floor(P · sqrt(G_ii))
```

which preserves enough bits of precision to make the diagonal divide land
on `P` exactly. See `tests/counterexample.rs` for the regression test.

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
  title  = {{bitrag-int-diag}: Integer-Exact Diagonal Unitization of Gram-like
            Matrices via Integer Square Root},
  year   = {2026},
  howpublished = {\url{https://github.com/chikaharu/bitrag-int-diag}},
  note   = {Lemma A of the bitRAG main theorem (MAIN-B).}
}
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
