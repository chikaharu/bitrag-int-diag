//! # bitrag-int-diag
//!
//! **Integer-exact diagonal unitization of Gram-like matrices**, achieving
//! **0 ppm error on the diagonal** using only `u128` integer arithmetic and
//! Newton–Raphson integer square root.
//!
//! ## Theorem A (Integer IDF² Diagonal Unitization)
//!
//! Let `G ∈ ℕ^{n×n}` be a Gram-like integer matrix with `G_ii > 0`
//! for every `i`. Let `P = 10^6` ([`PPM`]). Define
//!
//! ```text
//! norm_i := isqrt(G_ii · P²)
//! C_ij   := ⌊ G_ij · P³ / (norm_i · norm_j) ⌋
//! ```
//!
//! Then `C_ii == P` exactly for every `i`, with **zero ppm error**.
//!
//! ## Origin
//!
//! Extracted and formalized from
//! [chikaharu/bitRAG](https://github.com/chikaharu/bitRAG) experiment E145
//! ("XCORR重みIDF² 厳密対角単位化, 整数 cosine, 誤差 0 ppm").
//!
//! ## Counterexample
//!
//! A naive integer divison
//!
//! ```text
//! C'_ij = G_ij · P / (isqrt(G_ii) · isqrt(G_jj))
//! ```
//!
//! does **not** preserve diagonal exactness when `G_ii` is not a perfect
//! square. The key trick of Theorem A is to absorb the PPM scaling
//! **inside** the `isqrt` so that `norm_i² ≤ G_ii · P²`, which makes
//! `G_ii · P³ / norm_i² ≥ P` and the floored result lands on `P` exactly
//! whenever `norm_i² ≥ G_ii · P² · (P-1)/P`. See [`naive_diagonal_unitize`]
//! for the broken version, used in tests.
//!
//! ## Public API
//!
//! - [`PPM`] — the fixed-point unit (10^6).
//! - [`isqrt_u128`] — Newton–Raphson floored integer square root.
//! - [`norm_ppm`] — PPM-scaled self-norm of one row.
//! - [`cosine_ppm`] — pairwise PPM-scaled cosine.
//! - [`diagonal_unitize`] — main entry point.
//! - [`idf_squared_weights`] — integer IDF² weight constructor.
//! - [`naive_diagonal_unitize`] — counterexample (broken on non-square G_ii).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Fixed-point unit for the cosine matrix: `1.000000` is represented as
/// `1_000_000`.
pub const PPM: u64 = 1_000_000;

/// Newton–Raphson floored integer square root for `u128`.
///
/// Returns `floor(sqrt(n))`. Runs in `O(log log n)` iterations on average.
///
/// # Examples
///
/// ```
/// use bitrag_int_diag::isqrt_u128;
/// assert_eq!(isqrt_u128(0), 0);
/// assert_eq!(isqrt_u128(99), 9);
/// assert_eq!(isqrt_u128(100), 10);
/// assert_eq!(isqrt_u128(1_000_000), 1_000);
/// ```
#[inline]
pub fn isqrt_u128(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) / 2;
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// `norm_i = isqrt(G_ii · P²)` — the PPM-scaled self-norm of row `i`.
///
/// This is **the** trick: scaling **inside** the `isqrt` is what makes the
/// diagonal of the resulting cosine matrix land on `PPM` exactly.
#[inline]
pub fn norm_ppm(g_ii: u64) -> u128 {
    let scaled = (g_ii as u128) * (PPM as u128) * (PPM as u128);
    isqrt_u128(scaled)
}

/// `C_ij = floor(G_ij · P³ / (norm_i · norm_j))`, returning `0` when either
/// norm is zero.
///
/// **Diagonal exactness**: when `j == i` and `norm_i = isqrt(G_ii · P²)`,
/// this returns [`PPM`] (= `1.000000`) for every `G_ii > 0`.
#[inline]
pub fn cosine_ppm(g_ij: u64, norm_i: u128, norm_j: u128) -> u64 {
    if norm_i == 0 || norm_j == 0 {
        return 0;
    }
    let num = (g_ij as u128) * (PPM as u128) * (PPM as u128) * (PPM as u128);
    let denom = norm_i * norm_j;
    (num / denom) as u64
}

/// Diagonally unitize a square integer Gram matrix.
///
/// Given `G ∈ ℕ^{n×n}`, returns `(norms, C)` where
///
/// - `norms[i] = isqrt(G[i][i] · P²)`
/// - `C[i][j] = floor(G[i][j] · P³ / (norms[i] · norms[j]))`
///
/// **Guarantee** (Theorem A): `C[i][i] == PPM` exactly for every `i` with
/// `G[i][i] > 0`, with zero ppm error.
///
/// # Panics
///
/// Panics if `G` is not square.
pub fn diagonal_unitize(g: &[Vec<u64>]) -> (Vec<u128>, Vec<Vec<u64>>) {
    let n = g.len();
    assert!(
        g.iter().all(|row| row.len() == n),
        "G must be square (got rows of varying length)"
    );
    let norms: Vec<u128> = (0..n).map(|i| norm_ppm(g[i][i])).collect();
    let mut c = vec![vec![0u64; n]; n];
    for i in 0..n {
        for j in 0..n {
            c[i][j] = cosine_ppm(g[i][j], norms[i], norms[j]);
        }
    }
    (norms, c)
}

/// Integer IDF² weight vector. For each token with document-frequency
/// `df_h`, the weight is `(num_docs - df_h)²`, so the most discriminating
/// tokens get the largest weight and corpus-universal tokens get weight 0.
///
/// # Examples
///
/// ```
/// use bitrag_int_diag::idf_squared_weights;
/// assert_eq!(
///     idf_squared_weights(6, &[1, 6, 0, 3]),
///     vec![25, 0, 36, 9],
/// );
/// ```
pub fn idf_squared_weights(num_docs: u32, df: &[u32]) -> Vec<u64> {
    df.iter()
        .map(|&d| {
            let w = num_docs.saturating_sub(d) as u64;
            w * w
        })
        .collect()
}

/// **Counterexample** to diagonal exactness: a naive integer cosine that
/// `isqrt`s `G_ii` *outside* the PPM scaling, then divides.
///
/// `C'_ij = floor(G_ij · P / (isqrt(G_ii) · isqrt(G_jj)))`
///
/// This produces the **wrong** diagonal whenever `G_ii` is not a perfect
/// square. Used in [`tests/counterexample.rs`] to demonstrate why the
/// in-isqrt scaling of [`diagonal_unitize`] is essential.
pub fn naive_diagonal_unitize(g: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let n = g.len();
    let norm: Vec<u128> = (0..n).map(|i| isqrt_u128(g[i][i] as u128)).collect();
    let mut c = vec![vec![0u64; n]; n];
    for i in 0..n {
        for j in 0..n {
            c[i][j] = if norm[i] == 0 || norm[j] == 0 {
                0
            } else {
                let num = (g[i][j] as u128) * (PPM as u128);
                (num / (norm[i] * norm[j])) as u64
            };
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isqrt_basics() {
        assert_eq!(isqrt_u128(0), 0);
        assert_eq!(isqrt_u128(1), 1);
        assert_eq!(isqrt_u128(2), 1);
        assert_eq!(isqrt_u128(3), 1);
        assert_eq!(isqrt_u128(4), 2);
        assert_eq!(isqrt_u128(99), 9);
        assert_eq!(isqrt_u128(100), 10);
        assert_eq!(isqrt_u128(1_000_000), 1_000);
    }

    #[test]
    fn isqrt_large() {
        let n: u128 = 12_345_678_901_234_567_890u128;
        let r = isqrt_u128(n);
        assert!(r * r <= n);
        let next = r.checked_add(1).expect("no overflow");
        assert!(next.checked_mul(next).map_or(true, |s| s > n));
    }

    #[test]
    fn diagonal_exact_synthetic() {
        let g: Vec<Vec<u64>> = vec![
            vec![100, 30, 20, 10],
            vec![30, 200, 40, 5],
            vec![20, 40, 150, 25],
            vec![10, 5, 25, 80],
        ];
        let (_, c) = diagonal_unitize(&g);
        for (i, row) in c.iter().enumerate() {
            let v = row[i];
            assert_eq!(v, PPM, "diagonal must be exactly PPM at row {i}, got {v}");
        }
    }

    #[test]
    fn empty_matrix_ok() {
        let g: Vec<Vec<u64>> = vec![];
        let (norms, c) = diagonal_unitize(&g);
        assert!(norms.is_empty());
        assert!(c.is_empty());
    }

    #[test]
    fn zero_diagonal_row_yields_zero_row() {
        let g: Vec<Vec<u64>> = vec![vec![0, 5], vec![5, 100]];
        let (norms, c) = diagonal_unitize(&g);
        assert_eq!(norms[0], 0);
        assert_eq!(c[0][0], 0);
        assert_eq!(c[0][1], 0);
        assert_eq!(c[1][0], 0);
        assert_eq!(c[1][1], PPM);
    }

    #[test]
    #[should_panic(expected = "G must be square")]
    fn non_square_panics() {
        let g: Vec<Vec<u64>> = vec![vec![1, 2, 3], vec![1, 2]];
        let _ = diagonal_unitize(&g);
    }

    #[test]
    fn idf_squared_basic() {
        let w = idf_squared_weights(6, &[1, 6, 0, 3]);
        assert_eq!(w, vec![25, 0, 36, 9]);
    }
}
