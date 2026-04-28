//! Counterexample: the **naive** integer cosine that `isqrt`s `G_ii` outside
//! the PPM scaling **does not** preserve diagonal exactness.
//!
//! This is the negative half of the diagonal-exact identity: it shows
//! why scaling the PPM factor *inside* the `isqrt` (as
//! [`bitrag_int_diag::norm_ppm`] does) is essential, and not a notational
//! convenience.

use bitrag_int_diag::{diagonal_unitize, naive_diagonal_unitize, PPM};

#[test]
fn naive_breaks_diagonal_on_non_square_g_ii() {
    // G_ii values that are not perfect squares — the typical case for any
    // real corpus.
    let g: Vec<Vec<u64>> = vec![
        vec![1_234_567, 100, 50],
        vec![100, 9_876_543, 200],
        vec![50, 200, 1_111_111],
    ];

    let c_naive = naive_diagonal_unitize(&g);

    // Naive method: at least one diagonal entry must miss PPM.
    let diag_errors: Vec<i64> = c_naive
        .iter()
        .enumerate()
        .map(|(i, row)| row[i] as i64 - PPM as i64)
        .collect();
    assert!(
        diag_errors.iter().any(|&e| e != 0),
        "naive method should lose diagonal exactness on non-square G_ii (got errors {diag_errors:?})"
    );

    // Inside-isqrt method: diagonal must be exactly PPM for the same input.
    let (_, c_exact) = diagonal_unitize(&g);
    for (i, row) in c_exact.iter().enumerate() {
        let v = row[i];
        assert_eq!(
            v, PPM,
            "inside-isqrt method must give exact PPM at C[{i}][{i}], got {v}"
        );
    }
}

#[test]
fn naive_breaks_on_e145_law_corpus() {
    // Same G as the E145 replay — every G_ii is non-square.
    let g: Vec<Vec<u64>> = vec![
        vec![54_969, 22_395, 8_805, 5_775, 10_665, 15_218],
        vec![22_395, 469_364, 105_682, 56_249, 111_037, 134_509],
        vec![8_805, 105_682, 210_692, 25_280, 49_810, 59_359],
        vec![5_775, 56_249, 25_280, 129_108, 30_257, 46_623],
        vec![10_665, 111_037, 49_810, 30_257, 234_697, 85_487],
        vec![15_218, 134_509, 59_359, 46_623, 85_487, 303_102],
    ];

    let c_naive = naive_diagonal_unitize(&g);

    // At least one — in fact, all six — diagonal entries miss PPM.
    let misses = c_naive
        .iter()
        .enumerate()
        .filter(|(i, row)| row[*i] != PPM)
        .count();
    assert!(
        misses >= 1,
        "naive method should break diagonal on the E145 corpus, but got 0 misses"
    );
}
