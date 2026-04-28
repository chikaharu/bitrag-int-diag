//! Frozen E145 Gram matrix replay (regression fixture).
//!
//! Source: chikaharu/bitRAG `artifacts/bitrag/experiment-145` ran against the
//! 六法 (kenpo, minpo, shoho, keiho, minso, keiso) Japanese law corpus,
//! FNV-1a 16-bit n4-gram hashes, integer IDF² weight `w_h = (6 - df_h)²`.
//!
//! The G matrix below is the published E145 output, bit-for-bit. Any drift
//! in `diagonal_unitize` will break this test — that is the whole point.

use bitrag_int_diag::{diagonal_unitize, PPM};

/// E145 published Gram matrix `G = X^T diag(idf²) X`, rows in order:
/// kenpo, minpo, shoho, keiho, minso, keiso.
const G: [[u64; 6]; 6] = [
    [54_969, 22_395, 8_805, 5_775, 10_665, 15_218],
    [22_395, 469_364, 105_682, 56_249, 111_037, 134_509],
    [8_805, 105_682, 210_692, 25_280, 49_810, 59_359],
    [5_775, 56_249, 25_280, 129_108, 30_257, 46_623],
    [10_665, 111_037, 49_810, 30_257, 234_697, 85_487],
    [15_218, 134_509, 59_359, 46_623, 85_487, 303_102],
];

/// E145 published `‖x_i‖ × PPM = isqrt(G_ii · PPM²)`.
const NORM_PPM_EXPECTED: [u128; 6] = [
    234_454_686,
    685_101_452,
    459_011_982,
    359_316_016,
    484_455_364,
    550_547_000,
];

/// E145 cosine matrix `C[i][j]` for every unique pair `i < j`, in PPM units
/// (full 6 digits). Computed bit-for-bit from the published G + norms above.
const C_OFF_DIAG_EXPECTED: &[(usize, usize, u64)] = &[
    (0, 1, 139_423),
    (0, 2, 81_817),
    (0, 3, 68_551),
    (0, 4, 93_896),
    (0, 5, 117_897),
    (1, 2, 336_064),
    (1, 3, 228_498),
    (1, 4, 334_548),
    (1, 5, 356_617),
    (2, 3, 153_276),
    (2, 4, 223_995),
    (2, 5, 234_891),
    (3, 4, 173_818),
    (3, 5, 235_683),
    (4, 5, 320_517),
];

#[test]
fn e145_diagonal_zero_ppm() {
    let g_vec: Vec<Vec<u64>> = G.iter().map(|r| r.to_vec()).collect();
    let (norms, c) = diagonal_unitize(&g_vec);

    // Diagonal must be exactly PPM (= 1.000000), with 0 ppm error.
    for (i, row) in c.iter().enumerate() {
        let v = row[i];
        let err: i64 = v as i64 - PPM as i64;
        assert_eq!(
            err, 0,
            "C[{i}][{i}] expected exactly {PPM} (= 1.000000), got {v} (err {err} ppm)"
        );
    }

    // Norms must match the published E145 numbers bit-for-bit.
    for (i, (&got, &expected)) in norms.iter().zip(NORM_PPM_EXPECTED.iter()).enumerate() {
        assert_eq!(got, expected, "norm[{i}] expected {expected}, got {got}");
    }
}

#[test]
fn e145_off_diagonal_match() {
    let g_vec: Vec<Vec<u64>> = G.iter().map(|r| r.to_vec()).collect();
    let (_norms, c) = diagonal_unitize(&g_vec);

    for &(i, j, expected) in C_OFF_DIAG_EXPECTED {
        let got_ij = c[i][j];
        let got_ji = c[j][i];
        assert_eq!(
            got_ij, expected,
            "C[{i}][{j}] expected {expected} ppm, got {got_ij} ppm"
        );
        assert_eq!(
            got_ji, expected,
            "C[{j}][{i}] (mirror) expected {expected} ppm, got {got_ji} ppm"
        );
    }
}

#[test]
fn e145_symmetry_preserved() {
    let g_vec: Vec<Vec<u64>> = G.iter().map(|r| r.to_vec()).collect();
    let (_, c) = diagonal_unitize(&g_vec);
    let n = c.len();
    for (i, row_i) in c.iter().enumerate() {
        for (j, &v_ij) in row_i.iter().enumerate().take(n) {
            let v_ji = c[j][i];
            assert_eq!(v_ij, v_ji, "C is not symmetric at ({i},{j})");
        }
    }
}
