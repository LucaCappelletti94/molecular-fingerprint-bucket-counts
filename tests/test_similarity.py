from __future__ import annotations

import numpy as np
import pytest

from fp_bucket_counts.similarity import (
    covariance_from_cooccurrence,
    diagonal_mahalanobis,
    diagonal_precision_weights,
    entropy_hamming,
    entropy_weights,
    idf_tanimoto,
    idf_tanimoto_batch,
    idf_weights,
    load_similarity_weights_npz,
    mahalanobis,
    mahalanobis_batch,
    precision_matrix,
    save_similarity_weights_npz,
    tanimoto,
)


# ------------------------------------------------------------------
# Shared fixture: 4-molecule, 3-bit example (same as test_cooccurrence)
# ------------------------------------------------------------------
@pytest.fixture
def small_corpus():
    fps = np.array(
        [[1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1]],
        dtype=np.uint8,
    )
    cooc = (fps.T @ fps).astype(np.uint64)  # [[3,2,1],[2,2,0],[1,0,2]]
    counts = fps.sum(axis=0).astype(np.uint64)  # [3, 2, 2]
    n = 4
    return fps, cooc, counts, n


# ------------------------------------------------------------------
# Covariance
# ------------------------------------------------------------------
class TestCovariance:
    def test_known_values(self, small_corpus):
        _, cooc, _, n = small_corpus
        cov = covariance_from_cooccurrence(cooc, n)
        # p = [3/4, 1/2, 1/2]
        # Sigma_00 = p0*(1-p0) = 3/16
        np.testing.assert_almost_equal(cov[0, 0], 3 / 16)
        # Sigma_11 = 1/4
        np.testing.assert_almost_equal(cov[1, 1], 1 / 4)
        # Sigma_01 = cooc[0,1]/4 - p0*p1 = 2/4 - 3/4*1/2 = 1/2 - 3/8 = 1/8
        np.testing.assert_almost_equal(cov[0, 1], 1 / 8)
        # Sigma_12 = 0/4 - 1/2*1/2 = -1/4
        np.testing.assert_almost_equal(cov[1, 2], -1 / 4)

    def test_symmetric(self, small_corpus):
        _, cooc, _, n = small_corpus
        cov = covariance_from_cooccurrence(cooc, n)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_zero_molecules(self):
        cooc = np.zeros((3, 3), dtype=np.uint64)
        cov = covariance_from_cooccurrence(cooc, 0)
        np.testing.assert_array_equal(cov, 0.0)


# ------------------------------------------------------------------
# Precision matrix
# ------------------------------------------------------------------
class TestPrecisionMatrix:
    def test_roundtrip_identity(self, small_corpus):
        _, cooc, _, n = small_corpus
        omega = precision_matrix(cooc, n, shrinkage=0.1)
        cov = covariance_from_cooccurrence(cooc, n)
        from sklearn.covariance import shrunk_covariance

        cov_s = shrunk_covariance(cov, shrinkage=0.1)
        product = omega @ cov_s
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=10)

    def test_positive_definite(self, small_corpus):
        _, cooc, _, n = small_corpus
        omega = precision_matrix(cooc, n)
        eigvals = np.linalg.eigvalsh(omega)
        assert np.all(eigvals > 0)

    def test_symmetric(self, small_corpus):
        _, cooc, _, n = small_corpus
        omega = precision_matrix(cooc, n)
        np.testing.assert_array_almost_equal(omega, omega.T)

    def test_singular_covariance_handled(self):
        # All-identical fingerprints → rank-deficient covariance
        cooc = np.array([[4, 4], [4, 4]], dtype=np.uint64)
        omega = precision_matrix(cooc, 4, shrinkage=0.1)
        assert omega.shape == (2, 2)
        assert np.all(np.isfinite(omega))

    def test_zero_molecules(self):
        cooc = np.zeros((3, 3), dtype=np.uint64)
        omega = precision_matrix(cooc, 0)
        np.testing.assert_array_equal(omega, 0.0)


# ------------------------------------------------------------------
# IDF weights
# ------------------------------------------------------------------
class TestIdfWeights:
    def test_known_values(self, small_corpus):
        _, _, counts, n = small_corpus
        w = idf_weights(counts, n, smooth=True)
        # w[0] = log(4 / (1+3)) = log(1) = 0
        np.testing.assert_almost_equal(w[0], 0.0)
        # w[1] = log(4 / (1+2)) = log(4/3)
        np.testing.assert_almost_equal(w[1], np.log(4 / 3))

    def test_no_smooth(self, small_corpus):
        _, _, counts, n = small_corpus
        w = idf_weights(counts, n, smooth=False)
        # w[0] = log(4/3)
        np.testing.assert_almost_equal(w[0], np.log(4 / 3))

    def test_uniform_equal(self):
        counts = np.array([50, 50, 50], dtype=np.uint64)
        w = idf_weights(counts, 100, smooth=True)
        np.testing.assert_array_almost_equal(w, w[0])

    def test_zero_molecules(self):
        w = idf_weights(np.array([1, 2], dtype=np.uint64), 0)
        np.testing.assert_array_equal(w, 0.0)


# ------------------------------------------------------------------
# Entropy weights
# ------------------------------------------------------------------
class TestEntropyWeights:
    def test_half_gives_one(self):
        counts = np.array([50], dtype=np.uint64)
        h = entropy_weights(counts, 100)
        np.testing.assert_almost_equal(h[0], 1.0)

    def test_zero_or_all_gives_zero(self):
        counts = np.array([0, 100], dtype=np.uint64)
        h = entropy_weights(counts, 100)
        np.testing.assert_array_equal(h, 0.0)

    def test_zero_molecules(self):
        h = entropy_weights(np.array([1], dtype=np.uint64), 0)
        np.testing.assert_array_equal(h, 0.0)


# ------------------------------------------------------------------
# Diagonal precision weights
# ------------------------------------------------------------------
class TestDiagonalPrecisionWeights:
    def test_half_gives_four(self):
        counts = np.array([50], dtype=np.uint64)
        w = diagonal_precision_weights(counts, 100)
        np.testing.assert_almost_equal(w[0], 4.0)

    def test_constant_bit_clamped(self):
        counts = np.array([0, 100], dtype=np.uint64)
        w = diagonal_precision_weights(counts, 100)
        assert np.all(w > 0)
        np.testing.assert_almost_equal(w[0], 1 / 1e-12)

    def test_zero_molecules(self):
        w = diagonal_precision_weights(np.array([5], dtype=np.uint64), 0)
        np.testing.assert_array_equal(w, 0.0)


# ------------------------------------------------------------------
# Tanimoto
# ------------------------------------------------------------------
class TestTanimoto:
    def test_identical(self):
        x = np.array([1, 1, 0], dtype=np.uint8)
        assert tanimoto(x, x) == 1.0

    def test_disjoint(self):
        x = np.array([1, 1, 0], dtype=np.uint8)
        y = np.array([0, 0, 1], dtype=np.uint8)
        assert tanimoto(x, y) == 0.0

    def test_known_value(self):
        x = np.array([1, 1, 0], dtype=np.uint8)
        y = np.array([1, 0, 1], dtype=np.uint8)
        # dot=1, sum_x=2, sum_y=2 → 1/(2+2-1) = 1/3
        np.testing.assert_almost_equal(tanimoto(x, y), 1 / 3)

    def test_both_zero(self):
        x = np.zeros(3, dtype=np.uint8)
        assert tanimoto(x, x) == 0.0


# ------------------------------------------------------------------
# IDF Tanimoto
# ------------------------------------------------------------------
class TestIdfTanimoto:
    def test_uniform_matches_standard(self):
        x = np.array([1, 1, 0], dtype=np.uint8)
        y = np.array([1, 0, 1], dtype=np.uint8)
        w = np.ones(3, dtype=np.float64)
        np.testing.assert_almost_equal(idf_tanimoto(x, y, w), tanimoto(x, y))

    def test_both_zero(self):
        x = np.zeros(3, dtype=np.uint8)
        w = np.ones(3, dtype=np.float64)
        assert idf_tanimoto(x, x, w) == 0.0

    def test_known_weighted_value(self):
        x = np.array([1, 1, 0], dtype=np.uint8)
        y = np.array([1, 0, 1], dtype=np.uint8)
        w = np.array([2.0, 3.0, 5.0])
        # intersection bits: bit 0 only → num = w[0]*1*1 = 2
        # denom = (w[0]+w[1]) + (w[0]+w[2]) - 2 = 5 + 7 - 2 = 10
        np.testing.assert_almost_equal(idf_tanimoto(x, y, w), 2.0 / 10.0)


# ------------------------------------------------------------------
# Entropy Hamming
# ------------------------------------------------------------------
class TestEntropyHamming:
    def test_identical_zero(self):
        x = np.array([1, 1, 0], dtype=np.uint8)
        w = np.ones(3, dtype=np.float64)
        assert entropy_hamming(x, x, w) == 0.0

    def test_zero_weights(self):
        x = np.array([1, 0], dtype=np.uint8)
        y = np.array([0, 1], dtype=np.uint8)
        w = np.zeros(2, dtype=np.float64)
        assert entropy_hamming(x, y, w) == 0.0

    def test_known_value(self):
        x = np.array([1, 1, 0], dtype=np.uint8)
        y = np.array([1, 0, 1], dtype=np.uint8)
        w = np.array([2.0, 3.0, 5.0])
        # delta = [0, 1, 1], weighted = [0, 3, 5] → sum = 8
        # total_weight = 10 → 8/10 = 0.8
        np.testing.assert_almost_equal(entropy_hamming(x, y, w), 0.8)


# ------------------------------------------------------------------
# Diagonal Mahalanobis
# ------------------------------------------------------------------
class TestDiagonalMahalanobis:
    def test_identical_zero(self):
        x = np.array([1, 0, 1], dtype=np.uint8)
        w = np.ones(3, dtype=np.float64)
        assert diagonal_mahalanobis(x, x, w) == 0.0

    def test_unit_weights_equals_hamming(self):
        x = np.array([1, 1, 0], dtype=np.uint8)
        y = np.array([1, 0, 1], dtype=np.uint8)
        w = np.ones(3, dtype=np.float64)
        # Hamming distance = 2 differing bits
        np.testing.assert_almost_equal(diagonal_mahalanobis(x, y, w), 2.0)

    def test_known_weighted_value(self):
        x = np.array([1, 1, 0], dtype=np.uint8)
        y = np.array([1, 0, 1], dtype=np.uint8)
        w = np.array([2.0, 3.0, 5.0])
        # delta = [0, 1, -1], delta^2 = [0, 1, 1]
        # weighted = [0, 3, 5] → sum = 8
        np.testing.assert_almost_equal(diagonal_mahalanobis(x, y, w), 8.0)


# ------------------------------------------------------------------
# Full Mahalanobis
# ------------------------------------------------------------------
class TestMahalanobis:
    def test_identity_equals_hamming(self):
        x = np.array([1, 1, 0], dtype=np.uint8)
        y = np.array([1, 0, 1], dtype=np.uint8)
        omega = np.eye(3)
        np.testing.assert_almost_equal(mahalanobis(x, y, omega), 2.0)

    def test_identical_zero(self):
        x = np.array([1, 0, 1], dtype=np.uint8)
        omega = np.eye(3) * 5
        assert mahalanobis(x, x, omega) == 0.0

    def test_non_negative(self, small_corpus):
        fps, cooc, _, n = small_corpus
        omega = precision_matrix(cooc, n)
        for i in range(len(fps)):
            for j in range(len(fps)):
                assert mahalanobis(fps[i], fps[j], omega) >= 0


# ------------------------------------------------------------------
# Batch variants
# ------------------------------------------------------------------
class TestBatch:
    def test_idf_tanimoto_batch_matches_loop(self):
        rng = np.random.default_rng(42)
        query = rng.integers(0, 2, size=8, dtype=np.uint8)
        targets = rng.integers(0, 2, size=(5, 8), dtype=np.uint8)
        weights = rng.random(8)
        batch_result = idf_tanimoto_batch(query, targets, weights)
        loop_result = np.array([idf_tanimoto(query, t, weights) for t in targets])
        np.testing.assert_array_almost_equal(batch_result, loop_result)

    def test_mahalanobis_batch_matches_loop(self):
        rng = np.random.default_rng(42)
        query = rng.integers(0, 2, size=8, dtype=np.uint8)
        targets = rng.integers(0, 2, size=(5, 8), dtype=np.uint8)
        omega = np.eye(8)
        batch_result = mahalanobis_batch(query, targets, omega)
        loop_result = np.array([mahalanobis(query, t, omega) for t in targets])
        np.testing.assert_array_almost_equal(batch_result, loop_result)

    def test_mahalanobis_batch_nontrivial_precision(self, small_corpus):
        fps, cooc, _, n = small_corpus
        omega = precision_matrix(cooc, n)
        query = fps[0]
        targets = fps[1:]
        batch_result = mahalanobis_batch(query, targets, omega)
        loop_result = np.array([mahalanobis(query, t, omega) for t in targets])
        np.testing.assert_array_almost_equal(batch_result, loop_result)
        assert np.all(batch_result >= 0)


# ------------------------------------------------------------------
# I/O roundtrip
# ------------------------------------------------------------------
class TestIO:
    def test_roundtrip_with_precision(self, tmp_path, small_corpus):
        _, cooc, counts, n = small_corpus
        idf = idf_weights(counts, n)
        ent = entropy_weights(counts, n)
        dp = diagonal_precision_weights(counts, n)
        prec = precision_matrix(cooc, n)

        path = tmp_path / "sim_weights.npz"
        save_similarity_weights_npz(
            path,
            idf=idf,
            entropy=ent,
            diagonal_precision=dp,
            precision=prec,
            shrinkage=0.1,
            total_molecules=n,
        )
        loaded = load_similarity_weights_npz(path)
        np.testing.assert_array_almost_equal(loaded["idf"], idf)
        np.testing.assert_array_almost_equal(loaded["entropy"], ent)
        np.testing.assert_array_almost_equal(loaded["diagonal_precision"], dp)
        np.testing.assert_array_almost_equal(loaded["precision"], prec)
        assert loaded["shrinkage"] == 0.1
        assert loaded["total_molecules"] == n

    def test_roundtrip_without_precision(self, tmp_path, small_corpus):
        _, _, counts, n = small_corpus
        idf = idf_weights(counts, n)
        ent = entropy_weights(counts, n)
        dp = diagonal_precision_weights(counts, n)

        path = tmp_path / "sim_no_prec.npz"
        save_similarity_weights_npz(
            path,
            idf=idf,
            entropy=ent,
            diagonal_precision=dp,
            precision=None,
            shrinkage=0.1,
            total_molecules=n,
        )
        loaded = load_similarity_weights_npz(path)
        assert loaded["precision"] is None
        np.testing.assert_array_almost_equal(loaded["idf"], idf)
