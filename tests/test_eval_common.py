from __future__ import annotations

import numpy as np
import pandas as pd

from fp_bucket_counts.eval_common import (
    _diagonal_mahalanobis_batch,
    _entropy_hamming_batch,
    _tanimoto_batch,
    aggregate_and_save,
    compute_auc,
    compute_ef,
    compute_roc,
    evaluate_target,
    fingerprint_smiles,
)
from fp_bucket_counts.similarity import (
    diagonal_mahalanobis,
    entropy_hamming,
    tanimoto,
)


# ------------------------------------------------------------------
# AUC
# ------------------------------------------------------------------
class TestComputeAuc:
    def test_perfect_ranking(self):
        labels = np.array([1, 1, 0, 0, 0], dtype=np.int8)
        scores = np.array([1.0, 0.9, 0.3, 0.2, 0.1])
        assert compute_auc(labels, scores) == 1.0

    def test_random_ranking(self):
        rng = np.random.default_rng(123)
        labels = np.zeros(10000, dtype=np.int8)
        labels[:100] = 1
        scores = rng.random(10000)
        auc = compute_auc(labels, scores)
        assert 0.4 < auc < 0.6

    def test_single_class_returns_nan(self):
        labels = np.array([0, 0, 0], dtype=np.int8)
        scores = np.array([0.5, 0.3, 0.1])
        assert np.isnan(compute_auc(labels, scores))


# ------------------------------------------------------------------
# EF1%
# ------------------------------------------------------------------
class TestComputeEf:
    def test_perfect_ranking(self):
        n = 1000
        labels = np.zeros(n, dtype=np.int8)
        labels[:5] = 1  # 5 actives
        scores = np.zeros(n)
        scores[:5] = 10.0  # actives ranked first
        ef = compute_ef(labels, scores, fraction=0.01)
        # top 1% = 10 molecules, all 5 actives present
        # expected = 5 * 0.01 = 0.05 -> EF = 5 / 0.05 = 100
        assert ef == 100.0

    def test_random_ranking(self):
        rng = np.random.default_rng(456)
        n = 100000
        labels = np.zeros(n, dtype=np.int8)
        labels[:1000] = 1
        scores = rng.random(n)
        ef = compute_ef(labels, scores, fraction=0.01)
        # Random should give EF ~1.0
        assert 0.5 < ef < 1.5

    def test_no_actives(self):
        labels = np.zeros(100, dtype=np.int8)
        scores = np.random.default_rng(0).random(100)
        assert compute_ef(labels, scores) == 0.0


# ------------------------------------------------------------------
# ROC curve
# ------------------------------------------------------------------
class TestComputeRoc:
    def test_perfect_ranking(self):
        labels = np.array([1, 1, 0, 0, 0], dtype=np.int8)
        scores = np.array([1.0, 0.9, 0.3, 0.2, 0.1])
        result = compute_roc(labels, scores)
        assert result is not None
        fpr, tpr = result
        assert fpr[0] == 0.0
        assert tpr[-1] == 1.0
        assert len(fpr) == len(tpr)

    def test_single_class_returns_none(self):
        labels = np.array([0, 0, 0], dtype=np.int8)
        scores = np.array([0.5, 0.3, 0.1])
        assert compute_roc(labels, scores) is None


# ------------------------------------------------------------------
# evaluate_target smoke test
# ------------------------------------------------------------------
class TestEvaluateTarget:
    def test_smoke(self):
        rng = np.random.default_rng(42)
        n_mols = 50
        fp_size = 16
        fps = rng.integers(0, 2, size=(n_mols, fp_size), dtype=np.uint8)

        actives_idx = np.arange(5, dtype=np.intp)
        decoys_idx = np.arange(5, n_mols, dtype=np.intp)

        weights = {
            "idf": np.ones(fp_size, dtype=np.float64),
            "entropy": np.ones(fp_size, dtype=np.float64),
            "diagonal_precision": np.ones(fp_size, dtype=np.float64),
            "precision": np.eye(fp_size, dtype=np.float64),
        }

        result, roc_curves = evaluate_target(fps, actives_idx, decoys_idx, weights, 2, rng)

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"query", "metric", "auc", "ef1"}
        # 2 queries x 5 metrics = 10 rows
        assert len(result) == 10
        expected_metrics = {
            "tanimoto",
            "idf_tanimoto",
            "entropy_hamming",
            "diagonal_mahalanobis",
            "full_mahalanobis",
        }
        assert set(result["metric"]) == expected_metrics

        # ROC curves: 2 queries x 5 metrics = 10 entries
        assert len(roc_curves) == 10
        for entry in roc_curves:
            assert set(entry.keys()) == {"query", "metric", "fpr", "tpr"}
            assert len(entry["fpr"]) == len(entry["tpr"])

    def test_without_precision(self):
        rng = np.random.default_rng(99)
        fps = rng.integers(0, 2, size=(20, 8), dtype=np.uint8)
        actives_idx = np.arange(3, dtype=np.intp)
        decoys_idx = np.arange(3, 20, dtype=np.intp)

        weights = {
            "idf": np.ones(8, dtype=np.float64),
            "entropy": np.ones(8, dtype=np.float64),
            "diagonal_precision": np.ones(8, dtype=np.float64),
            "precision": None,
        }

        result, roc_curves = evaluate_target(fps, actives_idx, decoys_idx, weights, 2, rng)
        # 2 queries x 4 metrics (no full mahalanobis) = 8 rows
        assert len(result) == 8
        assert "full_mahalanobis" not in result["metric"].values
        assert len(roc_curves) == 8


# ------------------------------------------------------------------
# Cross-validate batch functions against similarity scalars
# ------------------------------------------------------------------
class TestBatchVsScalar:
    def test_tanimoto_batch_matches_scalar(self):
        rng = np.random.default_rng(77)
        query = rng.integers(0, 2, size=16, dtype=np.uint8)
        targets = rng.integers(0, 2, size=(10, 16), dtype=np.uint8)
        batch = _tanimoto_batch(query, targets)
        loop = np.array([tanimoto(query, t) for t in targets])
        np.testing.assert_array_almost_equal(batch, loop)

    def test_entropy_hamming_batch_matches_scalar(self):
        rng = np.random.default_rng(78)
        query = rng.integers(0, 2, size=16, dtype=np.uint8)
        targets = rng.integers(0, 2, size=(10, 16), dtype=np.uint8)
        weights = rng.random(16)
        batch = _entropy_hamming_batch(query, targets, weights)
        loop = np.array([entropy_hamming(query, t, weights) for t in targets])
        np.testing.assert_array_almost_equal(batch, loop)

    def test_diagonal_mahalanobis_batch_matches_scalar(self):
        rng = np.random.default_rng(79)
        query = rng.integers(0, 2, size=16, dtype=np.uint8)
        targets = rng.integers(0, 2, size=(10, 16), dtype=np.uint8)
        weights = rng.random(16)
        batch = _diagonal_mahalanobis_batch(query, targets, weights)
        loop = np.array([diagonal_mahalanobis(query, t, weights) for t in targets])
        np.testing.assert_array_almost_equal(batch, loop)


# ------------------------------------------------------------------
# fingerprint_smiles
# ------------------------------------------------------------------
class TestFingerprintSmiles:
    def test_returns_fps_and_valid_mask(self):
        smiles = ["CCO", "CCN", "CCC"]
        fps, valid_mask = fingerprint_smiles(smiles, "ECFP", 1024)
        assert fps.shape == (3, 1024)
        assert fps.dtype == np.uint8
        assert valid_mask.shape == (3,)
        assert valid_mask.all()

    def test_invalid_smiles_excluded(self):
        smiles = ["CCO", "INVALID_SMILES_XYZ", "CCC"]
        fps, valid_mask = fingerprint_smiles(smiles, "ECFP", 1024)
        assert fps.shape == (2, 1024)  # only 2 valid
        assert valid_mask.shape == (3,)
        assert valid_mask[0] is np.True_
        assert valid_mask[1] is np.False_
        assert valid_mask[2] is np.True_


# ------------------------------------------------------------------
# aggregate_and_save
# ------------------------------------------------------------------
class TestAggregateAndSave:
    def test_writes_correctly_prefixed_files(self, tmp_path):
        results = pd.DataFrame(
            {
                "query": [0, 0],
                "metric": ["tanimoto", "idf_tanimoto"],
                "auc": [0.7, 0.8],
                "ef1": [2.0, 3.0],
                "target": ["t1", "t1"],
            }
        )
        roc_data = {
            "t1": [
                {
                    "query": 0,
                    "metric": "tanimoto",
                    "fpr": np.array([0.0, 1.0]),
                    "tpr": np.array([0.0, 1.0]),
                }
            ]
        }

        aggregate_and_save([results], roc_data, "ECFP_fp_size1024", tmp_path, "test")

        assert (tmp_path / "eval_test_ECFP_fp_size1024.csv").exists()
        assert (tmp_path / "eval_test_summary_ECFP_fp_size1024.csv").exists()
        assert (tmp_path / "roc_curves_test_ECFP_fp_size1024.npz").exists()

        summary = pd.read_csv(tmp_path / "eval_test_summary_ECFP_fp_size1024.csv")
        assert set(summary.columns) == {"metric", "mean_auc", "std_auc", "mean_ef1", "std_ef1"}
        assert len(summary) == 2
