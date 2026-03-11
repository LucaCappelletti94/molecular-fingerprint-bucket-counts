from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from fp_bucket_counts.eval_muv import (
    _diagonal_mahalanobis_batch,
    _entropy_hamming_batch,
    _tanimoto_batch,
    compute_auc,
    compute_ef,
    evaluate_target,
    load_muv,
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
        # expected = 5 * 0.01 = 0.05 → EF = 5 / 0.05 = 100
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

        result = evaluate_target(fps, actives_idx, decoys_idx, weights, 2, rng)

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

        result = evaluate_target(fps, actives_idx, decoys_idx, weights, 2, rng)
        # 2 queries x 4 metrics (no full mahalanobis) = 8 rows
        assert len(result) == 8
        assert "full_mahalanobis" not in result["metric"].values


# ------------------------------------------------------------------
# Cross-validate eval_muv batch functions against similarity scalars
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
# load_muv with synthetic CSV
# ------------------------------------------------------------------
class TestLoadMuv:
    def test_parse_synthetic_csv(self, tmp_path):
        csv_content = textwrap.dedent("""\
            mol_id,smiles,MUV-466,MUV-548
            1,CCO,1,0
            2,CCN,,1
            3,CCC,0,0
            4,,0,1
        """)
        csv_path = tmp_path / "muv.csv"
        csv_path.write_text(csv_content)

        df = load_muv(csv_path)
        # Row 4 has no SMILES, should be dropped
        assert len(df) == 3
        assert "smiles" in df.columns
        assert "MUV-466" in df.columns
        assert "MUV-548" in df.columns
        assert "mol_id" not in df.columns
        assert df.iloc[0]["smiles"] == "CCO"
