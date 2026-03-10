from __future__ import annotations

import numpy as np
import pytest

from fp_bucket_counts.cooccurrence import (
    compute_pmi_matrix,
    load_cooccurrence_npz,
    merge_worker_cooccurrence,
    save_cooccurrence_npz,
    save_cooccurrence_summary_csv,
    save_skipped_cooccurrence_npz,
)


def test_cooccurrence_symmetric():
    """fps.T @ fps should be symmetric and diagonal should equal per-bit counts."""
    rng = np.random.default_rng(42)
    fps = rng.integers(0, 2, size=(100, 16), dtype=np.uint8)
    cooc = fps.T @ fps
    np.testing.assert_array_equal(cooc, cooc.T)
    np.testing.assert_array_equal(np.diag(cooc), fps.sum(axis=0))


def test_merge_worker_files(tmp_path):
    """Synthetic .npy files merge to correct sum."""
    fp_size = 8
    config_idx = 3
    a = np.ones((fp_size, fp_size), dtype=np.uint32) * 5
    b = np.ones((fp_size, fp_size), dtype=np.uint32) * 7
    np.save(tmp_path / f"cooc_1001_{config_idx}.npy", a)
    np.save(tmp_path / f"cooc_1002_{config_idx}.npy", b)

    merged = merge_worker_cooccurrence(tmp_path, config_idx, fp_size)
    np.testing.assert_array_equal(merged, np.full((fp_size, fp_size), 12, dtype=np.uint64))

    # Verify worker files were cleaned up
    assert list(tmp_path.glob(f"cooc_*_{config_idx}.npy")) == []


def test_merge_no_files(tmp_path):
    """Merge with no files returns zeros."""
    merged = merge_worker_cooccurrence(tmp_path, 0, 4)
    np.testing.assert_array_equal(merged, np.zeros((4, 4), dtype=np.uint64))


def test_npz_roundtrip(tmp_path):
    """save -> load produces identical matrix."""
    rng = np.random.default_rng(123)
    fp_size = 16
    fps = rng.integers(0, 2, size=(50, fp_size), dtype=np.uint8)
    cooc = (fps.T @ fps).astype(np.uint64)

    path = tmp_path / "test_cooc.npz"
    save_cooccurrence_npz(cooc, path, total_molecules=50)
    loaded, total = load_cooccurrence_npz(path)

    assert total == 50
    np.testing.assert_array_equal(loaded, cooc)


def test_skipped_npz_raises_on_load(tmp_path):
    path = tmp_path / "skipped_cooc.npz"
    save_skipped_cooccurrence_npz(path, fp_size=4096, total_molecules=50, reason="too large")

    with pytest.raises(ValueError, match="too large"):
        load_cooccurrence_npz(path)


def test_pmi_known_values():
    """Hand-computed PMI for a small example."""
    # 4 molecules, 3 bits:
    # mol0: [1, 1, 0]
    # mol1: [1, 1, 0]
    # mol2: [0, 0, 1]
    # mol3: [1, 0, 1]
    fps = np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    cooc = (fps.T @ fps).astype(np.uint64)
    n = 4

    # P(0) = 3/4, P(1) = 2/4, P(2) = 2/4
    # P(0,1) = 2/4, expected = 3/4 * 2/4 = 6/16 = 3/8
    # PMI(0,1) = log2((2/4) / (3/8)) = log2(4/3)
    pmi = compute_pmi_matrix(cooc, n)

    assert pmi.shape == (3, 3)
    np.testing.assert_almost_equal(pmi[0, 1], np.log2(4.0 / 3.0), decimal=10)
    np.testing.assert_almost_equal(pmi[1, 0], pmi[0, 1])  # symmetric

    # P(0,2) = 1/4, expected = 3/4 * 2/4 = 3/8
    # PMI(0,2) = log2((1/4) / (3/8)) = log2(2/3)
    np.testing.assert_almost_equal(pmi[0, 2], np.log2(2.0 / 3.0), decimal=10)

    # Diagonal should be 0
    np.testing.assert_array_equal(np.diag(pmi), 0.0)


def test_pmi_zero_molecules():
    """PMI with 0 molecules should return zeros."""
    cooc = np.zeros((4, 4), dtype=np.uint64)
    pmi = compute_pmi_matrix(cooc, 0)
    np.testing.assert_array_equal(pmi, 0.0)


def test_top_pairs_sorted(tmp_path):
    """Summary CSV has pairs sorted by absolute PMI descending."""
    rng = np.random.default_rng(99)
    fp_size = 8
    fps = rng.integers(0, 2, size=(200, fp_size), dtype=np.uint8)
    cooc = (fps.T @ fps).astype(np.uint64)
    n = 200

    path = tmp_path / "summary.csv"
    save_cooccurrence_summary_csv(cooc, path, n, k=10)

    lines = path.read_text().strip().split("\n")
    assert lines[0].startswith("# total_molecules:")
    assert lines[1] == "bit_i,bit_j,count,fraction,expected_fraction,pmi"

    # Parse PMI values and check they're sorted by |PMI| descending
    pmi_vals = []
    for line in lines[2:]:
        parts = line.split(",")
        pmi_vals.append(float(parts[5]))

    abs_pmis = [abs(v) for v in pmi_vals]
    assert abs_pmis == sorted(abs_pmis, reverse=True)
