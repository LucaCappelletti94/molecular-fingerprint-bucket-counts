from __future__ import annotations

import numpy as np
import pytest

from fp_bucket_counts.analysis import save_counts_csv
from fp_bucket_counts.cooccurrence import save_cooccurrence_npz, save_skipped_cooccurrence_npz
from fp_bucket_counts.sim_cli import load_bit_counts_csv, run_sim_weights
from fp_bucket_counts.similarity import load_similarity_weights_npz


@pytest.fixture
def small_corpus():
    """4 molecules, 3-bit fingerprints."""
    fps = np.array(
        [[1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1]],
        dtype=np.uint8,
    )
    cooc = (fps.T @ fps).astype(np.uint64)
    counts = fps.sum(axis=0).astype(np.uint64)
    n = 4
    return fps, cooc, counts, n


def test_sim_cli_produces_weights(tmp_path, small_corpus):
    _, cooc, counts, n = small_corpus

    save_counts_csv(counts, tmp_path / "bit_counts_FOO.csv", n)
    save_cooccurrence_npz(cooc, tmp_path / "cooc_FOO.npz", n)

    run_sim_weights(tmp_path)

    weights_path = tmp_path / "sim_weights_FOO.npz"
    assert weights_path.exists()

    loaded = load_similarity_weights_npz(weights_path)
    assert loaded["idf"].shape == (3,)
    assert loaded["entropy"].shape == (3,)
    assert loaded["diagonal_precision"].shape == (3,)
    assert loaded["precision"] is not None
    assert loaded["precision"].shape == (3, 3)
    assert loaded["total_molecules"] == n
    assert loaded["shrinkage"] == 0.1


def test_sim_cli_no_cooc(tmp_path, small_corpus):
    _, _, counts, n = small_corpus

    save_counts_csv(counts, tmp_path / "bit_counts_BAR.csv", n)
    # No cooc file at all

    run_sim_weights(tmp_path)

    weights_path = tmp_path / "sim_weights_BAR.npz"
    assert weights_path.exists()

    loaded = load_similarity_weights_npz(weights_path)
    assert loaded["precision"] is None
    assert loaded["idf"].shape == (3,)


def test_sim_cli_skipped_cooc(tmp_path, small_corpus):
    _, _, counts, n = small_corpus

    save_counts_csv(counts, tmp_path / "bit_counts_BAZ.csv", n)
    save_skipped_cooccurrence_npz(
        tmp_path / "cooc_BAZ.npz", fp_size=3, total_molecules=n, reason="too large"
    )

    run_sim_weights(tmp_path)

    weights_path = tmp_path / "sim_weights_BAZ.npz"
    assert weights_path.exists()

    loaded = load_similarity_weights_npz(weights_path)
    assert loaded["precision"] is None
    assert loaded["idf"].shape == (3,)


def test_load_bit_counts_csv_roundtrip(tmp_path, small_corpus):
    _, _, counts, n = small_corpus
    csv_path = tmp_path / "bit_counts_RT.csv"

    save_counts_csv(counts, csv_path, n)
    loaded_counts, loaded_n = load_bit_counts_csv(csv_path)

    np.testing.assert_array_equal(loaded_counts, counts)
    assert loaded_n == n
