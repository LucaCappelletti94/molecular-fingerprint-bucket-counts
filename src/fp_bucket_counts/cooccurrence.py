from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

MAX_DENSE_COOCCURRENCE_FP_SIZE = 2048
MAX_HEATMAP_FP_SIZE = 4096


def supports_dense_cooccurrence(fp_size: int) -> bool:
    return fp_size <= MAX_DENSE_COOCCURRENCE_FP_SIZE


def dense_cooccurrence_skip_reason(fp_size: int) -> str:
    return (
        f"dense co-occurrence skipped for fp_size={fp_size} "
        f"(>{MAX_DENSE_COOCCURRENCE_FP_SIZE} bits)"
    )


def merge_worker_cooccurrence(tmp_dir: Path, config_index: int, fp_size: int) -> np.ndarray:
    """Scan temp dir for per-worker cooccurrence .npy files, sum them, return merged matrix."""
    pattern = f"cooc_*_{config_index}.npy"
    files = sorted(tmp_dir.glob(pattern))
    merged = np.zeros((fp_size, fp_size), dtype=np.uint64)
    if not files:
        log.warning("No worker files found for config %d — matrix will be all zeros", config_index)
    for f in files:
        arr = np.load(f)
        merged += arr.astype(np.uint64)
        f.unlink()
    log.info("Merged %d worker files for config %d (fp_size=%d)", len(files), config_index, fp_size)
    return merged


def save_cooccurrence_npz(matrix: np.ndarray, path: Path, total_molecules: int) -> None:
    """Store upper triangle (including diagonal) as compressed .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fp_size = matrix.shape[0]
    upper_triangle = matrix[np.triu_indices(fp_size)].astype(np.uint64)
    np.savez_compressed(
        path,
        upper_triangle=upper_triangle,
        fp_size=np.array(fp_size),
        total_molecules=np.array(total_molecules),
    )


def save_skipped_cooccurrence_npz(
    path: Path,
    fp_size: int,
    total_molecules: int,
    reason: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        upper_triangle=np.array([], dtype=np.uint64),
        fp_size=np.array(fp_size),
        total_molecules=np.array(total_molecules),
        skip_reason=np.array(reason),
    )


def load_cooccurrence_npz(path: Path) -> tuple[np.ndarray, int]:
    """Reconstruct full symmetric matrix from stored upper triangle."""
    data = np.load(path)
    if "skip_reason" in data.files:
        reason = str(data["skip_reason"].item())
        raise ValueError(f"Co-occurrence matrix not stored: {reason}")
    fp_size = int(data["fp_size"])
    total_molecules = int(data["total_molecules"])
    upper_triangle = data["upper_triangle"]
    matrix = np.zeros((fp_size, fp_size), dtype=np.uint64)
    ri, ci = np.triu_indices(fp_size)
    matrix[ri, ci] = upper_triangle
    matrix[ci, ri] = upper_triangle
    return matrix, total_molecules


def compute_pmi_matrix(cooc: np.ndarray, total_molecules: int) -> np.ndarray:
    """Compute PMI(i,j) = log2(P(i,j) / (P(i)*P(j))). Diagonal set to 0."""
    n = total_molecules
    if n == 0:
        return np.zeros_like(cooc, dtype=np.float64)

    diag = np.diag(cooc).astype(np.float64)
    p_i = diag / n  # P(bit i is set)
    p_ij = cooc.astype(np.float64) / n  # P(both i and j set)

    # outer product P(i)*P(j) — expected co-occurrence under independence
    expected = np.outer(p_i, p_i)

    # Avoid log(0) and division by 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(p_ij / expected)

    # Clean up: NaN/inf from zero counts → 0
    pmi = np.where(np.isfinite(pmi), pmi, 0.0)
    np.fill_diagonal(pmi, 0.0)
    return pmi


def _compute_sparse_pmi(
    cooc: np.ndarray, total_molecules: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute PMI only for non-zero upper-triangle pairs. Returns (ri, ci, counts, pmi)."""
    n = total_molecules
    diag = np.diag(cooc).astype(np.float64)

    # Find non-zero entries in the upper triangle (excluding diagonal)
    ri, ci = np.nonzero(np.triu(cooc, k=1))
    counts = cooc[ri, ci]

    if len(counts) == 0 or n == 0:
        empty = np.array([], dtype=np.float64)
        empty_int = np.array([], dtype=np.intp)
        return empty_int, empty_int, np.array([], dtype=np.uint64), empty

    p_ij = counts.astype(np.float64) / n
    p_i = diag[ri] / n
    p_j = diag[ci] / n
    expected = p_i * p_j

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(p_ij / expected)
    pmi = np.where(np.isfinite(pmi), pmi, 0.0)

    return ri, ci, counts, pmi


def save_cooccurrence_summary_csv(
    cooc: np.ndarray,
    path: Path,
    total_molecules: int,
    k: int = 1000,
) -> None:
    """CSV with top-k most positively and negatively correlated bit pairs by PMI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = total_molecules
    ri, ci, counts, pmi_vals = _compute_sparse_pmi(cooc, n)
    diag = np.diag(cooc).astype(np.float64)

    # Top-k positive and negative
    if len(pmi_vals) == 0:
        top_indices = np.array([], dtype=int)
    elif len(pmi_vals) <= 2 * k:
        top_indices = np.arange(len(pmi_vals))
    else:
        top_pos = np.argpartition(pmi_vals, -k)[-k:]
        top_neg = np.argpartition(pmi_vals, k)[:k]
        top_indices = np.unique(np.concatenate([top_pos, top_neg]))

    with open(path, "w", newline="") as f:
        f.write(f"# total_molecules: {total_molecules}\n")
        writer = csv.writer(f)
        writer.writerow(["bit_i", "bit_j", "count", "fraction", "expected_fraction", "pmi"])

        # Sort by absolute PMI descending
        sorted_idx = top_indices[np.argsort(-np.abs(pmi_vals[top_indices]))]
        for idx in sorted_idx:
            bi, bj = int(ri[idx]), int(ci[idx])
            count = int(counts[idx])
            fraction = count / n if n > 0 else 0.0
            expected = (diag[bi] / n) * (diag[bj] / n) if n > 0 else 0.0
            writer.writerow(
                [bi, bj, count, f"{fraction:.8f}", f"{expected:.8f}", f"{pmi_vals[idx]:.6f}"]
            )


def save_skipped_cooccurrence_summary_csv(
    path: Path,
    total_molecules: int,
    reason: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(f"# total_molecules: {total_molecules}\n")
        f.write(f"# skipped: {reason}\n")
        writer = csv.writer(f)
        writer.writerow(["bit_i", "bit_j", "count", "fraction", "expected_fraction", "pmi"])


def _plot_placeholder_heatmap(path: Path, label: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, f"{label}\n{message}", ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)


def plot_skipped_cooccurrence_heatmap(
    path: Path,
    label: str,
    fp_size: int,
    reason: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _plot_placeholder_heatmap(path, label, f"fp_size={fp_size}\n{reason}")


def plot_cooccurrence_heatmap(
    cooc: np.ndarray,
    path: Path,
    total_molecules: int,
    label: str,
) -> None:
    """PMI heatmap with a placeholder for fingerprints that are too wide to render."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fp_size = cooc.shape[0]

    if fp_size > MAX_HEATMAP_FP_SIZE:
        _plot_placeholder_heatmap(
            path,
            label,
            f"fp_size={fp_size}\nheatmap skipped\n(>{MAX_HEATMAP_FP_SIZE} bits)",
        )
        return

    pmi = compute_pmi_matrix(cooc, total_molecules)

    max_pixels = 512
    if fp_size > max_pixels:
        block = fp_size // max_pixels
        trimmed = fp_size - (fp_size % block)
        pmi_trimmed = pmi[:trimmed, :trimmed]
        pmi_down = pmi_trimmed.reshape(trimmed // block, block, trimmed // block, block).mean(
            axis=(1, 3)
        )
    else:
        pmi_down = pmi

    vmax = np.percentile(np.abs(pmi_down[pmi_down != 0]), 99) if np.any(pmi_down != 0) else 1.0

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(
        pmi_down, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest"
    )
    ax.set_xlabel("Bit position")
    ax.set_ylabel("Bit position")
    ax.set_title(f"{label}: PMI co-occurrence (n={total_molecules:,})")
    fig.colorbar(im, ax=ax, label="PMI (bits)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
