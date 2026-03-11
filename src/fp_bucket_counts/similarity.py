from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.linalg  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.covariance import shrunk_covariance  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# 1. Precomputation: weights & precision from corpus statistics
# ---------------------------------------------------------------------------


def covariance_from_cooccurrence(
    cooc: NDArray[np.uint64], total_molecules: int
) -> NDArray[np.float64]:
    """Exact binary covariance: Sigma_ij = cooc[i,j]/N - p_i * p_j."""
    if total_molecules == 0:
        return np.zeros_like(cooc, dtype=np.float64)
    n = total_molecules
    diag = np.diag(cooc).astype(np.float64)
    p = diag / n
    cov = cooc.astype(np.float64) / n - np.outer(p, p)
    return cov


def precision_matrix(
    cooc: NDArray[np.uint64],
    total_molecules: int,
    *,
    shrinkage: float = 0.1,
) -> NDArray[np.float64]:
    """Regularized inverse covariance (precision) matrix from co-occurrence counts."""
    cov = covariance_from_cooccurrence(cooc, total_molecules)
    if total_molecules == 0:
        return np.zeros_like(cov)
    cov_s = shrunk_covariance(cov, shrinkage=shrinkage)
    try:
        omega = scipy.linalg.inv(cov_s)
    except scipy.linalg.LinAlgError:
        eigvals, eigvecs = scipy.linalg.eigh(cov_s)
        eigvals = np.maximum(eigvals, 1e-10)
        omega = (eigvecs / eigvals) @ eigvecs.T
    omega = (omega + omega.T) / 2
    return np.asarray(omega, dtype=np.float64)


def idf_weights(
    bit_counts: NDArray[np.uint64], total_molecules: int, *, smooth: bool = True
) -> NDArray[np.float64]:
    """IDF-style per-bit weights: log(N / (1 + count)) or log(N / count)."""
    counts = bit_counts.astype(np.float64)
    n = float(total_molecules)
    if n == 0:
        return np.zeros_like(counts)
    if smooth:
        return np.log(n / (1.0 + counts))
    else:
        counts = np.maximum(counts, 1.0)
        return np.log(n / counts)


def entropy_weights(bit_counts: NDArray[np.uint64], total_molecules: int) -> NDArray[np.float64]:
    """Binary entropy H(p) per bit, in [0, 1]."""
    if total_molecules == 0:
        return np.zeros(len(bit_counts), dtype=np.float64)
    p = bit_counts.astype(np.float64) / total_molecules
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return np.where(np.isfinite(h), h, 0.0)


def diagonal_precision_weights(
    bit_counts: NDArray[np.uint64], total_molecules: int
) -> NDArray[np.float64]:
    """Diagonal of precision matrix: 1 / max(p*(1-p), eps)."""
    if total_molecules == 0:
        return np.zeros(len(bit_counts), dtype=np.float64)
    p = bit_counts.astype(np.float64) / total_molecules
    var = p * (1 - p)
    return 1.0 / np.maximum(var, 1e-12)


# ---------------------------------------------------------------------------
# 2. Pairwise scoring functions
# ---------------------------------------------------------------------------


def tanimoto(x: NDArray[np.uint8], y: NDArray[np.uint8]) -> float:
    """Standard Tanimoto similarity for binary vectors."""
    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    dot = xf @ yf
    denom = xf.sum() + yf.sum() - dot
    return 0.0 if denom == 0 else float(dot / denom)


def idf_tanimoto(x: NDArray[np.uint8], y: NDArray[np.uint8], weights: NDArray[np.float64]) -> float:
    """Weighted Tanimoto: sum(w*x*y) / (sum(w*x) + sum(w*y) - sum(w*x*y))."""
    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    wxy = weights * xf * yf
    num = wxy.sum()
    denom = (weights * xf).sum() + (weights * yf).sum() - num
    return 0.0 if denom == 0 else float(num / denom)


def entropy_hamming(
    x: NDArray[np.uint8], y: NDArray[np.uint8], weights: NDArray[np.float64]
) -> float:
    """Entropy-weighted normalized Hamming: sum(H * |delta|) / sum(H)."""
    delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return float((weights * delta).sum() / total_weight)


def diagonal_mahalanobis(
    x: NDArray[np.uint8], y: NDArray[np.uint8], weights: NDArray[np.float64]
) -> float:
    """Diagonal Mahalanobis squared distance: sum(w * delta^2)."""
    delta = x.astype(np.float64) - y.astype(np.float64)
    return float((weights * delta * delta).sum())


def mahalanobis(
    x: NDArray[np.uint8], y: NDArray[np.uint8], precision: NDArray[np.float64]
) -> float:
    """Full Mahalanobis squared distance: delta^T @ Omega @ delta, clamped >= 0."""
    delta = x.astype(np.float64) - y.astype(np.float64)
    d2 = float(delta @ precision @ delta)
    return max(0.0, d2)


# ---------------------------------------------------------------------------
# 3. Batch variants
# ---------------------------------------------------------------------------


def idf_tanimoto_batch(
    query: NDArray[np.uint8],
    targets: NDArray[np.uint8],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized IDF Tanimoto: query (d,) vs targets (n, d). Returns (n,) similarities."""
    qf = query.astype(np.float64)
    tf = targets.astype(np.float64)
    wq = weights * qf  # (d,)
    wt = tf * weights  # (n, d)
    wqt = wt * qf  # (n, d) — element-wise with broadcast
    num = wqt.sum(axis=1)
    denom = wq.sum() + wt.sum(axis=1) - num
    result = np.where(denom != 0, num / denom, 0.0)
    return result


def mahalanobis_batch(
    query: NDArray[np.uint8],
    targets: NDArray[np.uint8],
    precision: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized Mahalanobis d^2: query (d,) vs targets (n, d). Returns (n,) squared distances."""
    qf = query.astype(np.float64)
    deltas = qf - targets.astype(np.float64)  # (n, d)
    # (n, d) @ (d, d) → (n, d), then element-wise * deltas and sum
    od = deltas @ precision  # (n, d)
    d2 = (od * deltas).sum(axis=1)
    return np.asarray(np.maximum(d2, 0.0), dtype=np.float64)


# ---------------------------------------------------------------------------
# 4. I/O
# ---------------------------------------------------------------------------


def save_similarity_weights_npz(
    path: Path,
    *,
    idf: NDArray[np.float64],
    entropy: NDArray[np.float64],
    diagonal_precision: NDArray[np.float64],
    precision: NDArray[np.float64] | None,
    shrinkage: float,
    total_molecules: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prec = precision if precision is not None else np.array([])
    has_prec = np.array(precision is not None)
    np.savez_compressed(
        str(path),
        idf=idf,
        entropy=entropy,
        diagonal_precision=diagonal_precision,
        precision=prec,
        has_precision=has_prec,
        shrinkage=np.array(shrinkage),
        total_molecules=np.array(total_molecules),
    )


def load_similarity_weights_npz(path: Path) -> dict:
    data = np.load(path)
    has_precision = bool(data["has_precision"])
    return {
        "idf": data["idf"],
        "entropy": data["entropy"],
        "diagonal_precision": data["diagonal_precision"],
        "precision": data["precision"] if has_precision else None,
        "shrinkage": float(data["shrinkage"]),
        "total_molecules": int(data["total_molecules"]),
    }
