from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score, roc_curve  # type: ignore[import-untyped]

from .fingerprint import compute_fingerprints, create_fingerprinter
from .similarity import (
    idf_tanimoto_batch,
    mahalanobis_batch,
)

log = logging.getLogger(__name__)


def _identity_target(target: str) -> str:
    return target


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def fingerprint_smiles(
    smiles_list: list[str], fp_name: str, fp_size: int
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Fingerprint SMILES strings. Returns ``(fps, valid_mask)``.

    *fps* has shape ``(n_valid, fp_size)``; *valid_mask* has shape ``(len(smiles_list),)``
    with ``True`` for molecules that parsed successfully.
    """
    from rdkit.Chem import MolFromSmiles  # type: ignore[import-untyped]

    mols = [MolFromSmiles(s) for s in smiles_list]
    valid_mask = np.array([m is not None for m in mols], dtype=np.bool_)
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        log.warning("Skipping %d molecules that could not be parsed", n_invalid)
    valid_mols = [m for m in mols if m is not None]
    fingerprinter = create_fingerprinter(fp_name, fp_size)
    fps = compute_fingerprints(fingerprinter, valid_mols)
    return fps, valid_mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_auc(labels: NDArray[np.int8], scores: NDArray[np.float64]) -> float:
    """ROC AUC from binary labels and scores (higher score = more likely active)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_roc(
    labels: NDArray[np.int8], scores: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Compute ROC curve (FPR, TPR) from binary labels and scores."""
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr.astype(np.float64), tpr.astype(np.float64)


def compute_ef(
    labels: NDArray[np.int8], scores: NDArray[np.float64], fraction: float = 0.01
) -> float:
    """Enrichment factor at given fraction (default 1%).

    Sort by score descending, count actives in top fraction, divide by expected random count.
    """
    n_total = len(labels)
    n_actives = int(labels.sum())
    if n_actives == 0 or n_total == 0:
        return 0.0
    top_n = max(1, int(n_total * fraction))
    ranked_idx = np.argsort(scores)[::-1]
    top_labels = labels[ranked_idx[:top_n]]
    hits = int(top_labels.sum())
    expected = n_actives * fraction
    if expected == 0:
        return 0.0
    return float(hits / expected)


# ---------------------------------------------------------------------------
# Per-target evaluation
# ---------------------------------------------------------------------------


def _tanimoto_batch(query: NDArray[np.uint8], targets: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Vectorized standard Tanimoto: query (d,) vs targets (n, d)."""
    qf = query.astype(np.float64)
    tf = targets.astype(np.float64)
    dot = tf @ qf
    denom = qf.sum() + tf.sum(axis=1) - dot
    return np.where(denom != 0, dot / denom, 0.0)


def _entropy_hamming_batch(
    query: NDArray[np.uint8],
    targets: NDArray[np.uint8],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized entropy-weighted Hamming distance."""
    delta = np.abs(targets.astype(np.float64) - query.astype(np.float64))
    total_weight = weights.sum()
    if total_weight == 0:
        return np.zeros(len(targets), dtype=np.float64)
    result: NDArray[np.float64] = (delta * weights).sum(axis=1) / total_weight
    return result


def _diagonal_mahalanobis_batch(
    query: NDArray[np.uint8],
    targets: NDArray[np.uint8],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized diagonal Mahalanobis squared distance."""
    delta = targets.astype(np.float64) - query.astype(np.float64)
    result: NDArray[np.float64] = (weights * delta * delta).sum(axis=1)
    return result


def evaluate_target(
    fps: NDArray[np.uint8],
    actives_idx: NDArray[np.intp],
    decoys_idx: NDArray[np.intp],
    weights: dict,
    num_queries: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run virtual screening for one target.

    Returns (DataFrame with columns: query/metric/auc/ef1, list of ROC curve dicts).
    """
    n_actives = len(actives_idx)
    if n_actives <= num_queries:
        query_idx = actives_idx.copy()
    else:
        query_idx = rng.choice(actives_idx, size=num_queries, replace=False)

    remaining_actives = np.setdiff1d(actives_idx, query_idx)
    library_idx = np.concatenate([remaining_actives, decoys_idx])
    library_fps = fps[library_idx]
    labels = np.zeros(len(library_idx), dtype=np.int8)
    labels[: len(remaining_actives)] = 1

    idf_w = weights["idf"]
    entropy_w = weights["entropy"]
    diag_prec_w = weights["diagonal_precision"]
    precision_mat = weights.get("precision")

    rows: list[dict] = []
    roc_curves: list[dict] = []

    def _record(qi: int, metric: str, scores: NDArray[np.float64]) -> None:
        rows.append(
            {
                "query": qi,
                "metric": metric,
                "auc": compute_auc(labels, scores),
                "ef1": compute_ef(labels, scores),
            }
        )
        roc = compute_roc(labels, scores)
        if roc is not None:
            fpr, tpr = roc
            roc_curves.append({"query": qi, "metric": metric, "fpr": fpr, "tpr": tpr})

    for qi, qidx in enumerate(query_idx):
        query_fp = fps[qidx]

        # Tanimoto (similarity -- higher is better)
        tani_scores = _tanimoto_batch(query_fp, library_fps)
        _record(qi, "tanimoto", tani_scores)

        # IDF-Tanimoto (similarity -- higher is better)
        idf_scores = idf_tanimoto_batch(query_fp, library_fps, idf_w)
        _record(qi, "idf_tanimoto", idf_scores)

        # Entropy-Hamming (distance -- negate so higher = better)
        eh_dist = _entropy_hamming_batch(query_fp, library_fps, entropy_w)
        _record(qi, "entropy_hamming", -eh_dist)

        # Diagonal Mahalanobis (distance -- negate)
        dm_dist = _diagonal_mahalanobis_batch(query_fp, library_fps, diag_prec_w)
        _record(qi, "diagonal_mahalanobis", -dm_dist)

        # Full Mahalanobis (distance -- negate)
        if precision_mat is not None:
            fm_dist = mahalanobis_batch(query_fp, library_fps, precision_mat)
            _record(qi, "full_mahalanobis", -fm_dist)

    return pd.DataFrame(rows), roc_curves


# ---------------------------------------------------------------------------
# ROC curve persistence
# ---------------------------------------------------------------------------


def _sanitize_key(name: str) -> str:
    """Replace hyphens with underscores for NPZ key compatibility."""
    return name.replace("-", "_")


def save_roc_curves_npz(path: Path, roc_data: dict[str, list[dict]]) -> None:
    """Save per-target ROC curve data to a compressed NPZ file.

    *roc_data* maps target names to lists of
    ``{"query": int, "metric": str, "fpr": array, "tpr": array}`` dicts.
    """
    arrays: dict[str, Any] = {}
    for target, curves in roc_data.items():
        t = _sanitize_key(target)
        for entry in curves:
            m = entry["metric"]
            q = entry["query"]
            arrays[f"{t}__{m}__q{q}_fpr"] = entry["fpr"]
            arrays[f"{t}__{m}__q{q}_tpr"] = entry["tpr"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)
    log.info("Wrote ROC curves (%d arrays) to %s", len(arrays), path)


def load_roc_curves_npz(
    path: Path,
    *,
    unsanitize_target: Callable[[str], str] | None = None,
) -> dict[str, list[dict]]:
    """Load ROC curves previously saved by :func:`save_roc_curves_npz`.

    Returns dict mapping target names to lists of
    ``{"query": int, "metric": str, "fpr": array, "tpr": array}`` dicts.

    *unsanitize_target* is an optional callable to reverse key sanitization
    (e.g. ``MUV_466`` -> ``MUV-466``).  Defaults to identity (no transformation).
    """
    if unsanitize_target is None:
        unsanitize_target = _identity_target

    data = np.load(path)
    result: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for key in data.files:
        if not key.endswith("_fpr"):
            continue
        # key format: {target}__{metric}__q{query}_fpr
        prefix = key.removesuffix("_fpr")
        if prefix in seen:
            continue
        seen.add(prefix)
        parts = prefix.split("__")
        target_sanitized = parts[0]
        metric = parts[1]
        query = int(parts[2].removeprefix("q"))
        target = unsanitize_target(target_sanitized)
        tpr_key = prefix + "_tpr"
        entry = {
            "query": query,
            "metric": metric,
            "fpr": data[key].astype(np.float64),
            "tpr": data[tpr_key].astype(np.float64),
        }
        result.setdefault(target, []).append(entry)
    return result


# ---------------------------------------------------------------------------
# FP config parsing
# ---------------------------------------------------------------------------


def _parse_fp_config(label: str) -> tuple[str, int]:
    """Extract fingerprint name and size from a weight file label.

    E.g. 'ECFP_fp_size2048' -> ('ECFP', 2048).
    """
    parts = label.split("_")
    fp_name = parts[0]
    fp_size = 2048  # default
    for i, p in enumerate(parts):
        if p == "fp" and i + 1 < len(parts) and parts[i + 1].startswith("size"):
            fp_size = int(parts[i + 1].removeprefix("size"))
            break
    return fp_name, fp_size


# ---------------------------------------------------------------------------
# Aggregation & output
# ---------------------------------------------------------------------------


def aggregate_and_save(
    all_results: list[pd.DataFrame],
    all_roc: dict[str, list[dict]],
    label: str,
    output_dir: Path,
    benchmark_id: str,
) -> None:
    """Aggregate per-target results into summary CSVs and save ROC NPZ.

    Produces:
    - ``eval_{benchmark_id}_{label}.csv`` (per-target, per-metric means)
    - ``eval_{benchmark_id}_summary_{label}.csv`` (summary across targets)
    - ``roc_curves_{benchmark_id}_{label}.npz``
    """
    results_df = pd.concat(all_results, ignore_index=True)

    # Per-target, per-metric averages
    per_target = results_df.groupby(["target", "metric"])[["auc", "ef1"]].mean().reset_index()
    per_target_path = output_dir / f"eval_{benchmark_id}_{label}.csv"
    per_target.to_csv(per_target_path, index=False)
    log.info("Wrote %s", per_target_path)

    # Summary across targets
    summary = (
        per_target.groupby("metric")
        .agg(
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_ef1=("ef1", "mean"),
            std_ef1=("ef1", "std"),
        )
        .reset_index()
    )
    summary_path = output_dir / f"eval_{benchmark_id}_summary_{label}.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Wrote %s", summary_path)

    # Log summary
    log.info("Summary for %s:", label)
    for _, row in summary.iterrows():
        log.info(
            "  %-25s  AUC=%.3f +/- %.3f  EF1%%=%.2f +/- %.2f",
            row["metric"],
            row["mean_auc"],
            row["std_auc"],
            row["mean_ef1"],
            row["std_ef1"],
        )

    # Save ROC curve data
    roc_path = output_dir / f"roc_curves_{benchmark_id}_{label}.npz"
    save_roc_curves_npz(roc_path, all_roc)
