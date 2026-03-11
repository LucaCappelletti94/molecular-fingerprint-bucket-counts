from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

from .fingerprint import compute_fingerprints, create_fingerprinter
from .similarity import (
    idf_tanimoto_batch,
    load_similarity_weights_npz,
    mahalanobis_batch,
)

log = logging.getLogger(__name__)

MUV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz"
MUV_TARGETS = [
    "MUV-466",
    "MUV-548",
    "MUV-600",
    "MUV-644",
    "MUV-652",
    "MUV-689",
    "MUV-692",
    "MUV-712",
    "MUV-713",
    "MUV-733",
    "MUV-737",
    "MUV-810",
    "MUV-832",
    "MUV-846",
    "MUV-852",
    "MUV-858",
    "MUV-859",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def download_muv(data_dir: Path) -> Path:
    """Download muv.csv.gz if not already cached. Return path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "muv.csv.gz"
    if dest.exists():
        log.info("MUV data already cached: %s", dest)
        return dest

    from .download import download_file

    download_file(MUV_URL, dest)
    return dest


def load_muv(path: Path) -> pd.DataFrame:
    """Parse MUV CSV. Returns DataFrame with 'smiles' and target columns."""
    df = pd.read_csv(path, compression="gzip" if str(path).endswith(".gz") else None)
    # Rename columns: the CSV uses 'mol_id' and 'smiles' plus target columns
    # Keep only smiles and target columns
    cols_to_keep = ["smiles"] + [c for c in df.columns if c.startswith("MUV-")]
    df = df[cols_to_keep].copy()
    # Drop rows without a SMILES string
    df = df.dropna(subset=["smiles"]).reset_index(drop=True)
    return df


def fingerprint_muv(smiles_list: list[str], fp_name: str, fp_size: int) -> NDArray[np.uint8]:
    """Fingerprint MUV molecules from SMILES. Returns (n_mols, fp_size) uint8 array."""
    from rdkit.Chem import MolFromSmiles  # type: ignore[import-untyped]

    mols = [MolFromSmiles(s) for s in smiles_list]
    valid_mask = [m is not None for m in mols]
    if not all(valid_mask):
        n_invalid = sum(1 for v in valid_mask if not v)
        log.warning("Skipping %d molecules that could not be parsed", n_invalid)
    valid_mols = [m for m in mols if m is not None]
    fingerprinter = create_fingerprinter(fp_name, fp_size)
    return compute_fingerprints(fingerprinter, valid_mols)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_auc(labels: NDArray[np.int8], scores: NDArray[np.float64]) -> float:
    """ROC AUC from binary labels and scores (higher score = more likely active)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


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
) -> pd.DataFrame:
    """Run virtual screening for one target.

    Returns DataFrame with columns: query, metric, auc, ef1.
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

    for qi, qidx in enumerate(query_idx):
        query_fp = fps[qidx]

        # Tanimoto (similarity — higher is better)
        tani_scores = _tanimoto_batch(query_fp, library_fps)
        rows.append(
            {
                "query": qi,
                "metric": "tanimoto",
                "auc": compute_auc(labels, tani_scores),
                "ef1": compute_ef(labels, tani_scores),
            }
        )

        # IDF-Tanimoto (similarity — higher is better)
        idf_scores = idf_tanimoto_batch(query_fp, library_fps, idf_w)
        rows.append(
            {
                "query": qi,
                "metric": "idf_tanimoto",
                "auc": compute_auc(labels, idf_scores),
                "ef1": compute_ef(labels, idf_scores),
            }
        )

        # Entropy-Hamming (distance — negate so higher = better)
        eh_dist = _entropy_hamming_batch(query_fp, library_fps, entropy_w)
        rows.append(
            {
                "query": qi,
                "metric": "entropy_hamming",
                "auc": compute_auc(labels, -eh_dist),
                "ef1": compute_ef(labels, -eh_dist),
            }
        )

        # Diagonal Mahalanobis (distance — negate)
        dm_dist = _diagonal_mahalanobis_batch(query_fp, library_fps, diag_prec_w)
        rows.append(
            {
                "query": qi,
                "metric": "diagonal_mahalanobis",
                "auc": compute_auc(labels, -dm_dist),
                "ef1": compute_ef(labels, -dm_dist),
            }
        )

        # Full Mahalanobis (distance — negate)
        if precision_mat is not None:
            fm_dist = mahalanobis_batch(query_fp, library_fps, precision_mat)
            rows.append(
                {
                    "query": qi,
                    "metric": "full_mahalanobis",
                    "auc": compute_auc(labels, -fm_dist),
                    "ef1": compute_ef(labels, -fm_dist),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main evaluation loop
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


def run_muv_evaluation(weights_dir: Path, output_dir: Path, num_queries: int, seed: int) -> None:
    """Main evaluation loop across all weight files."""
    weight_files = sorted(weights_dir.glob("sim_weights_*.npz"))
    if not weight_files:
        log.warning("No sim_weights_*.npz files found in %s", weights_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    muv_path = download_muv(data_dir)
    df = load_muv(muv_path)
    log.info("Loaded MUV: %d molecules", len(df))

    for wf in weight_files:
        label = wf.stem.removeprefix("sim_weights_")
        log.info("=== Evaluating %s ===", label)
        fp_name, fp_size = _parse_fp_config(label)

        weights = load_similarity_weights_npz(wf)

        # Fingerprint all molecules
        smiles_list = df["smiles"].tolist()
        fps = fingerprint_muv(smiles_list, fp_name, fp_size)

        # Track which SMILES produced valid molecules
        from rdkit.Chem import MolFromSmiles  # type: ignore[import-untyped]

        valid_mask = np.array([MolFromSmiles(s) is not None for s in smiles_list], dtype=bool)
        # Map from valid-molecule index back to DataFrame row index
        valid_df_idx = np.where(valid_mask)[0]

        rng = np.random.default_rng(seed)
        all_results: list[pd.DataFrame] = []

        for target in MUV_TARGETS:
            if target not in df.columns:
                continue
            target_col = df[target]

            # Find actives and decoys among valid molecules
            actives_df_idx = np.array(
                [i for i in valid_df_idx if target_col.iloc[i] == 1], dtype=np.intp
            )
            decoys_df_idx = np.array(
                [i for i in valid_df_idx if target_col.iloc[i] == 0], dtype=np.intp
            )

            if len(actives_df_idx) == 0 or len(decoys_df_idx) == 0:
                log.warning("Skipping %s: no actives or no decoys", target)
                continue

            # Convert DataFrame indices to fingerprint-array indices
            df_to_fp = {df_i: fp_i for fp_i, df_i in enumerate(valid_df_idx)}
            actives_fp_idx = np.array([df_to_fp[i] for i in actives_df_idx], dtype=np.intp)
            decoys_fp_idx = np.array([df_to_fp[i] for i in decoys_df_idx], dtype=np.intp)

            log.info(
                "  %s: %d actives, %d decoys",
                target,
                len(actives_fp_idx),
                len(decoys_fp_idx),
            )
            result = evaluate_target(fps, actives_fp_idx, decoys_fp_idx, weights, num_queries, rng)
            result["target"] = target

            # Average across queries
            all_results.append(result)

        if not all_results:
            log.warning("No targets evaluated for %s", label)
            continue

        results_df = pd.concat(all_results, ignore_index=True)

        # Per-target, per-metric averages
        per_target = results_df.groupby(["target", "metric"])[["auc", "ef1"]].mean().reset_index()
        per_target_path = output_dir / f"eval_muv_{label}.csv"
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
        summary_path = output_dir / f"eval_muv_summary_{label}.csv"
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate similarity metrics on MUV virtual screening benchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Where to write evaluation results (default: output/)",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing sim_weights_*.npz files (default: output/)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of active queries per target (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query selection (default: 42)",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    run_muv_evaluation(args.weights_dir, args.output_dir, args.num_queries, args.seed)


if __name__ == "__main__":
    main()
