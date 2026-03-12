from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .eval_common import (  # noqa: F401
    _diagonal_mahalanobis_batch,
    _entropy_hamming_batch,
    _parse_fp_config,
    _tanimoto_batch,
    aggregate_and_save,
    compute_auc,
    compute_ef,
    compute_roc,
    evaluate_target,
    fingerprint_smiles,
    load_roc_curves_npz,
    save_roc_curves_npz,
)
from .similarity import load_similarity_weights_npz

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


def _unsanitize_key(key: str) -> str:
    """Reverse ``_sanitize_key``: ``MUV_466`` -> ``MUV-466``."""
    return key.replace("MUV_", "MUV-", 1)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


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
        fps, valid_mask = fingerprint_smiles(smiles_list, fp_name, fp_size)

        # Map from valid-molecule index back to DataFrame row index
        valid_df_idx = np.where(valid_mask)[0]

        rng = np.random.default_rng(seed)
        all_results: list[pd.DataFrame] = []
        all_roc: dict[str, list[dict]] = {}

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
            result, target_roc = evaluate_target(
                fps, actives_fp_idx, decoys_fp_idx, weights, num_queries, rng
            )
            result["target"] = target
            all_roc[target] = target_roc

            # Average across queries
            all_results.append(result)

        if not all_results:
            log.warning("No targets evaluated for %s", label)
            continue

        aggregate_and_save(all_results, all_roc, label, output_dir, "muv")


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
