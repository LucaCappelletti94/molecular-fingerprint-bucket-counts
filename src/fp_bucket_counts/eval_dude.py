from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .eval_common import (
    _parse_fp_config,
    aggregate_and_save,
    evaluate_target,
    fingerprint_smiles,
)
from .similarity import load_similarity_weights_npz

log = logging.getLogger(__name__)

DUDE_BASE_URL = "https://dude.docking.org/targets"
DUDE_TARGETS = [
    "aa2ar",
    "abl1",
    "ace",
    "aces",
    "ada",
    "ada17",
    "adrb1",
    "adrb2",
    "akt1",
    "akt2",
    "aldr",
    "ampc",
    "andr",
    "aofb",
    "bace1",
    "braf",
    "cah2",
    "casp3",
    "cdk2",
    "comt",
    "cp2c9",
    "cp3a4",
    "csf1r",
    "cxcr4",
    "def",
    "dhi1",
    "dpp4",
    "drd3",
    "dyr",
    "egfr",
    "esr1",
    "esr2",
    "fa10",
    "fa7",
    "fabp4",
    "fak1",
    "fgfr1",
    "fkb1a",
    "fnta",
    "fpps",
    "gcr",
    "glcm",
    "gria2",
    "grik1",
    "hdac2",
    "hdac8",
    "hivint",
    "hivpr",
    "hivrt",
    "hmdh",
    "hs90a",
    "hxk4",
    "igf1r",
    "inha",
    "ital",
    "jak2",
    "kif11",
    "kit",
    "kpcb",
    "lck",
    "lkha4",
    "mapk2",
    "mcr",
    "met",
    "mk01",
    "mk10",
    "mk14",
    "mmp13",
    "mp2k1",
    "nram",
    "pa2ga",
    "parp1",
    "pde5a",
    "pgh1",
    "pgh2",
    "plk1",
    "pnph",
    "ppara",
    "ppard",
    "pparg",
    "prgr",
    "pur2",
    "pygm",
    "pyrd",
    "reni",
    "rock1",
    "rxra",
    "sahh",
    "src",
    "tgfr1",
    "thb",
    "thrb",
    "try1",
    "tryb1",
    "tysy",
    "urok",
    "vgfr2",
    "wee1",
    "xiap",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _download_with_retries(url: str, dest: Path, max_retries: int = 3) -> None:
    """Download a file with retries on transient HTTP errors."""
    import time
    import urllib.error

    from .download import download_file

    for attempt in range(1, max_retries + 1):
        try:
            download_file(url, dest)
            return
        except urllib.error.HTTPError as exc:
            if exc.code >= 500 and attempt < max_retries:
                wait = 2**attempt
                log.warning(
                    "HTTP %d for %s (attempt %d/%d), retrying in %ds",
                    exc.code, url, attempt, max_retries, wait,
                )
                time.sleep(wait)
            else:
                raise


def download_dude_target(target: str, data_dir: Path) -> Path:
    """Download actives_final.ism and decoys_final.ism for one target. Return target dir."""
    target_dir = data_dir / target
    actives_path = target_dir / "actives_final.ism"
    decoys_path = target_dir / "decoys_final.ism"

    if actives_path.exists() and decoys_path.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    base = f"{DUDE_BASE_URL}/{target}"
    if not actives_path.exists():
        _download_with_retries(f"{base}/actives_final.ism", actives_path)
    if not decoys_path.exists():
        _download_with_retries(f"{base}/decoys_final.ism", decoys_path)
    return target_dir


def load_ism(path: Path) -> list[str]:
    """Load SMILES from an ISM file (one SMILES per line, optional ID after space)."""
    smiles: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smiles.append(line.split()[0])
    return smiles


def load_dude_target(target_dir: Path) -> tuple[list[str], list[str]]:
    """Load active and decoy SMILES for one DUD-E target directory."""
    actives = load_ism(target_dir / "actives_final.ism")
    decoys = load_ism(target_dir / "decoys_final.ism")
    return actives, decoys


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_dude_evaluation(weights_dir: Path, output_dir: Path, num_queries: int, seed: int) -> None:
    """Main DUD-E evaluation loop across all weight files."""
    weight_files = sorted(weights_dir.glob("sim_weights_*.npz"))
    if not weight_files:
        log.warning("No sim_weights_*.npz files found in %s", weights_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data" / "dude"

    for wf in weight_files:
        label = wf.stem.removeprefix("sim_weights_")
        log.info("=== DUD-E Evaluating %s ===", label)
        fp_name, fp_size = _parse_fp_config(label)

        weights = load_similarity_weights_npz(wf)

        rng = np.random.default_rng(seed)
        all_results: list[pd.DataFrame] = []
        all_roc: dict[str, list[dict]] = {}

        for target in DUDE_TARGETS:
            target_dir = download_dude_target(target, data_dir)

            active_smiles, decoy_smiles = load_dude_target(target_dir)
            if not active_smiles or not decoy_smiles:
                log.warning("Skipping %s: no actives or no decoys", target)
                continue

            combined = active_smiles + decoy_smiles
            n_actives_orig = len(active_smiles)

            fps, valid_mask = fingerprint_smiles(combined, fp_name, fp_size)

            # Build active/decoy index arrays from concatenation order + valid_mask
            valid_indices = np.where(valid_mask)[0]
            actives_fp: list[int] = []
            decoys_fp: list[int] = []
            fp_i = 0
            for orig_i in valid_indices:
                if orig_i < n_actives_orig:
                    actives_fp.append(fp_i)
                else:
                    decoys_fp.append(fp_i)
                fp_i += 1

            actives_idx = np.array(actives_fp, dtype=np.intp)
            decoys_idx = np.array(decoys_fp, dtype=np.intp)

            if len(actives_idx) == 0 or len(decoys_idx) == 0:
                log.warning("Skipping %s: no valid actives or decoys after parsing", target)
                continue

            log.info(
                "  %s: %d actives, %d decoys",
                target,
                len(actives_idx),
                len(decoys_idx),
            )
            result, target_roc = evaluate_target(
                fps, actives_idx, decoys_idx, weights, num_queries, rng
            )
            result["target"] = target
            all_roc[target] = target_roc
            all_results.append(result)

        if not all_results:
            log.warning("No targets evaluated for %s", label)
            continue

        aggregate_and_save(all_results, all_roc, label, output_dir, "dude")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate similarity metrics on DUD-E virtual screening benchmark"
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
    run_dude_evaluation(args.weights_dir, args.output_dir, args.num_queries, args.seed)


if __name__ == "__main__":
    main()
