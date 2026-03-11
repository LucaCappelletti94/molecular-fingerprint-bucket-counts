from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .cli import FP_CONFIGS, run_pipeline
from .eval_muv import run_muv_evaluation
from .fingerprint import config_label
from .sim_cli import run_sim_weights

OUTPUT_DIR = Path("output")


def _check_cache(
    output_dir: Path, fp_configs: list[dict[str, Any]]
) -> tuple[bool, list[str], list[str]]:
    """Check whether step-1 outputs already exist for every fingerprint config.

    Returns ``(all_cached, cached_labels, missing_labels)``.
    """
    cached: list[str] = []
    missing: list[str] = []
    for conf in fp_configs:
        label = config_label(conf)
        bit_counts = output_dir / f"bit_counts_{label}.csv"
        cooc = output_dir / f"cooc_{label}.npz"
        if bit_counts.exists() and cooc.exists():
            cached.append(label)
        else:
            missing.append(label)
    return (len(missing) == 0, cached, missing)


def run_experiment(
    *,
    limit: int | None = None,
    output_dir: Path = OUTPUT_DIR,
    shrinkage: float = 0.1,
    num_queries: int = 5,
    seed: int = 42,
    force: bool = False,
    skip_eval: bool = False,
) -> None:
    log = logging.getLogger(__name__)
    data_dir = output_dir / "data"

    # --- Step 1: fingerprint pipeline (expensive) ---
    all_cached, cached_labels, missing_labels = _check_cache(output_dir, FP_CONFIGS)

    if force:
        log.info("--force: re-running step 1 (fingerprint pipeline)")
        run_pipeline(limit=limit, output_dir=output_dir, data_dir=data_dir)
    elif all_cached:
        log.info(
            "Step 1 cached — skipping pipeline (%d labels: %s)",
            len(cached_labels),
            ", ".join(cached_labels),
        )
    else:
        log.info(
            "Step 1 cache miss (%d missing: %s) — running pipeline",
            len(missing_labels),
            ", ".join(missing_labels),
        )
        run_pipeline(limit=limit, output_dir=output_dir, data_dir=data_dir)

    # --- Step 2: similarity weights (fast) ---
    log.info("Step 2: deriving similarity weights")
    run_sim_weights(output_dir, shrinkage=shrinkage)

    # --- Step 3: MUV evaluation (moderate) ---
    if skip_eval:
        log.info("Step 3 skipped (--skip-eval)")
    else:
        log.info("Step 3: MUV virtual screening evaluation")
        run_muv_evaluation(output_dir, output_dir, num_queries, seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full fingerprint experiment pipeline")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N unique molecules (step 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Shared output directory (default: output/)",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.1,
        help="Shrinkage parameter for precision matrix (default: 0.1)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of active queries per MUV target (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MUV query selection (default: 42)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run step 1 even if cached outputs exist",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip MUV evaluation (steps 1+2 only)",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    run_experiment(
        limit=args.limit,
        output_dir=args.output_dir,
        shrinkage=args.shrinkage,
        num_queries=args.num_queries,
        seed=args.seed,
        force=args.force,
        skip_eval=args.skip_eval,
    )


if __name__ == "__main__":
    main()
