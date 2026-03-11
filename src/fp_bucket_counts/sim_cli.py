from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .cooccurrence import load_cooccurrence_npz
from .similarity import (
    diagonal_precision_weights,
    entropy_weights,
    idf_weights,
    precision_matrix,
    save_similarity_weights_npz,
)

OUTPUT_DIR = Path("output")


def load_bit_counts_csv(path: Path) -> tuple[NDArray[np.uint64], int]:
    """Parse a bit_counts CSV produced by ``analysis.save_counts_csv``.

    Returns ``(counts, total_molecules)`` where *counts* is a uint64 array.
    """
    total_molecules = 0
    counts: list[int] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# total_molecules:"):
                total_molecules = int(line.split(":", 1)[1].strip())
            elif line.startswith("bit_position"):
                continue  # header
            elif line:
                parts = line.split(",")
                counts.append(int(parts[1]))
    return np.array(counts, dtype=np.uint64), total_molecules


def run_sim_weights(output_dir: Path, *, shrinkage: float = 0.1) -> None:
    log = logging.getLogger(__name__)

    csv_paths = sorted(output_dir.glob("bit_counts_*.csv"))
    if not csv_paths:
        log.warning("No bit_counts_*.csv files found in %s", output_dir)
        return

    for csv_path in csv_paths:
        label = csv_path.stem.removeprefix("bit_counts_")
        log.info("Processing %s", label)

        bit_counts, total_molecules = load_bit_counts_csv(csv_path)

        idf = idf_weights(bit_counts, total_molecules)
        entropy = entropy_weights(bit_counts, total_molecules)
        diag_prec = diagonal_precision_weights(bit_counts, total_molecules)

        precision = None
        cooc_path = output_dir / f"cooc_{label}.npz"
        try:
            cooc_matrix, _ = load_cooccurrence_npz(cooc_path)
            precision = precision_matrix(cooc_matrix, total_molecules, shrinkage=shrinkage)
        except (FileNotFoundError, ValueError):
            pass

        save_similarity_weights_npz(
            output_dir / f"sim_weights_{label}.npz",
            idf=idf,
            entropy=entropy,
            diagonal_precision=diag_prec,
            precision=precision,
            shrinkage=shrinkage,
            total_molecules=total_molecules,
        )
        log.info("Saved sim_weights_%s.npz", label)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive similarity weights from pipeline output")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory containing pipeline output (default: output/)",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.1,
        help="Shrinkage parameter for precision matrix (default: 0.1)",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    run_sim_weights(args.output_dir, shrinkage=args.shrinkage)


if __name__ == "__main__":
    main()
