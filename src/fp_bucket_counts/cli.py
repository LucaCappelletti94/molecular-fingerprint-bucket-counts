from __future__ import annotations

import argparse
import itertools
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .analysis import (
    plot_histogram,
    print_summary,
    save_counts_csv,
)
from .download import ensure_data
from .fingerprint import create_fingerprinter, get_fp_size
from .normalize import _init_fused_worker, _normalize_and_count_batch
from .stream import stream_inchi

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

FP_CONFIGS: list[dict[str, Any]] = [
    {"name": "ECFP", "fp_size": 2048},
    {"name": "ECFP", "fp_size": 1024},
    {"name": "AtomPair", "fp_size": 2048},
    {"name": "MACCS"},
]
BATCH_SIZE = 10_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Molecular fingerprint bucket count analysis")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N unique molecules",
    )
    return parser.parse_args(argv)


def run_pipeline(limit: int | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    n_jobs = os.cpu_count() or 1

    gz_path = ensure_data(DATA_DIR)

    fp_labels = []
    for fp_conf in FP_CONFIGS:
        name = fp_conf["name"]
        fp_size = fp_conf.get("fp_size")
        fpr = create_fingerprinter(name, fp_size=fp_size)
        actual_size = get_fp_size(fpr)
        fp_labels.append((name, actual_size))
        log.info("Registered fingerprint: %s (size=%d)", name, actual_size)

    fp_sizes = [size for _, size in fp_labels]

    # Phase 1: Load and deduplicate all InChIs
    inchi_stream = stream_inchi(gz_path, limit=limit)
    inchis = list(tqdm(inchi_stream, total=limit, desc="Loading InChIs", unit="mol"))
    log.info("Loaded %d unique InChIs", len(inchis))

    # Phase 2: Fused normalize + fingerprint in workers, return only count arrays
    inchi_batches = list(
        itertools.batched(
            tqdm(iter(inchis), desc="Chunking InChIs", unit="mol", total=len(inchis)), BATCH_SIZE
        )
    )
    accumulators = [np.zeros(sz, dtype=np.uint64) for sz in fp_sizes]
    total_molecules = 0

    with Pool(n_jobs, initializer=_init_fused_worker, initargs=(FP_CONFIGS,)) as pool:
        for partial_counts, n_valid in tqdm(
            pool.imap_unordered(_normalize_and_count_batch, inchi_batches, chunksize=1),
            total=len(inchi_batches),
            desc="Processing",
            unit="batch",
        ):
            total_molecules += n_valid
            for acc, partial in zip(accumulators, partial_counts):
                acc += partial

    log.info("Processed %d molecules", total_molecules)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for (name, fp_size), acc in tqdm(
        zip(fp_labels, accumulators), desc="Reporting", unit="fingerprint"
    ):
        csv_path = OUTPUT_DIR / f"bit_counts_{name}_{fp_size}.csv"
        png_path = OUTPUT_DIR / f"histogram_{name}_{fp_size}.png"

        save_counts_csv(acc, csv_path, total_molecules)
        plot_histogram(acc, png_path, total_molecules, name, fp_size)
        print_summary(acc, total_molecules, name, fp_size)

        log.info("Saved: %s", csv_path)
        log.info("Saved: %s", png_path)


def main() -> None:
    args = parse_args()
    run_pipeline(limit=args.limit)


if __name__ == "__main__":
    main()
