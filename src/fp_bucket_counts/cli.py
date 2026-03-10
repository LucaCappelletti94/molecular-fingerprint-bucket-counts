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
from .fingerprint import config_label, create_fingerprinter, get_fp_size
from .normalize import _init_fused_worker, _normalize_and_count_batch
from .stream import stream_inchi

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

FP_CONFIGS: list[dict[str, Any]] = [
    # --- Hashed fingerprints at standard 2048 ---
    {"name": "ECFP", "fp_size": 2048},
    {"name": "AtomPair", "fp_size": 2048},
    {"name": "TopologicalTorsion", "fp_size": 2048},
    {"name": "RDKit", "fp_size": 2048},
    {"name": "MHFP", "fp_size": 2048},
    {"name": "Avalon", "fp_size": 2048},
    {"name": "MAP", "fp_size": 2048},
    {"name": "SECFP", "fp_size": 2048},
    {"name": "Lingo", "fp_size": 2048},
    # --- Larger sizes ---
    {"name": "ECFP", "fp_size": 4096},
    {"name": "ECFP", "fp_size": 8192},
    {"name": "ECFP", "fp_size": 16384},
    {"name": "AtomPair", "fp_size": 4096},
    {"name": "AtomPair", "fp_size": 8192},
    {"name": "AtomPair", "fp_size": 16384},
    {"name": "TopologicalTorsion", "fp_size": 4096},
    {"name": "TopologicalTorsion", "fp_size": 8192},
    {"name": "TopologicalTorsion", "fp_size": 16384},
    {"name": "RDKit", "fp_size": 4096},
    {"name": "RDKit", "fp_size": 8192},
    {"name": "RDKit", "fp_size": 16384},
    {"name": "MHFP", "fp_size": 4096},
    {"name": "MHFP", "fp_size": 8192},
    {"name": "MHFP", "fp_size": 16384},
    {"name": "Avalon", "fp_size": 4096},
    {"name": "Avalon", "fp_size": 8192},
    {"name": "Avalon", "fp_size": 16384},
    # --- Smaller default sizes ---
    {"name": "ECFP", "fp_size": 1024},
    {"name": "Avalon", "fp_size": 512},
    {"name": "MAP", "fp_size": 1024},
    {"name": "Lingo", "fp_size": 1024},
    # --- Radius variants ---
    {"name": "ECFP", "fp_size": 2048, "radius": 3},
    {"name": "MAP", "fp_size": 2048, "radius": 3},
    {"name": "MAP", "fp_size": 2048, "radius": 4},
    # --- Fixed-size structural keys ---
    {"name": "MACCS"},
    {"name": "PubChem"},
    {"name": "KlekotaRoth"},
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

    fp_entries = []
    for fp_conf in FP_CONFIGS:
        name = fp_conf["name"]
        fp_size = fp_conf.get("fp_size")
        extra = {k: v for k, v in fp_conf.items() if k not in ("name", "fp_size")}
        fpr = create_fingerprinter(name, fp_size=fp_size, **extra)
        actual_size = get_fp_size(fpr)
        label = config_label(fp_conf)
        fp_entries.append((label, actual_size))
        log.info("Registered fingerprint: %s (size=%d)", label, actual_size)

    fp_sizes = [size for _, size in fp_entries]

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
    total_molecules = [0] * len(FP_CONFIGS)

    with Pool(n_jobs, initializer=_init_fused_worker, initargs=(FP_CONFIGS,)) as pool:
        for partial_counts, fp_counts in tqdm(
            pool.imap_unordered(_normalize_and_count_batch, inchi_batches, chunksize=1),
            total=len(inchi_batches),
            desc="Processing",
            unit="batch",
        ):
            for i, (acc, partial) in enumerate(zip(accumulators, partial_counts)):
                acc += partial
                total_molecules[i] += fp_counts[i]

    log.info("Processed molecules (per-fingerprint counts vary)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for (label, _fp_size), acc, mol_count in tqdm(
        zip(fp_entries, accumulators, total_molecules), desc="Reporting", unit="fingerprint"
    ):
        csv_path = OUTPUT_DIR / f"bit_counts_{label}.csv"
        png_path = OUTPUT_DIR / f"histogram_{label}.png"

        save_counts_csv(acc, csv_path, mol_count)
        plot_histogram(acc, png_path, mol_count, label)
        print_summary(acc, mol_count, label)

        log.info("Saved: %s", csv_path)
        log.info("Saved: %s", png_path)


def main() -> None:
    args = parse_args()
    run_pipeline(limit=args.limit)


if __name__ == "__main__":
    main()
