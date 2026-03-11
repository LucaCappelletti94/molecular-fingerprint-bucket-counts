from __future__ import annotations

import argparse
import itertools
import logging
import os
import shutil
import tempfile
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
from .cooccurrence import (
    dense_cooccurrence_skip_reason,
    merge_worker_cooccurrence,
    plot_cooccurrence_heatmap,
    plot_skipped_cooccurrence_heatmap,
    save_cooccurrence_npz,
    save_cooccurrence_summary_csv,
    save_skipped_cooccurrence_npz,
    save_skipped_cooccurrence_summary_csv,
    supports_dense_cooccurrence,
)
from .download import ensure_data
from .fingerprint import config_label, create_fingerprinter, get_fp_size
from .normalize import (
    _flush_cooc_to_disk,
    _init_fused_worker,
    _normalize_and_count_batch,
    _save_cooc_accumulators,
)
from .ntfy import generate_topic, notify
from .stream import stream_inchi

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

FP_CONFIGS: list[dict[str, Any]] = [
    # --- ECFP sizes ---
    {"name": "ECFP", "fp_size": 1024},
    {"name": "ECFP", "fp_size": 2048},
    {"name": "ECFP", "fp_size": 4096},
    # # --- Other hashed fingerprints at standard 2048 ---
    # {"name": "AtomPair", "fp_size": 2048},
    # {"name": "TopologicalTorsion", "fp_size": 2048},
    # {"name": "RDKit", "fp_size": 2048},
    # {"name": "MHFP", "fp_size": 2048},
    # {"name": "Avalon", "fp_size": 2048},
    # {"name": "MAP", "fp_size": 2048},
    # {"name": "SECFP", "fp_size": 2048},
    # {"name": "Lingo", "fp_size": 2048},
    # # --- Other larger sizes ---
    # {"name": "AtomPair", "fp_size": 4096},
    # {"name": "TopologicalTorsion", "fp_size": 4096},
    # {"name": "RDKit", "fp_size": 4096},
    # {"name": "MHFP", "fp_size": 4096},
    # {"name": "Avalon", "fp_size": 4096},
    # # --- Other smaller sizes ---
    # {"name": "Avalon", "fp_size": 512},
    # {"name": "Lingo", "fp_size": 1024},
    # # --- Radius variants ---
    # {"name": "ECFP", "fp_size": 2048, "radius": 3},
    # # --- Fixed-size structural keys ---
    # {"name": "MACCS"},
    # {"name": "PubChem"},
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

    topic = generate_topic()
    ntfy_url = f"https://ntfy.sh/{topic}"
    log.info("ntfy topic: %s", ntfy_url)
    print(f"\nNotifications: {ntfy_url}\n")
    notify(topic, f"Pipeline started (limit={limit})", title="Pipeline Started", tags="rocket")

    gz_path = ensure_data(DATA_DIR)
    notify(topic, f"Data ready: {gz_path}", title="Data Downloaded", tags="white_check_mark")

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
    cooc_enabled = [supports_dense_cooccurrence(size) for size in fp_sizes]

    # Compute worker count accounting for co-occurrence memory budget
    per_worker_cooc_bytes = sum(
        sz * sz * 4 for sz, enabled in zip(fp_sizes, cooc_enabled) if enabled
    )  # uint32 matrices
    try:
        import psutil  # type: ignore[import-untyped]

        available_ram = psutil.virtual_memory().available
    except Exception:
        available_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    max_by_ram = (
        max(1, int(available_ram * 0.8 / per_worker_cooc_bytes))
        if per_worker_cooc_bytes > 0
        else os.cpu_count() or 1
    )
    n_jobs = min(os.cpu_count() or 1, max_by_ram)
    log.info(
        "Workers: %d (cooc budget %.1f GB/worker, %.1f GB available)",
        n_jobs,
        per_worker_cooc_bytes / 1e9,
        available_ram / 1e9,
    )
    log.info(
        "Dense co-occurrence enabled for %d/%d fingerprints",
        sum(cooc_enabled),
        len(cooc_enabled),
    )

    # Phase 1: Load and deduplicate all InChIs
    inchi_stream = stream_inchi(gz_path, limit=limit)
    inchis = list(tqdm(inchi_stream, total=limit, desc="Loading InChIs", unit="mol"))
    log.info("Loaded %d unique InChIs", len(inchis))
    notify(topic, f"Loaded {len(inchis):,} unique InChIs", title="InChIs Loaded", tags="test_tube")

    # Phase 2: Fused normalize + fingerprint in workers, return only count arrays
    inchi_batches = list(
        itertools.batched(
            tqdm(iter(inchis), desc="Chunking InChIs", unit="mol", total=len(inchis)), BATCH_SIZE
        )
    )
    accumulators = [np.zeros(sz, dtype=np.uint64) for sz in fp_sizes]
    total_molecules = [0] * len(FP_CONFIGS)

    cooc_tmp_dir = tempfile.mkdtemp(prefix="cooc_")
    use_serial = False

    try:
        pool = Pool(
            n_jobs,
            initializer=_init_fused_worker,
            initargs=(FP_CONFIGS, cooc_tmp_dir, cooc_enabled),
        )
    except PermissionError:
        use_serial = True

    if use_serial:
        log.warning("Multiprocessing unavailable; falling back to serial batch processing")
        _init_fused_worker(FP_CONFIGS, cooc_tmp_dir, cooc_enabled)
        for partial_counts, fp_counts in tqdm(
            map(_normalize_and_count_batch, inchi_batches),
            total=len(inchi_batches),
            desc="Processing",
            unit="batch",
        ):
            for i, (acc, partial) in enumerate(zip(accumulators, partial_counts)):
                acc += partial
                total_molecules[i] += fp_counts[i]
        _save_cooc_accumulators()
    else:
        try:
            batch_results = pool.imap_unordered(
                _normalize_and_count_batch, inchi_batches, chunksize=1
            )
            for partial_counts, fp_counts in tqdm(
                batch_results,
                total=len(inchi_batches),
                desc="Processing",
                unit="batch",
            ):
                for i, (acc, partial) in enumerate(zip(accumulators, partial_counts)):
                    acc += partial
                    total_molecules[i] += fp_counts[i]

            # Flush co-occurrence accumulators from all workers
            saved_pids: set[int] = set()
            while len(saved_pids) < n_jobs:
                pid = pool.apply(_flush_cooc_to_disk)
                saved_pids.add(pid)
            log.info("Flushed co-occurrence accumulators from %d workers", len(saved_pids))
        finally:
            pool.close()
            pool.join()

    log.info("Processed molecules (per-fingerprint counts vary)")
    notify(
        topic,
        f"Fingerprinting complete across {len(FP_CONFIGS)} configurations",
        title="Processing Complete",
        tags="bar_chart",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for (label, _fp_size), acc, mol_count in tqdm(
        zip(fp_entries, accumulators, total_molecules), desc="Reporting", unit="fingerprint"
    ):
        csv_path = OUTPUT_DIR / f"bit_counts_{label}.csv"
        svg_path = OUTPUT_DIR / f"histogram_{label}.svg"

        save_counts_csv(acc, csv_path, mol_count)
        plot_histogram(acc, svg_path, mol_count, label)
        print_summary(acc, mol_count, label)

        log.info("Saved: %s", csv_path)
        log.info("Saved: %s", svg_path)

    # Phase 4: Merge and save co-occurrence matrices
    cooc_tmp_path = Path(cooc_tmp_dir)
    for i, ((label, fp_size), mol_count) in tqdm(
        enumerate(zip(fp_entries, total_molecules)),
        total=len(fp_entries),
        desc="Co-occurrence",
        unit="fingerprint",
    ):
        npz_path = OUTPUT_DIR / f"cooc_{label}.npz"
        summary_path = OUTPUT_DIR / f"cooc_summary_{label}.csv"
        heatmap_path = OUTPUT_DIR / f"cooc_heatmap_{label}.svg"

        if cooc_enabled[i]:
            cooc_matrix = merge_worker_cooccurrence(cooc_tmp_path, i, fp_size)
            save_cooccurrence_npz(cooc_matrix, npz_path, mol_count)
            save_cooccurrence_summary_csv(cooc_matrix, summary_path, mol_count)
            plot_cooccurrence_heatmap(cooc_matrix, heatmap_path, mol_count, label)
        else:
            reason = dense_cooccurrence_skip_reason(fp_size)
            save_skipped_cooccurrence_npz(npz_path, fp_size, mol_count, reason)
            save_skipped_cooccurrence_summary_csv(summary_path, mol_count, reason)
            plot_skipped_cooccurrence_heatmap(heatmap_path, label, fp_size, reason)

        log.info("Saved: %s, %s, %s", npz_path, summary_path, heatmap_path)

    shutil.rmtree(cooc_tmp_dir, ignore_errors=True)

    notify(
        topic,
        f"All done! {len(FP_CONFIGS)} fingerprints reported to {OUTPUT_DIR}/",
        title="Pipeline Complete",
        tags="tada",
    )


def main() -> None:
    args = parse_args()
    run_pipeline(limit=args.limit)


if __name__ == "__main__":
    main()
