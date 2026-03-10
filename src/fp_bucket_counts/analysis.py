from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_counts_csv(
    bit_counts: np.ndarray,
    path: Path,
    total_molecules: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bit_position", "count", "fraction"])
        for i, count in enumerate(bit_counts):
            fraction = int(count) / total_molecules if total_molecules > 0 else 0.0
            writer.writerow([i, int(count), f"{fraction:.8f}"])


def plot_histogram(
    bit_counts: np.ndarray,
    path: Path,
    total_molecules: int,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.bar(range(len(bit_counts)), bit_counts, width=1.0, edgecolor="none")
    ax1.set_xlabel("Bit position")
    ax1.set_ylabel("Count")
    ax1.set_title(f"{label}: count per bit position")
    ax1.set_xlim(-0.5, len(bit_counts) - 0.5)

    ax2.hist(bit_counts, bins=50, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Count")
    ax2.set_ylabel("Number of bit positions")
    ax2.set_title(f"{label}: distribution of bit counts")

    fig.suptitle(f"Total molecules: {total_molecules:,}", fontsize=10, y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(
    bit_counts: np.ndarray,
    total_molecules: int,
    label: str,
) -> None:
    counts = bit_counts.astype(np.float64)
    mean = np.mean(counts)
    std = np.std(counts)
    cv = std / mean if mean > 0 else float("inf")

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Total molecules:  {total_molecules:>12,}")
    print(f"  Min count:        {int(np.min(bit_counts)):>12,}")
    print(f"  Max count:        {int(np.max(bit_counts)):>12,}")
    print(f"  Mean count:       {mean:>15,.1f}")
    print(f"  Median count:     {np.median(counts):>15,.1f}")
    print(f"  Std deviation:    {std:>15,.1f}")
    print(f"  CV (std/mean):    {cv:>15.4f}")
    print(f"  Zero bits:        {int(np.sum(bit_counts == 0)):>12,}")
    print(f"  Full bits:        {int(np.sum(bit_counts == total_molecules)):>12,}")
    print(f"{'=' * 60}")
