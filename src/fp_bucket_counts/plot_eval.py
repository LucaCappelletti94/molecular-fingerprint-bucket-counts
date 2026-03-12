from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eval_common import load_roc_curves_npz

log = logging.getLogger(__name__)


def _target_name_restorer(label: str):
    if label.startswith("muv_"):
        return lambda target: target.replace("MUV_", "MUV-", 1)
    return None


def plot_auc_bar_chart(
    summary_paths: list[Path], output_path: Path, benchmark_id: str = "muv"
) -> None:
    """Grouped bar chart of mean AUC (with std error bars) across FP configs."""
    prefix = f"eval_{benchmark_id}_summary_"
    frames: list[tuple[str, pd.DataFrame]] = []
    for p in sorted(summary_paths):
        label = p.stem.removeprefix(prefix)
        df = pd.read_csv(p)
        frames.append((label, df))

    if not frames:
        log.warning("No summary CSVs to plot")
        return

    metrics = sorted(frames[0][1]["metric"].unique())
    n_metrics = len(metrics)
    n_groups = len(frames)
    x = np.arange(n_metrics)
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, df) in enumerate(frames):
        means = [float(df.loc[df["metric"] == m, "mean_auc"].iloc[0]) for m in metrics]
        stds = [float(df.loc[df["metric"] == m, "std_auc"].iloc[0]) for m in metrics]
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, label=label, capsize=3)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("Mean AUC")
    ax.set_title(f"{benchmark_id.upper()} Virtual Screening: Mean AUC by Metric and FP Config")
    ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info("Wrote AUC bar chart: %s", output_path)


def plot_roc_curves(roc_npz_path: Path, output_dir: Path, label: str) -> None:
    """Per-target ROC curve plots, averaging across queries per metric."""
    roc_data = load_roc_curves_npz(
        roc_npz_path,
        unsanitize_target=_target_name_restorer(label),
    )
    if not roc_data:
        log.warning("No ROC data in %s", roc_npz_path)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    common_fpr = np.linspace(0, 1, 200)

    for target, curves in sorted(roc_data.items()):
        # Group curves by metric
        by_metric: dict[str, list[dict]] = {}
        for entry in curves:
            by_metric.setdefault(entry["metric"], []).append(entry)

        fig, ax = plt.subplots(figsize=(7, 6))
        for metric in sorted(by_metric):
            entries = by_metric[metric]
            interp_tprs = []
            for e in entries:
                interp_tpr = np.interp(common_fpr, e["fpr"], e["tpr"])
                interp_tpr[0] = 0.0
                interp_tprs.append(interp_tpr)
            mean_tpr = np.mean(interp_tprs, axis=0)
            mean_tpr[-1] = 1.0
            auc_val = float(np.trapezoid(mean_tpr, common_fpr))
            ax.plot(common_fpr, mean_tpr, linewidth=1.5, label=f"{metric} (AUC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {target} [{label}]")
        ax.legend(fontsize=7, loc="lower right")

        fig.tight_layout()
        safe_target = target.replace("-", "_")
        out_path = output_dir / f"roc_{label}_{safe_target}.svg"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info("Wrote ROC plot: %s", out_path)


def plot_all_eval(output_dir: Path) -> None:
    """Discover evaluation outputs and generate all plots."""
    # Bar charts from summary CSVs — one per benchmark
    for benchmark_id in ("muv", "dude"):
        summary_paths = sorted(output_dir.glob(f"eval_{benchmark_id}_summary_*.csv"))
        if summary_paths:
            plot_auc_bar_chart(
                summary_paths,
                output_dir / f"eval_{benchmark_id}_auc_bar.svg",
                benchmark_id,
            )

    # ROC curves from NPZ files
    roc_files = sorted(output_dir.glob("roc_curves_*.npz"))
    for roc_path in roc_files:
        label = roc_path.stem.removeprefix("roc_curves_")
        plot_roc_curves(roc_path, output_dir, label)
    if not roc_files:
        log.warning("No roc_curves_*.npz files found in %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots (AUC bar chart, ROC curves)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing evaluation outputs (default: output/)",
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    plot_all_eval(args.output_dir)


if __name__ == "__main__":
    main()
