from __future__ import annotations

import numpy as np
import pandas as pd

from fp_bucket_counts.eval_common import save_roc_curves_npz
from fp_bucket_counts.plot_eval import plot_all_eval, plot_auc_bar_chart, plot_roc_curves


def _write_summary_csvs(tmp_path, benchmark_id: str, labels: list[str]) -> None:
    metrics = [
        "tanimoto",
        "idf_tanimoto",
        "entropy_hamming",
        "diagonal_mahalanobis",
        "full_mahalanobis",
    ]
    for label in labels:
        df = pd.DataFrame(
            {
                "metric": metrics,
                "mean_auc": [0.6, 0.62, 0.55, 0.58, 0.59],
                "std_auc": [0.05, 0.04, 0.06, 0.05, 0.04],
                "mean_ef1": [2.0, 2.1, 1.5, 1.8, 1.9],
                "std_ef1": [0.5, 0.4, 0.6, 0.5, 0.4],
            }
        )
        df.to_csv(tmp_path / f"eval_{benchmark_id}_summary_{label}.csv", index=False)


class TestPlotAucBarChart:
    def test_creates_svg(self, tmp_path):
        _write_summary_csvs(tmp_path, "muv", ["ECFP_fp_size1024", "ECFP_fp_size2048"])

        out = tmp_path / "auc_bar.svg"
        summary_paths = sorted(tmp_path.glob("eval_muv_summary_*.csv"))
        plot_auc_bar_chart(summary_paths, out, "muv")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_svg_with_dude_benchmark(self, tmp_path):
        _write_summary_csvs(tmp_path, "dude", ["ECFP_fp_size1024"])

        out = tmp_path / "auc_bar_dude.svg"
        summary_paths = sorted(tmp_path.glob("eval_dude_summary_*.csv"))
        plot_auc_bar_chart(summary_paths, out, "dude")
        assert out.exists()
        assert out.stat().st_size > 0


class TestPlotRocCurves:
    def test_creates_per_target_svgs(self, tmp_path):
        roc_data = {}
        for target in ["MUV-466", "MUV-548"]:
            curves = []
            for metric in ["tanimoto", "idf_tanimoto"]:
                for q in range(2):
                    fpr = np.array([0.0, 0.1, 0.5, 1.0])
                    tpr = np.array([0.0, 0.4, 0.8, 1.0])
                    curves.append({"query": q, "metric": metric, "fpr": fpr, "tpr": tpr})
            roc_data[target] = curves

        npz_path = tmp_path / "roc_curves_muv_ECFP_fp_size1024.npz"
        save_roc_curves_npz(npz_path, roc_data)

        plot_roc_curves(npz_path, tmp_path, "muv_ECFP_fp_size1024")

        for target in ["MUV_466", "MUV_548"]:
            svg = tmp_path / f"roc_muv_ECFP_fp_size1024_{target}.svg"
            assert svg.exists(), f"Missing {svg.name}"
            assert svg.stat().st_size > 0


class TestPlotAllEval:
    def test_discovers_both_benchmarks(self, tmp_path):
        # Create summary CSVs for both benchmarks
        _write_summary_csvs(tmp_path, "muv", ["ECFP_fp_size1024"])
        _write_summary_csvs(tmp_path, "dude", ["ECFP_fp_size1024"])

        # Create ROC NPZ for one benchmark
        roc_data = {
            "t1": [
                {
                    "query": 0,
                    "metric": "tanimoto",
                    "fpr": np.array([0.0, 0.5, 1.0]),
                    "tpr": np.array([0.0, 0.8, 1.0]),
                }
            ]
        }
        save_roc_curves_npz(tmp_path / "roc_curves_muv_ECFP_fp_size1024.npz", roc_data)

        plot_all_eval(tmp_path)

        assert (tmp_path / "eval_muv_auc_bar.svg").exists()
        assert (tmp_path / "eval_dude_auc_bar.svg").exists()
        assert (tmp_path / "roc_muv_ECFP_fp_size1024_t1.svg").exists()

    def test_no_roc_files_no_crash(self, tmp_path):
        _write_summary_csvs(tmp_path, "muv", ["ECFP_fp_size1024"])
        plot_all_eval(tmp_path)
        assert (tmp_path / "eval_muv_auc_bar.svg").exists()
