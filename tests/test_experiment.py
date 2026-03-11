from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fp_bucket_counts.experiment import _check_cache, run_experiment

FP_CONFIGS_FIXTURE = [
    {"name": "ECFP", "fp_size": 1024},
    {"name": "ECFP", "fp_size": 2048},
]


def _populate_cache(output_dir: Path, labels: list[str]) -> None:
    """Create dummy bit_counts + cooc files for the given labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (output_dir / f"bit_counts_{label}.csv").write_text("dummy")
        (output_dir / f"cooc_{label}.npz").write_text("dummy")


class TestCheckCache:
    def test_all_present(self, tmp_path: Path) -> None:
        labels = ["ECFP_fp_size1024", "ECFP_fp_size2048"]
        _populate_cache(tmp_path, labels)

        all_cached, cached, missing = _check_cache(tmp_path, FP_CONFIGS_FIXTURE)
        assert all_cached is True
        assert sorted(cached) == sorted(labels)
        assert missing == []

    def test_partial_missing(self, tmp_path: Path) -> None:
        _populate_cache(tmp_path, ["ECFP_fp_size1024"])

        all_cached, cached, missing = _check_cache(tmp_path, FP_CONFIGS_FIXTURE)
        assert all_cached is False
        assert cached == ["ECFP_fp_size1024"]
        assert missing == ["ECFP_fp_size2048"]

    def test_none_present(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)

        all_cached, cached, missing = _check_cache(tmp_path, FP_CONFIGS_FIXTURE)
        assert all_cached is False
        assert cached == []
        assert sorted(missing) == ["ECFP_fp_size1024", "ECFP_fp_size2048"]

    def test_only_bit_counts_not_enough(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "bit_counts_ECFP_fp_size1024.csv").write_text("dummy")
        # cooc file missing

        all_cached, cached, missing = _check_cache(tmp_path, [{"name": "ECFP", "fp_size": 1024}])
        assert all_cached is False
        assert cached == []
        assert missing == ["ECFP_fp_size1024"]


@patch("fp_bucket_counts.experiment.FP_CONFIGS", FP_CONFIGS_FIXTURE)
class TestRunExperiment:
    def test_skips_step1_when_cached(self, tmp_path: Path) -> None:
        labels = ["ECFP_fp_size1024", "ECFP_fp_size2048"]
        _populate_cache(tmp_path, labels)

        with (
            patch("fp_bucket_counts.experiment.run_pipeline") as mock_pipeline,
            patch("fp_bucket_counts.experiment.run_sim_weights") as mock_sim,
            patch("fp_bucket_counts.experiment.run_muv_evaluation") as mock_eval,
        ):
            run_experiment(output_dir=tmp_path, skip_eval=True)

        mock_pipeline.assert_not_called()
        mock_sim.assert_called_once_with(tmp_path, shrinkage=0.1)
        mock_eval.assert_not_called()

    def test_force_reruns_step1(self, tmp_path: Path) -> None:
        labels = ["ECFP_fp_size1024", "ECFP_fp_size2048"]
        _populate_cache(tmp_path, labels)

        with (
            patch("fp_bucket_counts.experiment.run_pipeline") as mock_pipeline,
            patch("fp_bucket_counts.experiment.run_sim_weights") as mock_sim,
            patch("fp_bucket_counts.experiment.run_muv_evaluation") as mock_eval,
        ):
            run_experiment(output_dir=tmp_path, force=True, skip_eval=True)

        mock_pipeline.assert_called_once_with(
            limit=None, output_dir=tmp_path, data_dir=tmp_path / "data"
        )
        mock_sim.assert_called_once()
        mock_eval.assert_not_called()

    def test_runs_pipeline_on_cache_miss(self, tmp_path: Path) -> None:
        # Only one config cached
        _populate_cache(tmp_path, ["ECFP_fp_size1024"])

        with (
            patch("fp_bucket_counts.experiment.run_pipeline") as mock_pipeline,
            patch("fp_bucket_counts.experiment.run_sim_weights") as mock_sim,
            patch("fp_bucket_counts.experiment.run_muv_evaluation") as mock_eval,
        ):
            run_experiment(output_dir=tmp_path, limit=100, skip_eval=True)

        mock_pipeline.assert_called_once_with(
            limit=100, output_dir=tmp_path, data_dir=tmp_path / "data"
        )
        mock_sim.assert_called_once()
        mock_eval.assert_not_called()

    def test_skip_eval(self, tmp_path: Path) -> None:
        labels = ["ECFP_fp_size1024", "ECFP_fp_size2048"]
        _populate_cache(tmp_path, labels)

        with (
            patch("fp_bucket_counts.experiment.run_pipeline"),
            patch("fp_bucket_counts.experiment.run_sim_weights"),
            patch("fp_bucket_counts.experiment.run_muv_evaluation") as mock_eval,
        ):
            run_experiment(output_dir=tmp_path, skip_eval=True)

        mock_eval.assert_not_called()

    def test_runs_eval_by_default(self, tmp_path: Path) -> None:
        labels = ["ECFP_fp_size1024", "ECFP_fp_size2048"]
        _populate_cache(tmp_path, labels)

        with (
            patch("fp_bucket_counts.experiment.run_pipeline"),
            patch("fp_bucket_counts.experiment.run_sim_weights"),
            patch("fp_bucket_counts.experiment.run_muv_evaluation") as mock_eval,
        ):
            run_experiment(output_dir=tmp_path, num_queries=3, seed=99)

        mock_eval.assert_called_once_with(tmp_path, tmp_path, 3, 99)
