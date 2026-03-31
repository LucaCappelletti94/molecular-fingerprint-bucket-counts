from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fp_bucket_counts.eval_dude import (
    download_dude_target,
    load_dude_target,
    load_ism,
    run_dude_evaluation,
)


# ------------------------------------------------------------------
# load_ism
# ------------------------------------------------------------------
class TestLoadIsm:
    def test_smiles_only_lines(self, tmp_path):
        ism = tmp_path / "test.ism"
        ism.write_text("CCO\nCCN\nCCC\n")
        result = load_ism(ism)
        assert result == ["CCO", "CCN", "CCC"]

    def test_smiles_with_id(self, tmp_path):
        ism = tmp_path / "test.ism"
        ism.write_text("CCO mol1\nCCN mol2\n")
        result = load_ism(ism)
        assert result == ["CCO", "CCN"]

    def test_blank_lines_skipped(self, tmp_path):
        ism = tmp_path / "test.ism"
        ism.write_text("CCO\n\n\nCCN\n")
        result = load_ism(ism)
        assert result == ["CCO", "CCN"]


# ------------------------------------------------------------------
# load_dude_target
# ------------------------------------------------------------------
class TestLoadDudeTarget:
    def test_loads_actives_and_decoys(self, tmp_path):
        target_dir = tmp_path / "aa2ar"
        target_dir.mkdir()
        (target_dir / "actives_final.ism").write_text("CCO act1\nCCN act2\n")
        (target_dir / "decoys_final.ism").write_text("CCC dec1\nCCCO dec2\nCCCC dec3\n")

        actives, decoys = load_dude_target(target_dir)
        assert actives == ["CCO", "CCN"]
        assert decoys == ["CCC", "CCCO", "CCCC"]


# ------------------------------------------------------------------
# download_dude_target
# ------------------------------------------------------------------
class TestDownloadDudeTarget:
    def test_skips_if_cached(self, tmp_path):
        target_dir = tmp_path / "aa2ar"
        target_dir.mkdir(parents=True)
        (target_dir / "actives_final.ism").write_text("CCO\n")
        (target_dir / "decoys_final.ism").write_text("CCC\n")

        with patch("fp_bucket_counts.download.download_file") as mock_dl:
            result = download_dude_target("aa2ar", tmp_path)

        mock_dl.assert_not_called()
        assert result == target_dir

    def test_downloads_missing_files(self, tmp_path):
        with patch("fp_bucket_counts.download.download_file") as mock_dl:
            result = download_dude_target("aa2ar", tmp_path)

        assert mock_dl.call_count == 2
        assert result == tmp_path / "aa2ar"
        # Check correct URLs
        calls = [c.args for c in mock_dl.call_args_list]
        assert calls[0][0] == "https://dude.docking.org/targets/aa2ar/actives_final.ism"
        assert calls[1][0] == "https://dude.docking.org/targets/aa2ar/decoys_final.ism"


# ------------------------------------------------------------------
# run_dude_evaluation smoke test
# ------------------------------------------------------------------
class TestRunDudeEvaluationSmoke:
    def test_two_synthetic_targets(self, tmp_path):
        # Create synthetic DUD-E directory structure
        dude_data = tmp_path / "data" / "dude"

        for target in ["aa2ar", "abl1"]:
            tdir = dude_data / target
            tdir.mkdir(parents=True)
            (tdir / "actives_final.ism").write_text("CCO\nCCN\nCCC\nc1ccccc1\nCC(=O)O\n")
            (tdir / "decoys_final.ism").write_text(
                "\n".join(f"C{'C' * i}O" for i in range(20)) + "\n"
            )

        # Create a mock weight file
        fp_size = 64
        np.savez(
            tmp_path / "sim_weights_ECFP_fp_size64.npz",
            idf=np.ones(fp_size),
            entropy=np.ones(fp_size),
            diagonal_precision=np.ones(fp_size),
            has_precision=False,
            shrinkage=0.1,
            total_molecules=100,
        )

        # Mock download to return pre-created dirs
        def fake_download(target, data_dir):
            return data_dir / target

        with (
            patch(
                "fp_bucket_counts.eval_dude.download_dude_target",
                side_effect=fake_download,
            ),
            patch(
                "fp_bucket_counts.eval_dude.DUDE_TARGETS",
                ["aa2ar", "abl1"],
            ),
        ):
            run_dude_evaluation(tmp_path, tmp_path, num_queries=2, seed=42)

        # Verify output files
        assert (tmp_path / "eval_dude_ECFP_fp_size64.csv").exists()
        assert (tmp_path / "eval_dude_summary_ECFP_fp_size64.csv").exists()
        assert (tmp_path / "roc_curves_dude_ECFP_fp_size64.npz").exists()

        # Check contents
        per_target = pd.read_csv(tmp_path / "eval_dude_ECFP_fp_size64.csv")
        assert set(per_target["target"].unique()) == {"aa2ar", "abl1"}

        summary = pd.read_csv(tmp_path / "eval_dude_summary_ECFP_fp_size64.csv")
        assert len(summary) > 0
        assert "mean_auc" in summary.columns
