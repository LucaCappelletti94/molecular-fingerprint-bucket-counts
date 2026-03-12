from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fp_bucket_counts.eval_dude import (
    extract_dude,
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
# extract_dude
# ------------------------------------------------------------------
class TestExtractDude:
    def _make_tar(self, tmp_path: Path) -> Path:
        """Create a minimal DUD-E-like tar.gz with one target."""
        tar_path = tmp_path / "all.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for name, content in [
                ("all/aa2ar/actives_final.ism", "CCO act1\nCCN act2\n"),
                ("all/aa2ar/decoys_final.ism", "CCC dec1\nCCCC dec2\n"),
            ]:
                data = content.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return tar_path

    def test_extracts_to_target_dirs(self, tmp_path):
        tar_path = self._make_tar(tmp_path)
        extract_dir = extract_dude(tar_path, tmp_path)

        assert extract_dir == tmp_path / "all"
        assert (extract_dir / "aa2ar" / "actives_final.ism").exists()
        assert (extract_dir / "aa2ar" / "decoys_final.ism").exists()

    def test_sentinel_skip(self, tmp_path):
        tar_path = self._make_tar(tmp_path)
        # First extraction
        extract_dude(tar_path, tmp_path)
        # Remove tar to prove it's not re-read
        tar_path.unlink()
        # Should skip because sentinel exists
        extract_dir = extract_dude(tar_path, tmp_path)
        assert extract_dir == tmp_path / "all"


# ------------------------------------------------------------------
# run_dude_evaluation smoke test
# ------------------------------------------------------------------
class TestRunDudeEvaluationSmoke:
    def test_two_synthetic_targets(self, tmp_path):
        # Create synthetic DUD-E directory structure
        dude_data = tmp_path / "data" / "dude"
        dude_data.mkdir(parents=True)
        extract_dir = dude_data / "all"

        for target in ["aa2ar", "abl1"]:
            tdir = extract_dir / target
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

        # Mock download/extract to use our synthetic data
        with (
            patch(
                "fp_bucket_counts.eval_dude.download_dude",
                return_value=dude_data / "all.tar.gz",
            ),
            patch(
                "fp_bucket_counts.eval_dude.extract_dude",
                return_value=extract_dir,
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
