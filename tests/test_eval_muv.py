from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from fp_bucket_counts.eval_muv import (
    _diagonal_mahalanobis_batch,
    _entropy_hamming_batch,
    _tanimoto_batch,
    compute_auc,
    compute_ef,
    compute_roc,
    evaluate_target,
    load_muv,
)


# ------------------------------------------------------------------
# Verify backward-compatible re-exports from eval_muv
# ------------------------------------------------------------------
class TestReExports:
    def test_functions_are_importable(self):
        assert callable(compute_auc)
        assert callable(compute_ef)
        assert callable(compute_roc)
        assert callable(evaluate_target)
        assert callable(_tanimoto_batch)
        assert callable(_entropy_hamming_batch)
        assert callable(_diagonal_mahalanobis_batch)


# ------------------------------------------------------------------
# load_muv with synthetic CSV
# ------------------------------------------------------------------
class TestLoadMuv:
    def test_parse_synthetic_csv(self, tmp_path):
        csv_content = textwrap.dedent("""\
            mol_id,smiles,MUV-466,MUV-548
            1,CCO,1,0
            2,CCN,,1
            3,CCC,0,0
            4,,0,1
        """)
        csv_path = tmp_path / "muv.csv"
        csv_path.write_text(csv_content)

        df = load_muv(csv_path)
        # Row 4 has no SMILES, should be dropped
        assert len(df) == 3
        assert "smiles" in df.columns
        assert "MUV-466" in df.columns
        assert "MUV-548" in df.columns
        assert "mol_id" not in df.columns
        assert df.iloc[0]["smiles"] == "CCO"
