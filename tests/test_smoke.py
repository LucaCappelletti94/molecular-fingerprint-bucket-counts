import gzip
from pathlib import Path

import pytest

from fp_bucket_counts import cli
from fp_bucket_counts.cli import run_pipeline

SAMPLE_INCHIS = [
    ("1", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
    ("2", "InChI=1S/CH4/h1H4", "VNWKTOKETHGBQD-UHFFFAOYSA-N"),
    ("3", "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H", "UHOVQNZJYSORNB-UHFFFAOYSA-N"),
    ("4", "InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)", "QTBSBXVTEAMEQO-UHFFFAOYSA-N"),
    ("5", "InChI=1S/C3H8O/c1-2-3-4/h4H,2-3H2,1H3", "DNIAPMSPPWPWGF-UHFFFAOYSA-N"),
    (
        "6",
        "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
        "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
    ),
    (
        "7",
        "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
        "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
    ),
    (
        "8",
        "InChI=1S/C10H15N3O/c1-13(2)8-6-4-5-7-10(8)12-9(3)11-14/h4-7,14H,1-3H3",
        "LYPFDBRUNKHDLR-UHFFFAOYSA-N",
    ),
    ("9", "InChI=1S/H2O/h1H2", "XLYOFNOQVPJJNP-UHFFFAOYSA-N"),
    (
        "10",
        "InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2",
        "WQZGKKKJIJFFOK-UHFFFAOYSA-N",
    ),
]


def _create_test_data(tmpdir: Path) -> Path:
    gz_path = tmpdir / "CID-InChI-Key.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        for cid, inchi, key in SAMPLE_INCHIS:
            f.write(f"{cid}\t{inchi}\t{key}\n")
    return gz_path


@pytest.fixture
def smoke_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _create_test_data(data_dir)

    output_dir = tmp_path / "output"

    monkeypatch.setattr(cli, "DATA_DIR", data_dir)
    monkeypatch.setattr(cli, "OUTPUT_DIR", output_dir)

    return {"data_dir": data_dir, "output_dir": output_dir}


@pytest.mark.timeout(300)
def test_smoke_pipeline(smoke_env):
    run_pipeline(limit=10)

    output_dir = smoke_env["output_dir"]
    assert output_dir.exists()

    from fp_bucket_counts.cli import FP_CONFIGS

    expected_count = len(FP_CONFIGS)
    csv_files = list(output_dir.glob("bit_counts_*.csv"))
    png_files = list(output_dir.glob("histogram_*.png"))
    assert len(csv_files) == expected_count, f"Expected {expected_count} CSV files, got {csv_files}"
    assert len(png_files) == expected_count, f"Expected {expected_count} PNG files, got {png_files}"

    # Check CSV content
    for csv_file in csv_files:
        lines = csv_file.read_text().strip().split("\n")
        header = lines[0]
        assert header == "bit_position,count,fraction"
        assert len(lines) > 1

    # Verify ECFP_fp_size2048 has 2048 rows (plus header)
    ecfp_2048_csv = [f for f in csv_files if f.name == "bit_counts_ECFP_fp_size2048.csv"][0]
    assert len(ecfp_2048_csv.read_text().strip().split("\n")) == 2049

    # Verify ECFP_fp_size1024 has 1024 rows (plus header)
    ecfp_1024_csv = [f for f in csv_files if f.name == "bit_counts_ECFP_fp_size1024.csv"][0]
    assert len(ecfp_1024_csv.read_text().strip().split("\n")) == 1025

    # Verify MACCS has 166 rows (plus header)
    maccs_csv = [f for f in csv_files if f.name == "bit_counts_MACCS.csv"][0]
    maccs_lines = maccs_csv.read_text().strip().split("\n")
    assert len(maccs_lines) == 167
