import numpy as np
import pytest
from rdkit.Chem import MolFromSmiles

from fp_bucket_counts.fingerprint import (
    FINGERPRINT_REGISTRY,
    compute_fingerprints,
    create_fingerprinter,
    get_fp_size,
)


@pytest.fixture
def sample_mols():
    smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    return [MolFromSmiles(s) for s in smiles_list]


class TestCreateFingerprinter:
    def test_ecfp_default(self):
        fpr = create_fingerprinter("ECFP")
        assert get_fp_size(fpr) == 2048

    def test_ecfp_custom_size(self):
        fpr = create_fingerprinter("ECFP", fp_size=1024)
        assert get_fp_size(fpr) == 1024

    def test_maccs_fixed_size(self):
        fpr = create_fingerprinter("MACCS")
        assert get_fp_size(fpr) == 166

    def test_maccs_rejects_fp_size(self):
        with pytest.raises(ValueError, match="fixed size"):
            create_fingerprinter("MACCS", fp_size=256)

    def test_unknown_fingerprint(self):
        with pytest.raises(ValueError, match="Unknown fingerprint"):
            create_fingerprinter("NonExistent")

    def test_all_registry_entries_instantiate(self):
        for name in FINGERPRINT_REGISTRY:
            fpr = create_fingerprinter(name)
            assert get_fp_size(fpr) > 0


class TestComputeFingerprints:
    def test_ecfp_output_shape(self, sample_mols):
        fpr = create_fingerprinter("ECFP", fp_size=2048)
        fps = compute_fingerprints(fpr, sample_mols)
        assert fps.shape == (4, 2048)
        assert fps.dtype == np.uint8

    def test_maccs_output_shape(self, sample_mols):
        fpr = create_fingerprinter("MACCS")
        fps = compute_fingerprints(fpr, sample_mols)
        assert fps.shape == (4, 166)

    def test_fingerprints_are_binary(self, sample_mols):
        fpr = create_fingerprinter("ECFP")
        fps = compute_fingerprints(fpr, sample_mols)
        assert set(np.unique(fps)).issubset({0, 1})

    def test_different_molecules_different_fps(self, sample_mols):
        fpr = create_fingerprinter("ECFP")
        fps = compute_fingerprints(fpr, sample_mols)
        # ethanol and benzene should differ
        assert not np.array_equal(fps[0], fps[1])
