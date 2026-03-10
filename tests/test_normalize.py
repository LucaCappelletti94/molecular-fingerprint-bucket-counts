from rdkit.Chem import MolToSmiles

from fp_bucket_counts.normalize import MolNormalizer


class TestMolNormalizer:
    def setup_method(self):
        self.normalizer = MolNormalizer()

    def test_simple_molecule(self):
        mol = self.normalizer.normalize("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3")
        assert mol is not None
        assert MolToSmiles(mol) == "CCO"

    def test_invalid_inchi_returns_none(self):
        mol = self.normalizer.normalize("not_an_inchi")
        assert mol is None

    def test_empty_string_returns_none(self):
        mol = self.normalizer.normalize("")
        assert mol is None

    def test_normalize_batch_filters_failures(self):
        inchis = [
            "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # ethanol
            "garbage",
            "InChI=1S/CH4/h1H4",  # methane
        ]
        mols = self.normalizer.normalize_batch(inchis)
        assert len(mols) == 2

    def test_normalize_batch_empty(self):
        mols = self.normalizer.normalize_batch([])
        assert mols == []

    def test_metal_disconnection(self):
        # Sodium acetate
        mol = self.normalizer.normalize("InChI=1S/C2H4O2.Na/c1-2(3)4;/h1H3,(H,3,4);/q;+1/p-1")
        assert mol is not None

    def test_largest_fragment(self):
        # Salt: should pick the organic fragment
        mol = self.normalizer.normalize("InChI=1S/C2H4O2.ClH/c1-2(3)4;/h1H3,(H,3,4);1H")
        assert mol is not None
