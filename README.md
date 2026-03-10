# molecular-fingerprint-bucket-counts

Are molecular fingerprint buckets uniform? This pipeline downloads all ~116M InChI entries from PubChem, normalizes molecules with RDKit, computes multiple fingerprints via scikit-fingerprints, and counts how many molecules activate each bit position. The output is per-fingerprint CSVs and histograms showing the distribution across buckets.

## Setup

```
uv sync
```

## Usage

```
# Full run (downloads ~6.8 GB from PubChem, processes all ~116M molecules)
uv run fp-bucket-counts

# Quick test with 1000 molecules
uv run fp-bucket-counts --limit 1000
```

Results (CSVs and PNGs) are written to `output/`.

## Tests

```
uv run pytest tests/ -v
```
