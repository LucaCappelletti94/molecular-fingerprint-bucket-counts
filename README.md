# molecular-fingerprint-bucket-counts

[![CI](https://github.com/LucaCappelletti94/molecular-fingerprint-bucket-counts/actions/workflows/ci.yml/badge.svg)](https://github.com/LucaCappelletti94/molecular-fingerprint-bucket-counts/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

Are molecular fingerprint buckets uniform? This pipeline downloads all ~116M InChI entries from PubChem, normalizes molecules with RDKit, computes multiple fingerprints via scikit-fingerprints, and counts how many molecules activate each bit position. The output is per-fingerprint CSVs and histograms showing the distribution across buckets.

## Setup

```bash
uv sync
```

## Usage

```bash
# Full run (downloads ~6.8 GB from PubChem, processes all ~116M molecules)
uv run fp-bucket-counts

# Quick test with 1000 molecules
uv run fp-bucket-counts --limit 1000
```

Results (CSVs and SVGs) are written to `output/`.

## Tests

```bash
uv run pytest tests/ -v
```
