# Molecular Fingerprint Bucket Counts

[![CI](https://github.com/LucaCappelletti94/molecular-fingerprint-bucket-counts/actions/workflows/ci.yml/badge.svg)](https://github.com/LucaCappelletti94/molecular-fingerprint-bucket-counts/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

Are molecular fingerprint buckets uniform? This project downloads PubChem InChIs, normalizes molecules with RDKit, computes fingerprints via scikit-fingerprints, and writes per-fingerprint bucket counts, histograms, co-occurrence matrices, derived similarity weights, and MUV virtual-screening evaluation summaries.

## Outputs

The pipeline writes local artifacts under `output/`, including:

- `bit_counts_*.csv` and `histogram_*.svg` for bucket occupancy.
- `cooc_*.npz`, `cooc_summary_*.csv`, and `cooc_heatmap_*.svg` for bit co-occurrence.
- `sim_weights_*.npz` for IDF, entropy, and precision-derived similarity weights.
- `eval_muv_*.csv` and `eval_muv_summary_*.csv` for MUV screening benchmarks.

## Setup

```bash
uv sync
```

## Usage

```bash
# End-to-end experiment: pipeline, weight derivation, then MUV evaluation
uv run fp-experiment

# Quick local run: pipeline + weights only
uv run fp-experiment --limit 1000 --skip-eval

# Raw pipeline only
uv run fp-bucket-counts --limit 1000

# Derive similarity weights from an existing output directory
uv run fp-sim-weights --output-dir output
```

The raw fingerprint pipeline prints a dedicated `ntfy.sh` topic URL and sends progress notifications for the main milestones. Generated artifacts are treated as local outputs and are ignored by Git.

## Tests

```bash
uv run pytest tests/ -v
```
