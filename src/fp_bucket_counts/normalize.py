from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

from rdkit.Chem import MolFromInchi  # noqa: E402
from rdkit.Chem.MolStandardize.rdMolStandardize import (  # noqa: E402
    LargestFragmentChooser,
    MetalDisconnector,
    Normalizer,
    Reionizer,
    Uncharger,
)

if TYPE_CHECKING:
    from rdkit.Chem import Mol

log = logging.getLogger(__name__)


class MolNormalizer:
    def __init__(self) -> None:
        self._disconnector = MetalDisconnector()
        self._normalizer = Normalizer()
        self._reionizer = Reionizer()
        self._chooser = LargestFragmentChooser(preferOrganic=True)
        self._uncharger = Uncharger(canonicalOrder=True)

    def normalize(self, inchi: str) -> Mol | None:
        try:
            mol = MolFromInchi(inchi, sanitize=True, removeHs=True)
            if mol is None:
                return None
            mol = self._disconnector.Disconnect(mol)
            mol = self._normalizer.normalize(mol)
            mol = self._reionizer.reionize(mol)
            mol = self._chooser.choose(mol)
            mol = self._uncharger.uncharge(mol)
            return mol
        except Exception:
            log.debug("Failed to normalize InChI: %s", inchi[:80])
            return None

    def normalize_batch(self, inchis: Sequence[str]) -> list[Mol]:
        results = []
        for inchi in inchis:
            mol = self.normalize(inchi)
            if mol is not None:
                results.append(mol)
        return results


# --- Fused normalize + fingerprint worker infrastructure ---

_fused_normalizer: MolNormalizer | None = None
_fused_fingerprinters: list | None = None
_cooc_accumulators: list[np.ndarray | None] | None = None
_cooc_tmp_dir: str | None = None


def _save_cooc_accumulators() -> None:
    """atexit handler: save per-worker co-occurrence accumulators to disk."""
    if _cooc_accumulators is None or _cooc_tmp_dir is None:
        return
    import os

    if not os.path.isdir(_cooc_tmp_dir):
        return

    pid = os.getpid()
    for i, acc in enumerate(_cooc_accumulators):
        if acc is None:
            continue
        np.save(f"{_cooc_tmp_dir}/cooc_{pid}_{i}.npy", acc)


def _init_fused_worker(
    fp_configs: list[dict],
    cooc_tmp_dir: str | None = None,
    cooc_enabled: Sequence[bool] | None = None,
) -> None:
    global _fused_normalizer, _fused_fingerprinters, _cooc_accumulators, _cooc_tmp_dir
    from .fingerprint import create_fingerprinter, get_fp_size

    _fused_normalizer = MolNormalizer()
    _fused_fingerprinters = []
    for fp_conf in fp_configs:
        name = fp_conf["name"]
        fp_size = fp_conf.get("fp_size")
        extra = {k: v for k, v in fp_conf.items() if k not in ("name", "fp_size")}
        _fused_fingerprinters.append(create_fingerprinter(name, fp_size=fp_size, **extra))

    if cooc_tmp_dir is not None:
        import atexit

        _cooc_tmp_dir = cooc_tmp_dir
        enabled_flags = (
            list(cooc_enabled) if cooc_enabled is not None else [True] * len(_fused_fingerprinters)
        )
        if len(enabled_flags) != len(_fused_fingerprinters):
            raise ValueError("cooc_enabled must match the number of fingerprint configs")
        _cooc_accumulators = [
            np.zeros((get_fp_size(fpr), get_fp_size(fpr)), dtype=np.uint32) if enabled else None
            for enabled, fpr in zip(enabled_flags, _fused_fingerprinters)
        ]
        atexit.register(_save_cooc_accumulators)
    else:
        _cooc_accumulators = None
        _cooc_tmp_dir = None


def _normalize_and_count_batch(inchi_batch: Sequence[str]) -> tuple[list[np.ndarray], list[int]]:
    from .fingerprint import compute_fingerprints, get_fp_size

    assert _fused_normalizer is not None
    assert _fused_fingerprinters is not None

    mols = _fused_normalizer.normalize_batch(inchi_batch)
    n_valid = len(mols)

    if n_valid == 0:
        partials = [np.zeros(get_fp_size(fpr), dtype=np.uint64) for fpr in _fused_fingerprinters]
        return partials, [0] * len(_fused_fingerprinters)

    partials = []
    counts = []
    for i, fpr in enumerate(_fused_fingerprinters):
        try:
            fps = compute_fingerprints(fpr, mols)
            partials.append(fps.astype(np.uint64).sum(axis=0))
            counts.append(n_valid)
            accumulator = _cooc_accumulators[i] if _cooc_accumulators is not None else None
            if accumulator is not None:
                fps32 = fps.astype(np.float32)
                accumulator += (fps32.T @ fps32).astype(np.uint32)
        except Exception:
            partials.append(np.zeros(get_fp_size(fpr), dtype=np.uint64))
            counts.append(0)

    return partials, counts
