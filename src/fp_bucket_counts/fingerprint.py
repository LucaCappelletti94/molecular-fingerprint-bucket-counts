from __future__ import annotations

from typing import Any

import numpy as np
from skfp.fingerprints import (
    AtomPairFingerprint,
    AvalonFingerprint,
    ECFPFingerprint,
    KlekotaRothFingerprint,
    LingoFingerprint,
    MACCSFingerprint,
    MAPFingerprint,
    MHFPFingerprint,
    PubChemFingerprint,
    RDKitFingerprint,
    SECFPFingerprint,
    TopologicalTorsionFingerprint,
)

FINGERPRINT_REGISTRY: dict[str, type] = {
    "ECFP": ECFPFingerprint,
    "AtomPair": AtomPairFingerprint,
    "TopologicalTorsion": TopologicalTorsionFingerprint,
    "RDKit": RDKitFingerprint,
    "MHFP": MHFPFingerprint,
    "Avalon": AvalonFingerprint,
    "MAP": MAPFingerprint,
    "SECFP": SECFPFingerprint,
    "Lingo": LingoFingerprint,
    "MACCS": MACCSFingerprint,
    "PubChem": PubChemFingerprint,
    "KlekotaRoth": KlekotaRothFingerprint,
}

FIXED_SIZE_FINGERPRINTS = {"MACCS", "PubChem", "KlekotaRoth"}


def create_fingerprinter(name: str, fp_size: int | None = None, **extra_kwargs: Any) -> object:
    if name not in FINGERPRINT_REGISTRY:
        raise ValueError(f"Unknown fingerprint '{name}'. Available: {sorted(FINGERPRINT_REGISTRY)}")

    cls = FINGERPRINT_REGISTRY[name]
    kwargs: dict = {"n_jobs": 1}

    if name in FIXED_SIZE_FINGERPRINTS:
        if fp_size is not None:
            raise ValueError(f"{name} has a fixed size; do not specify fp_size")
    elif fp_size is not None:
        kwargs["fp_size"] = fp_size

    kwargs.update(extra_kwargs)
    return cls(**kwargs)


def config_label(conf: dict[str, Any]) -> str:
    name = conf["name"]
    parts = [name]
    for key, val in conf.items():
        if key != "name":
            parts.append(f"{key}{val}")
    return "_".join(parts)


def get_fp_size(fingerprinter: object) -> int:
    return fingerprinter.n_features_out  # type: ignore[attr-defined, no-any-return]


def compute_fingerprints(fingerprinter: object, mols: list) -> np.ndarray:
    result = fingerprinter.transform(mols)  # type: ignore[attr-defined]
    return np.asarray(result, dtype=np.uint8)
