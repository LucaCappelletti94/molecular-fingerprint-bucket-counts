from __future__ import annotations

import numpy as np
from skfp.fingerprints import (
    AtomPairFingerprint,
    ECFPFingerprint,
    MACCSFingerprint,
)

FINGERPRINT_REGISTRY: dict[str, type] = {
    "ECFP": ECFPFingerprint,
    "AtomPair": AtomPairFingerprint,
    "MACCS": MACCSFingerprint,
}

FIXED_SIZE_FINGERPRINTS = {"MACCS"}


def create_fingerprinter(name: str, fp_size: int | None = None) -> object:
    if name not in FINGERPRINT_REGISTRY:
        raise ValueError(f"Unknown fingerprint '{name}'. Available: {sorted(FINGERPRINT_REGISTRY)}")

    cls = FINGERPRINT_REGISTRY[name]
    kwargs: dict = {"n_jobs": 1}

    if name in FIXED_SIZE_FINGERPRINTS:
        if fp_size is not None:
            raise ValueError(f"{name} has a fixed size; do not specify fp_size")
    elif fp_size is not None:
        kwargs["fp_size"] = fp_size

    return cls(**kwargs)


def get_fp_size(fingerprinter: object) -> int:
    return fingerprinter.n_features_out  # type: ignore[attr-defined, no-any-return]


def compute_fingerprints(fingerprinter: object, mols: list) -> np.ndarray:
    result = fingerprinter.transform(mols)  # type: ignore[attr-defined]
    return np.asarray(result, dtype=np.uint8)
