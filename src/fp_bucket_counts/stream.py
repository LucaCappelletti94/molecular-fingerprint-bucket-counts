from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path


def stream_inchi(gz_path: Path, *, limit: int | None = None) -> Iterator[str]:
    seen: set[str] = set()
    count = 0

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                return

            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            inchi = parts[1]
            inchi_key = parts[2]

            if not inchi or not inchi_key:
                continue

            if inchi_key in seen:
                continue
            seen.add(inchi_key)

            count += 1
            yield inchi
