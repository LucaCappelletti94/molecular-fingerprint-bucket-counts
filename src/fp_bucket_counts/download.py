from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger(__name__)

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/"
FILENAME = "CID-InChI-Key.gz"


def download_file(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    log.info("Downloading %s -> %s", url, dest)

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with (
            open(tmp, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar,
        ):
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

    tmp.rename(dest)
    log.info("Download complete: %s", dest)


def verify_md5(filepath: Path, expected_md5: str) -> bool:
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            md5.update(chunk)
    actual = md5.hexdigest()
    if actual != expected_md5:
        log.warning("MD5 mismatch: expected %s, got %s", expected_md5, actual)
        return False
    return True


def download_md5(url: str) -> str | None:
    import urllib.request

    md5_url = url + ".md5"
    try:
        with urllib.request.urlopen(md5_url) as response:
            text: str = response.read().decode().strip()
            return text.split()[0]
    except Exception:
        log.warning("Could not download MD5 checksum from %s", md5_url)
        return None


def ensure_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    gz_path = data_dir / FILENAME
    url = BASE_URL + FILENAME

    if gz_path.exists():
        log.info("Data file already exists: %s", gz_path)
        return gz_path

    download_file(url, gz_path)

    expected_md5 = download_md5(url)
    if expected_md5 is not None:
        if not verify_md5(gz_path, expected_md5):
            gz_path.unlink()
            raise RuntimeError(f"MD5 verification failed for {gz_path}")
        log.info("MD5 verified: %s", gz_path)

    return gz_path
