from __future__ import annotations

import logging
import os
import struct
import time
import urllib.request

log = logging.getLogger(__name__)

NTFY_BASE_URL = "https://ntfy.sh"


def _uuid7() -> str:
    """Generate a UUIDv7 (timestamp + random) and return as a hex string."""
    timestamp_ms = int(time.time() * 1000)
    rand_bytes = os.urandom(10)

    # 48-bit timestamp
    ts_bytes = struct.pack(">Q", timestamp_ms)[2:]  # last 6 bytes

    # rand_a (12 bits) from rand_bytes[0:2], with version nibble
    rand_a = struct.unpack(">H", rand_bytes[0:2])[0] & 0x0FFF
    rand_a_with_ver = (0x7 << 12) | rand_a

    # rand_b (62 bits) from rand_bytes[2:10], with variant bits
    rand_b = struct.unpack(">Q", rand_bytes[2:10])[0]
    rand_b = (rand_b & 0x3FFFFFFFFFFFFFFF) | (0b10 << 62)

    uuid_bytes = ts_bytes + struct.pack(">H", rand_a_with_ver) + struct.pack(">Q", rand_b)
    h = uuid_bytes.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def generate_topic() -> str:
    return _uuid7()


def notify(topic: str, message: str, *, title: str | None = None, tags: str | None = None) -> None:
    """Send a notification to ntfy.sh. Failures are logged but never raise."""
    url = f"{NTFY_BASE_URL}/{topic}"
    try:
        req = urllib.request.Request(url, data=message.encode(), method="POST")
        if title:
            req.add_header("Title", title)
        if tags:
            req.add_header("Tags", tags)
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        log.debug("Failed to send ntfy notification to %s", url, exc_info=True)
