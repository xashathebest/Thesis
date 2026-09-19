"""Small startup-only helpers for reproducible local checkpoint metadata."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def shortened_sha256(path: Path | None, *, length: int = 12) -> str | None:
    """Return a short SHA-256 fingerprint without retaining model contents.

    Callers calculate this once after a checkpoint has loaded; it is never used
    in the per-frame inference path.
    """

    if path is None or not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]
