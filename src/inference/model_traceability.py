"""Small startup-only helpers for reproducible local checkpoint metadata.

SHA-256 helpers are calculated once after a checkpoint loads.  They are never
used in the per-frame inference path and must not hold model weights in memory.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def _compute_sha256(path: Path) -> str:
    """Stream the file in 1 MiB chunks and return the hex digest."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shortened_sha256(path: Path | None, *, length: int = 12) -> str | None:
    """Return a short SHA-256 fingerprint without retaining model contents.

    Callers calculate this once after a checkpoint has loaded; it is never used
    in the per-frame inference path.
    """

    if path is None or not path.is_file():
        return None
    return _compute_sha256(path)[:length]


def full_sha256(path: Path | None) -> str | None:
    """Return the full 64-character SHA-256 hex digest for a checkpoint.

    Use this when locking a model hash for a thesis validation study.
    Record the result in ``models/MODEL_REGISTRY.json`` so every session export
    can be traced back to the exact checkpoint used.
    """

    if path is None or not path.is_file():
        return None
    return _compute_sha256(path)


def assert_checkpoint_sha256(path: Path | None, expected_sha256: str) -> None:
    """Raise ``RuntimeError`` if the checkpoint does not match ``expected_sha256``.

    Call this from a validation study entry point to guarantee that no
    checkpoint swap can silently change a locked result.  Never call it in the
    live production path — computing a SHA-256 per-frame would break inference.

    Args:
        path: Absolute path to the checkpoint file.
        expected_sha256: The 64-character hex digest from ``MODEL_REGISTRY.json``.

    Raises:
        RuntimeError: When the checkpoint is missing or its hash differs.
    """

    if path is None or not path.is_file():
        raise RuntimeError(
            f"Checkpoint not found for hash validation: {path}. "
            "Set the appropriate MODEL_PATH environment variable."
        )
    actual = _compute_sha256(path)
    if actual != expected_sha256.strip().lower():
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch for {path.name}. "
            f"Expected {expected_sha256[:16]}… got {actual[:16]}…. "
            "Do not use a different checkpoint for a locked validation study."
        )
