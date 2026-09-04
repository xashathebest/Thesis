"""Detect and group visually near-duplicate raw images.

The detector uses a DCT perceptual hash (pHash) as the primary filter,
dHash/aHash as independent secondary evidence, and a global structural
similarity score on small grayscale thumbnails. All thresholds are explicit
and recorded in the output. Connected components become indivisible groups
for later dataset splitting; this command never deletes or rewrites images.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RAW_DIR,
    csv_bool,
    is_path_within,
    parse_csv_bool,
    protect_raw_directory,
    read_csv_rows,
    write_csv_atomic,
)


NEAR_DUPLICATE_FIELDS = (
    "near_duplicate_group_id",
    "canonical_image",
    "image_path",
    "image_class",
    "matched_image",
    "matched_class",
    "phash",
    "dhash",
    "ahash",
    "phash_distance",
    "dhash_distance",
    "ahash_distance",
    "ssim",
    "group_size",
    "cross_class_conflict",
    "duplicate_type",
    "status",
    "action",
    "hash_size",
    "max_phash_distance",
    "max_dhash_distance",
    "max_ahash_distance",
    "min_ssim",
    "match_rule",
)


@dataclass(frozen=True)
class PerceptualHashRecord:
    """Perceptual fingerprints and comparison thumbnail for one image."""

    image_path: str
    image_class: str
    file_hash: str
    phash: str
    dhash: str
    ahash: str
    thumbnail: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class MatchEdge:
    """A threshold-qualified similarity relationship between two images."""

    left: str
    right: str
    phash_distance: int
    dhash_distance: int
    ahash_distance: int
    ssim: float

    def other(self, image_path: str) -> str:
        """Return the image at the opposite end of this edge."""

        return self.right if self.left == image_path else self.left


@dataclass(frozen=True)
class NearDuplicateResult:
    """Output rows plus diagnostics that should not be hidden from callers."""

    rows: list[dict[str, object]]
    hashed_images: int
    skipped_images: tuple[str, ...]
    comparisons: int
    hash_candidate_pairs: int
    matched_pairs: int
    group_count: int


class _UnionFind:
    """Small deterministic disjoint-set implementation for grouping matches."""

    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _resample_filter() -> int:
    """Use high-quality downsampling across supported Pillow versions."""

    return Image.Resampling.LANCZOS


def _bits_to_hex(bits: np.ndarray) -> str:
    flattened = np.asarray(bits, dtype=bool).reshape(-1)
    binary = "".join("1" if value else "0" for value in flattened)
    width = math.ceil(len(binary) / 4)
    return f"{int(binary, 2):0{width}x}" if binary else ""


@lru_cache(maxsize=None)
def _dct_basis(size: int) -> np.ndarray:
    """Return an orthonormal DCT-II basis matrix."""

    frequencies = np.arange(size, dtype=np.float64).reshape(-1, 1)
    samples = np.arange(size, dtype=np.float64).reshape(1, -1)
    basis = np.cos(np.pi * (2.0 * samples + 1.0) * frequencies / (2.0 * size))
    basis[0, :] *= 1.0 / math.sqrt(2.0)
    basis *= math.sqrt(2.0 / size)
    return basis


def _perceptual_hash(gray: Image.Image, hash_size: int) -> str:
    dct_size = hash_size * 4
    pixels = np.asarray(
        gray.resize((dct_size, dct_size), _resample_filter()), dtype=np.float64
    )
    basis = _dct_basis(dct_size)
    low_frequencies = (basis @ pixels @ basis.T)[:hash_size, :hash_size]
    non_dc = low_frequencies.reshape(-1)[1:]
    threshold = float(np.median(non_dc)) if non_dc.size else 0.0
    return _bits_to_hex(low_frequencies > threshold)


def _difference_hash(gray: Image.Image, hash_size: int) -> str:
    pixels = np.asarray(
        gray.resize((hash_size + 1, hash_size), _resample_filter()), dtype=np.int16
    )
    return _bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


def _average_hash(gray: Image.Image, hash_size: int) -> str:
    pixels = np.asarray(
        gray.resize((hash_size, hash_size), _resample_filter()), dtype=np.float32
    )
    return _bits_to_hex(pixels > float(pixels.mean()))


def hamming_distance(left_hash: str, right_hash: str) -> int:
    """Calculate bitwise distance between equally sized hexadecimal hashes."""

    if len(left_hash) != len(right_hash):
        raise ValueError("Perceptual hashes must have the same encoded length.")
    return (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()


def global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    """Calculate global structural similarity for equal grayscale thumbnails.

    This follows the luminance/contrast/structure form of SSIM but computes one
    score over the full thumbnail rather than a sliding local window. It is used
    only after perceptual-hash filtering, making the all-pairs stage inexpensive.
    """

    if left.shape != right.shape:
        raise ValueError("SSIM thumbnails must have identical shapes.")
    left_values = left.astype(np.float64, copy=False)
    right_values = right.astype(np.float64, copy=False)
    left_mean = float(left_values.mean())
    right_mean = float(right_values.mean())
    left_centered = left_values - left_mean
    right_centered = right_values - right_mean
    left_variance = float(np.mean(left_centered * left_centered))
    right_variance = float(np.mean(right_centered * right_centered))
    covariance = float(np.mean(left_centered * right_centered))
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * left_mean * right_mean + c1) * (2.0 * covariance + c2)
    denominator = (left_mean**2 + right_mean**2 + c1) * (
        left_variance + right_variance + c2
    )
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def fingerprint_image(
    image_path: Path,
    relative_path: str,
    image_class: str,
    file_hash: str,
    hash_size: int = 8,
    thumbnail_size: int = 64,
) -> PerceptualHashRecord:
    """Compute three perceptual hashes and a small SSIM thumbnail."""

    if hash_size < 2:
        raise ValueError("hash_size must be at least 2.")
    if thumbnail_size < 8:
        raise ValueError("thumbnail_size must be at least 8.")

    with Image.open(image_path) as source:
        oriented = ImageOps.exif_transpose(source)
        gray = oriented.convert("L")
        phash = _perceptual_hash(gray, hash_size)
        dhash = _difference_hash(gray, hash_size)
        ahash = _average_hash(gray, hash_size)
        thumbnail = np.asarray(
            gray.resize((thumbnail_size, thumbnail_size), _resample_filter()),
            dtype=np.float32,
        ).copy()

    return PerceptualHashRecord(
        image_path=relative_path,
        image_class=image_class,
        file_hash=file_hash,
        phash=phash,
        dhash=dhash,
        ahash=ahash,
        thumbnail=thumbnail,
    )


def _fingerprint_inventory(
    raw_dir: Path,
    inventory_rows: Sequence[Mapping[str, str]],
    hash_size: int,
    thumbnail_size: int,
) -> tuple[list[PerceptualHashRecord], tuple[str, ...]]:
    records: list[PerceptualHashRecord] = []
    failures: list[str] = []
    seen_paths: set[str] = set()
    raw_dir = raw_dir.resolve()

    for row in sorted(inventory_rows, key=lambda item: item.get("image_path", "").casefold()):
        relative_path = row.get("image_path", "").strip()
        if not relative_path:
            raise ValueError("Inventory contains a row with an empty image_path.")
        if relative_path in seen_paths:
            raise ValueError(f"Inventory contains the same image_path more than once: {relative_path}")
        seen_paths.add(relative_path)
        if not parse_csv_bool(row.get("valid", "")):
            continue

        absolute_path = (raw_dir / Path(relative_path)).resolve()
        if not is_path_within(absolute_path, raw_dir):
            failures.append(f"{relative_path}: path escapes raw directory")
            continue
        try:
            records.append(
                fingerprint_image(
                    absolute_path,
                    relative_path=relative_path,
                    image_class=row.get("class", "").strip(),
                    file_hash=row.get("hash", "").strip().casefold(),
                    hash_size=hash_size,
                    thumbnail_size=thumbnail_size,
                )
            )
        except (OSError, SyntaxError, ValueError) as error:
            message = " ".join(str(error).splitlines()).strip() or type(error).__name__
            failures.append(f"{relative_path}: {type(error).__name__}: {message}")
    return records, tuple(failures)


def _validate_thresholds(
    hash_size: int,
    max_phash_distance: int,
    max_dhash_distance: int,
    max_ahash_distance: int,
    min_ssim: float,
) -> None:
    bit_count = hash_size * hash_size
    for name, value in (
        ("max_phash_distance", max_phash_distance),
        ("max_dhash_distance", max_dhash_distance),
        ("max_ahash_distance", max_ahash_distance),
    ):
        if not 0 <= value <= bit_count:
            raise ValueError(f"{name} must be between 0 and {bit_count}, got {value}.")
    if not -1.0 <= min_ssim <= 1.0:
        raise ValueError(f"min_ssim must be between -1 and 1, got {min_ssim}.")


def find_near_duplicate_groups(
    raw_dir: Path,
    inventory_rows: Sequence[Mapping[str, str]],
    *,
    hash_size: int = 8,
    thumbnail_size: int = 64,
    max_phash_distance: int = 6,
    max_dhash_distance: int = 8,
    max_ahash_distance: int = 8,
    min_ssim: float = 0.90,
    require_ssim: bool = True,
) -> NearDuplicateResult:
    """Find non-exact perceptual matches and return connected image groups.

    A pair must pass the pHash threshold and at least one secondary hash
    threshold. By default it must also pass the thumbnail SSIM threshold.
    SHA-identical pairs are intentionally left to ``exact_duplicates.csv``.
    """

    _validate_thresholds(
        hash_size,
        max_phash_distance,
        max_dhash_distance,
        max_ahash_distance,
        min_ssim,
    )
    records, failures = _fingerprint_inventory(
        raw_dir, inventory_rows, hash_size, thumbnail_size
    )
    records_by_path = {record.image_path: record for record in records}
    union_find = _UnionFind(list(records_by_path))
    edges: list[MatchEdge] = []
    comparisons = 0
    hash_candidate_pairs = 0

    for left, right in combinations(records, 2):
        comparisons += 1
        if left.file_hash and left.file_hash == right.file_hash:
            continue
        phash_distance = hamming_distance(left.phash, right.phash)
        if phash_distance > max_phash_distance:
            continue
        dhash_distance = hamming_distance(left.dhash, right.dhash)
        ahash_distance = hamming_distance(left.ahash, right.ahash)
        if dhash_distance > max_dhash_distance and ahash_distance > max_ahash_distance:
            continue

        hash_candidate_pairs += 1
        similarity = global_ssim(left.thumbnail, right.thumbnail)
        if require_ssim and similarity < min_ssim:
            continue

        edge = MatchEdge(
            left=left.image_path,
            right=right.image_path,
            phash_distance=phash_distance,
            dhash_distance=dhash_distance,
            ahash_distance=ahash_distance,
            ssim=similarity,
        )
        edges.append(edge)
        union_find.union(left.image_path, right.image_path)

    component_members: dict[str, list[str]] = defaultdict(list)
    for image_path in records_by_path:
        component_members[union_find.find(image_path)].append(image_path)
    groups = [sorted(members, key=lambda value: (value.casefold(), value)) for members in component_members.values() if len(members) > 1]
    groups.sort(key=lambda members: (members[0].casefold(), members[0]))

    incident_edges: dict[str, list[MatchEdge]] = defaultdict(list)
    for edge in edges:
        incident_edges[edge.left].append(edge)
        incident_edges[edge.right].append(edge)

    match_rule = "phash_and_one_secondary_and_ssim" if require_ssim else "phash_and_one_secondary"
    output_rows: list[dict[str, object]] = []
    for group_number, members in enumerate(groups, start=1):
        canonical = members[0]
        classes = {records_by_path[member].image_class for member in members}
        cross_class_conflict = len(classes) > 1
        status = "MANUAL_REVIEW" if cross_class_conflict else "NEAR_DUPLICATE"
        action = "MANUAL_REVIEW" if cross_class_conflict else "KEEP_GROUP_TOGETHER"

        for member in members:
            record = records_by_path[member]
            best_edge: MatchEdge | None = None
            if member != canonical:
                best_edge = min(
                    incident_edges[member],
                    key=lambda edge: (
                        edge.phash_distance,
                        edge.dhash_distance,
                        edge.ahash_distance,
                        -edge.ssim,
                        edge.other(member).casefold(),
                    ),
                )
            matched_image = best_edge.other(member) if best_edge is not None else ""
            matched_class = (
                records_by_path[matched_image].image_class if matched_image else ""
            )
            output_rows.append(
                {
                    "near_duplicate_group_id": f"near_duplicate_group_{group_number:04d}",
                    "canonical_image": canonical,
                    "image_path": member,
                    "image_class": record.image_class,
                    "matched_image": matched_image,
                    "matched_class": matched_class,
                    "phash": record.phash,
                    "dhash": record.dhash,
                    "ahash": record.ahash,
                    "phash_distance": best_edge.phash_distance if best_edge else "",
                    "dhash_distance": best_edge.dhash_distance if best_edge else "",
                    "ahash_distance": best_edge.ahash_distance if best_edge else "",
                    "ssim": f"{best_edge.ssim:.6f}" if best_edge else "",
                    "group_size": len(members),
                    "cross_class_conflict": csv_bool(cross_class_conflict),
                    "duplicate_type": "PERCEPTUAL_HASH_AND_SSIM" if require_ssim else "PERCEPTUAL_HASH",
                    "status": status,
                    "action": action,
                    "hash_size": hash_size,
                    "max_phash_distance": max_phash_distance,
                    "max_dhash_distance": max_dhash_distance,
                    "max_ahash_distance": max_ahash_distance,
                    "min_ssim": f"{min_ssim:.6f}" if require_ssim else "",
                    "match_rule": match_rule,
                }
            )

    return NearDuplicateResult(
        rows=output_rows,
        hashed_images=len(records),
        skipped_images=failures,
        comparisons=comparisons,
        hash_candidate_pairs=hash_candidate_pairs,
        matched_pairs=len(edges),
        group_count=len(groups),
    )


def write_near_duplicates(
    rows: Sequence[Mapping[str, object]], output_path: Path, raw_dir: Path
) -> None:
    """Write near-duplicate groups outside the immutable raw tree."""

    protect_raw_directory(output_path, raw_dir)
    write_csv_atomic(output_path, NEAR_DUPLICATE_FIELDS, rows)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Find visually near-duplicate raw images and form split-lock groups."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "image_inventory.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_DIR / "near_duplicates.csv",
    )
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--thumbnail-size", type=int, default=64)
    parser.add_argument("--max-phash-distance", type=int, default=6)
    parser.add_argument("--max-dhash-distance", type=int, default=8)
    parser.add_argument("--max-ahash-distance", type=int, default=8)
    parser.add_argument("--min-ssim", type=float, default=0.90)
    parser.add_argument(
        "--skip-ssim",
        action="store_true",
        help="Use only perceptual-hash evidence (less conservative)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run near-duplicate detection."""

    args = build_argument_parser().parse_args(argv)
    inventory_rows = read_csv_rows(
        args.inventory,
        required_fields=("image_path", "class", "hash", "valid"),
    )
    result = find_near_duplicate_groups(
        args.raw_dir,
        inventory_rows,
        hash_size=args.hash_size,
        thumbnail_size=args.thumbnail_size,
        max_phash_distance=args.max_phash_distance,
        max_dhash_distance=args.max_dhash_distance,
        max_ahash_distance=args.max_ahash_distance,
        min_ssim=args.min_ssim,
        require_ssim=not args.skip_ssim,
    )
    write_near_duplicates(result.rows, args.output, args.raw_dir)

    print(f"Near-duplicate manifest written: {args.output.resolve()}")
    print(
        f"Hashed images: {result.hashed_images}; comparisons: {result.comparisons}; "
        f"hash candidates: {result.hash_candidate_pairs}; matched pairs: {result.matched_pairs}; "
        f"groups: {result.group_count}"
    )
    if result.skipped_images:
        print(f"Warning: {len(result.skipped_images)} valid inventory row(s) could not be fingerprinted.")
        for failure in result.skipped_images:
            print(f"- {failure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
