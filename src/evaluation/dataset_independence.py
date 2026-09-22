"""Research-only dataset-independence auditing and split proposals.

This module is deliberately separate from the production inspection runtime.  It
does not load a model, alter a dataset, move files, or modify labels.  Instead it
turns a reviewable CSV/JSON manifest into a conservative answer to one question:
can a reported validation or test result be called independent of its training
data?

The auditor joins samples transitively when they share an explicit source group,
original source, augmentation parent, physical specimen, acquisition session,
source folder, filename, or SHA-256 value.  A filename-only match and an optional
perceptual-hash match are deliberately treated as *review blockers*, rather than
as proof that two physical fish are the same.  Missing provenance is also a
blocker.  Consequently, ``PASS`` is intentionally difficult to obtain.

Expected manifest fields are intentionally small and portable.  The preferred
ones are ``sample_id``, ``split``, ``source_group_id``, ``image_path``,
``annotation_path``, ``class_information``, and ``original_source``.  Common
aliases from the existing thesis manifests are accepted as a convenience.  A
separate training-lineage manifest can be supplied for a deployed checkpoint;
every record in it is conservatively considered training exposure.

Example::

    py -m src.evaluation.dataset_independence \
      --manifest future_study_manifest.csv \
      --model-training-manifest training_manifest.csv \
      --output results/independence_audit.json \
      --text-output results/independence_audit.txt \
      --near-duplicates

The optional ``--proposed-split-manifest`` output is only a deterministic,
group-aware *proposal* for human review.  It never materializes, deletes, or
replaces the supplied dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "development": "train",
    "dev": "train",
    "validation": "validation",
    "validate": "validation",
    "valid": "validation",
    "val": "validation",
    "validation_independent": "validation",
    "test": "test",
    "testing": "test",
    "locked_test": "test",
    "test_locked": "test",
}
UNKNOWN_VALUES = {"", "unknown", "unassigned", "none", "null", "n/a", "na", "tbd", "?"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROPOSAL_COLUMNS = (
    "sample_id",
    "source_group_id",
    "image_path",
    "annotation_path",
    "class_information",
    "split",
    "original_source",
)

# Field aliases are intentionally explicit.  The auditor does not guess a
# physical-origin relationship from a timestamp or a filename alone.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "sample_id": ("sample_id", "image_id", "id", "canonical_filename", "filename", "image"),
    "split": ("split", "canonical_split", "dataset_split", "source_export_split"),
    "image_path": ("image_path", "path", "source_path", "file_path", "image", "canonical_filename"),
    "annotation_path": ("annotation_path", "label_path", "labels_path", "annotation", "label"),
    "class_information": ("class_information", "class", "fish_class", "class_ids", "labels"),
    "file_sha256": ("file_sha256", "sha256", "hash", "original_sha256", "image_sha256"),
    "source_group": ("source_group_id", "source_group", "leakage_group_id", "leakage_group", "group_id", "group"),
    "original_source": ("original_source", "original_image", "source", "source_filename", "original"),
    "augmentation_parent": ("augmentation_parent", "parent_image", "parent_id", "derived_from", "augmentation_source"),
    "physical_fish": ("physical_fish_id", "specimen_id", "fish_id"),
    "recording_session": ("recording_session", "camera_session"),
    "capture_session": ("capture_session", "session_id"),
    "source_folder": ("source_folder",),
}
# ``source_group`` is an opaque collection identifier.  Unlike exported image
# names, a slash in it can be meaningful (for example, a recording/session
# namespace), so it must retain its path context.  ``original_source`` and an
# augmentation parent are commonly file-name-like values emitted by different
# exporters and retain the more permissive origin normalization below.
ORIGIN_RELATION_TYPES = {"source_group", "original_source", "augmentation_parent"}
CONFIRMED_RELATION_TYPES = {
    "source_group",
    "original_source",
    "augmentation_parent",
    "physical_fish",
    "recording_session",
    "capture_session",
    "source_folder",
    "sha256",
    "file_content_sha256",
    # ``origin`` bridges an explicit source-group value in one manifest to an
    # original-source/augmentation-parent value in another manifest.
    "origin",
}


class DatasetIndependenceError(ValueError):
    """Raised when a manifest cannot safely be audited or proposed."""


@dataclass(frozen=True)
class DatasetRecord:
    """One normalized row from a study or checkpoint-lineage manifest."""

    uid: str
    sample_id: str
    split: str
    source_name: str
    row_number: int
    image_path: str
    annotation_path: str
    class_information: str
    file_sha256: str
    filename: str
    provenance: Mapping[str, tuple[str, ...]]
    original_source: str
    resolved_image_path: Path | None = None
    missing_sample_id: bool = False
    missing_split: bool = False

    @property
    def has_explicit_provenance(self) -> bool:
        return any(self.provenance.values())

    @property
    def reference(self) -> str:
        return f"{self.source_name}:{self.sample_id}"


@dataclass
class _UnionFind:
    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, size: int) -> "_UnionFind":
        return cls(parent=list(range(size)), rank=[0] * size)

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


@dataclass
class _RelationshipGraph:
    records: list[DatasetRecord]
    union_find: _UnionFind
    token_members: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def add_token(self, token: str, record_index: int) -> None:
        """Join every record that presents the same relationship token."""

        members = self.token_members[token]
        if members:
            self.union_find.union(members[0], record_index)
        if record_index not in members:
            members.append(record_index)


@dataclass(frozen=True)
class _Component:
    group_id: str
    root: int
    record_indexes: tuple[int, ...]
    evidence: tuple[dict[str, object], ...]

    @property
    def evidence_types(self) -> set[str]:
        return {str(item["type"]) for item in self.evidence}


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _field_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _text(value).casefold()).strip("_")


def _row_fields(row: Mapping[str, object]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in row.items():
        normalized = _field_name(key)
        if not normalized:
            continue
        text = _text(value)
        # Preserve the first non-empty spelling: CSV headers occasionally
        # contain aliases such as both ``source`` and ``original_source``.
        if normalized not in values or (not values[normalized] and text):
            values[normalized] = text
    return values


def _nonempty(value: str) -> bool:
    return value.strip().casefold() not in UNKNOWN_VALUES


def _split_values(value: str) -> tuple[str, ...]:
    """Read pipe-separated relationship IDs without inventing new IDs."""

    values: list[str] = []
    for item in value.split("|"):
        cleaned = item.strip()
        if _nonempty(cleaned) and cleaned not in values:
            values.append(cleaned)
    return tuple(values)


def _values_for(fields: Mapping[str, str], aliases: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for alias in aliases:
        value = fields.get(alias, "")
        for item in _split_values(value):
            if item not in result:
                result.append(item)
    return tuple(result)


def _first_value(fields: Mapping[str, str], aliases: Iterable[str]) -> str:
    values = _values_for(fields, aliases)
    return values[0] if values else ""


def _normalize_relation(value: str) -> str:
    """Normalize opaque IDs conservatively while preserving path context."""

    return re.sub(r"/+", "/", value.replace("\\", "/").strip()).casefold()


def _normalize_origin(value: str) -> str:
    """Normalize common export filename spellings to one source-photo token.

    This deliberately handles Roboflow's ``.rf.<digest>`` suffix and the
    ``name_jpg`` form found in a retained training-lineage manifest.  It does
    not remove arbitrary words or infer a specimen from timestamp-like names.
    """

    cleaned = _normalize_relation(value)
    filename = cleaned.rsplit("/", 1)[-1]
    filename = re.sub(r"\.rf\.[0-9a-f]{8,}(?=\.[^.]+$|$)", "", filename, flags=re.IGNORECASE)
    suffix = Path(filename).suffix.casefold()
    if suffix in IMAGE_SUFFIXES:
        filename = filename[: -len(suffix)]
    filename = re.sub(r"[._-](?:jpg|jpeg|png|bmp|tif|tiff|webp)$", "", filename, flags=re.IGNORECASE)
    return filename.casefold()


def _normalize_hash(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _canonical_split(value: str) -> str:
    token = _field_name(value)
    return SPLIT_ALIASES.get(token, "unknown")


def _resolve_image_path(image_path: str, image_root: Path | None) -> Path | None:
    if not image_path:
        return None
    path = Path(image_path)
    if path.is_absolute():
        return path
    return (image_root / path) if image_root is not None else path


def _record_from_mapping(
    row: Mapping[str, object],
    *,
    source_name: str,
    row_number: int,
    image_root: Path | None,
    forced_split: str | None = None,
) -> DatasetRecord:
    fields = _row_fields(row)
    sample_value = _first_value(fields, FIELD_ALIASES["sample_id"])
    image_path = _first_value(fields, FIELD_ALIASES["image_path"])
    fallback_id = _normalize_origin(image_path) if image_path else ""
    missing_sample_id = not bool(sample_value)
    sample_id = sample_value or fallback_id or f"row-{row_number}"
    raw_split = forced_split if forced_split is not None else _first_value(fields, FIELD_ALIASES["split"])
    split = _canonical_split(raw_split)
    missing_split = split == "unknown"

    provenance: dict[str, tuple[str, ...]] = {}
    for relation, aliases in FIELD_ALIASES.items():
        if relation not in {
            "source_group",
            "original_source",
            "augmentation_parent",
            "physical_fish",
            "recording_session",
            "capture_session",
            "source_folder",
        }:
            continue
        raw_values = _values_for(fields, aliases)
        # Source-group values are opaque group IDs, not file names.  Preserve
        # a path-like namespace here so ``session-a/fish-001`` cannot be
        # silently joined to ``session-b/fish-001`` merely because their final
        # path segment is the same.
        normalizer = (
            _normalize_relation
            if relation == "source_group"
            else _normalize_origin
            if relation in ORIGIN_RELATION_TYPES
            else _normalize_relation
        )
        normalized = tuple(value for value in (normalizer(item) for item in raw_values) if value)
        if normalized:
            provenance[relation] = tuple(dict.fromkeys(normalized))

    filename = _first_value(fields, ("filename",))
    if not filename and image_path:
        filename = Path(image_path).name
    return DatasetRecord(
        uid=f"{source_name}:{row_number}",
        sample_id=sample_id,
        split=split,
        source_name=source_name,
        row_number=row_number,
        image_path=image_path,
        annotation_path=_first_value(fields, FIELD_ALIASES["annotation_path"]),
        class_information=_first_value(fields, FIELD_ALIASES["class_information"]),
        file_sha256=_normalize_hash(_first_value(fields, FIELD_ALIASES["file_sha256"])),
        filename=_normalize_origin(filename) if filename else "",
        provenance=provenance,
        original_source="|".join(_values_for(fields, FIELD_ALIASES["original_source"])),
        resolved_image_path=_resolve_image_path(image_path, image_root),
        missing_sample_id=missing_sample_id,
        missing_split=missing_split,
    )


def normalize_records(
    rows: Iterable[Mapping[str, object] | DatasetRecord],
    *,
    source_name: str = "study",
    image_root: Path | None = None,
    forced_split: str | None = None,
) -> list[DatasetRecord]:
    """Normalize manifest rows without reading, moving, or rewriting images."""

    normalized_forced_split = _canonical_split(forced_split) if forced_split is not None else None
    if forced_split is not None and normalized_forced_split not in SPLITS:
        raise DatasetIndependenceError(f"Forced split must be one of {', '.join(SPLITS)}.")

    result: list[DatasetRecord] = []
    for row_number, row in enumerate(rows, start=2):
        if isinstance(row, DatasetRecord):
            # Model-training lineage is sometimes already normalized by a
            # caller.  It must still be considered training exposure even if
            # its retained historical row says ``validation`` or ``test``.
            # ``DatasetRecord`` is frozen, hence the explicit replacement.
            changes: dict[str, object] = {}
            if normalized_forced_split is not None:
                changes.update(split=normalized_forced_split, missing_split=False)
            # A caller may normalize records before learning the image root.
            # Preserve a previously resolved path, but make an accessible
            # root usable for content-hash evidence when the record has none.
            if image_root is not None and (
                row.resolved_image_path is None or not row.resolved_image_path.is_absolute()
            ):
                changes["resolved_image_path"] = _resolve_image_path(row.image_path, image_root)
            result.append(replace(row, **changes) if changes else row)
        elif isinstance(row, Mapping):
            result.append(
                _record_from_mapping(
                    row,
                    source_name=source_name,
                    row_number=row_number,
                    image_root=image_root,
                    forced_split=normalized_forced_split,
                )
            )
        else:
            raise DatasetIndependenceError(f"Manifest row {row_number} must be an object.")
    if not result:
        raise DatasetIndependenceError("Manifest contains no records.")
    return result


def _json_rows(path: Path) -> list[Mapping[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetIndependenceError(f"Could not read JSON manifest {path}: {exc}") from exc
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = next(
            (payload[key] for key in ("records", "samples", "items", "rows") if isinstance(payload.get(key), list)),
            None,
        )
        if rows is None:
            raise DatasetIndependenceError("JSON manifest must be a list or contain records, samples, items, or rows.")
    else:
        raise DatasetIndependenceError("JSON manifest must be an object or a list.")
    if not all(isinstance(row, Mapping) for row in rows):
        raise DatasetIndependenceError("Every JSON manifest row must be an object.")
    return list(rows)


def load_manifest(
    path: Path | str,
    *,
    source_name: str = "study",
    image_root: Path | str | None = None,
    forced_split: str | None = None,
) -> list[DatasetRecord]:
    """Load a CSV or JSON manifest as normalized records.

    Relative ``image_path`` values are interpreted relative to ``image_root``
    when supplied, otherwise relative to the manifest's directory.  Paths are
    only used when optional perceptual fingerprints are requested.
    """

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise DatasetIndependenceError(f"Manifest does not exist: {manifest_path}")
    root = Path(image_root) if image_root is not None else manifest_path.parent
    suffix = manifest_path.suffix.casefold()
    if suffix == ".csv":
        try:
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames:
                    raise DatasetIndependenceError(f"CSV manifest has no header: {manifest_path}")
                rows = list(reader)
        except OSError as exc:
            raise DatasetIndependenceError(f"Could not read CSV manifest {manifest_path}: {exc}") from exc
    elif suffix == ".json":
        rows = _json_rows(manifest_path)
    else:
        raise DatasetIndependenceError("Manifest must be a .csv or .json file.")
    return normalize_records(rows, source_name=source_name, image_root=root, forced_split=forced_split)


def _build_graph(
    records: Sequence[DatasetRecord],
    *,
    file_content_hashes: Mapping[int, str] | None = None,
) -> _RelationshipGraph:
    graph = _RelationshipGraph(list(records), _UnionFind.create(len(records)))
    for index, record in enumerate(records):
        for relation_type, values in record.provenance.items():
            for value in values:
                graph.add_token(f"{relation_type}:{value}", index)
                # A source-group ID, original source, and augmentation parent
                # often use different column names in different exports.  The
                # shared origin token detects that lineage without requiring
                # exact header alignment.
                if relation_type in ORIGIN_RELATION_TYPES:
                    graph.add_token(f"origin:{value}", index)
        if record.filename:
            graph.add_token(f"filename:{record.filename}", index)
        if record.file_sha256:
            graph.add_token(f"sha256:{record.file_sha256}", index)
        content_hash = (file_content_hashes or {}).get(index, "")
        if content_hash:
            # Keep ``sha256`` as a compatibility token for consumers that
            # already inspect that evidence, while retaining the more precise
            # token type in the component evidence and audit report.
            graph.add_token(f"sha256:{content_hash}", index)
            graph.add_token(f"file_content_sha256:{content_hash}", index)
    return graph


def _stable_record_payload(record: DatasetRecord) -> dict[str, object]:
    """Return row-order-independent evidence for IDs and audit binding.

    ``uid`` deliberately includes a manifest row number and therefore cannot
    safely contribute to a reproducible group ID.  The normalized content is
    enough to retain duplicate-row multiplicity while making the order of the
    input iterable irrelevant.
    """

    return {
        "sample_id": record.sample_id,
        "split": record.split,
        "image_path": record.image_path,
        "annotation_path": record.annotation_path,
        "class_information": record.class_information,
        "file_sha256": record.file_sha256,
        "filename": record.filename,
        "provenance": {
            relation: sorted(values)
            for relation, values in sorted(record.provenance.items())
        },
        "original_source": record.original_source,
        "missing_sample_id": record.missing_sample_id,
        "missing_split": record.missing_split,
    }


def _stable_record_key(record: DatasetRecord) -> str:
    return json.dumps(_stable_record_payload(record), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _records_sha256(records: Sequence[DatasetRecord]) -> str:
    """Hash normalized study evidence independently of CSV/JSON row order."""

    digest = hashlib.sha256()
    for key in sorted(_stable_record_key(record) for record in records):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _component_id(records: Sequence[DatasetRecord], indexes: Sequence[int], tokens: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted((_stable_record_key(records[index]) for index in indexes)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for token in sorted(tokens):
        digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    return "lg_" + digest.hexdigest()[:16]


def _components(graph: _RelationshipGraph) -> list[_Component]:
    record_indexes: dict[int, list[int]] = defaultdict(list)
    for index in range(len(graph.records)):
        record_indexes[graph.union_find.find(index)].append(index)
    tokens_by_root: dict[int, list[str]] = defaultdict(list)
    for token, indexes in graph.token_members.items():
        roots = {graph.union_find.find(index) for index in indexes}
        for root in roots:
            tokens_by_root[root].append(token)
    result: list[_Component] = []
    for root, indexes in record_indexes.items():
        tokens = sorted(tokens_by_root[root])
        evidence = []
        for token in tokens:
            relation_type, _, value = token.partition(":")
            members = graph.token_members[token]
            cross_split = len({graph.records[index].split for index in members} & set(SPLITS)) > 1
            evidence.append({"type": relation_type, "value": value, "cross_split": cross_split})
        result.append(
            _Component(
                group_id=_component_id(graph.records, indexes, tokens),
                root=root,
                record_indexes=tuple(sorted(indexes)),
                evidence=tuple(evidence),
            )
        )
    return sorted(result, key=lambda component: component.group_id)


def _hash_image(path: Path) -> int:
    """Return a tiny 64-bit average hash for optional duplicate review.

    The fingerprint is a candidate generator, not a biometric identifier and
    not a substitute for source-group metadata.
    """

    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        thumbnail = image.resize((8, 8), resampling)
        flattened = getattr(thumbnail, "get_flattened_data", None)
        pixels = list(flattened()) if callable(flattened) else list(thumbnail.getdata())
    threshold = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= threshold)
    return value


def _file_sha256(path: Path) -> str:
    """Return a content digest without loading an image decoder or model."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_file_hash_scan(
    records: Sequence[DatasetRecord],
) -> tuple[dict[int, str], dict[str, object]]:
    """Compute exact file-content hashes wherever manifest paths are usable.

    Declared hashes remain useful provenance, but an older manifest often has
    none.  This scan closes that gap for accessible local files without
    requiring the optional perceptual-duplicate scan.  Missing files are
    reported rather than fabricated, moved, or treated as a successful scan.
    """

    fingerprints: dict[int, str] = {}
    unavailable: list[dict[str, str]] = []
    cache: dict[Path, str] = {}
    declared_mismatches: list[dict[str, str]] = []
    for index, record in enumerate(records):
        path = record.resolved_image_path
        if path is None or not path.is_file():
            unavailable.append({"sample": record.reference, "reason": "image_path is unavailable"})
            continue
        try:
            # Resolve for caching only; the original path remains the audit
            # evidence and no file is modified.
            cache_key = path.resolve()
            digest = cache.get(cache_key)
            if digest is None:
                digest = _file_sha256(path)
                cache[cache_key] = digest
            fingerprints[index] = digest
            if record.file_sha256 and record.file_sha256 != digest:
                declared_mismatches.append(
                    {
                        "sample": record.reference,
                        "declared_sha256": record.file_sha256,
                        "computed_sha256": digest,
                    }
                )
        except OSError as exc:
            unavailable.append({"sample": record.reference, "reason": f"could not hash image: {exc}"})

    if len(fingerprints) == len(records):
        status = "COMPLETE"
    elif fingerprints:
        status = "PARTIAL"
    else:
        status = "NOT_AVAILABLE"
    return fingerprints, {
        "algorithm": "sha256",
        "status": status,
        "fingerprinted_samples": len(fingerprints),
        "unavailable_sample_count": len(unavailable),
        "unavailable_samples": unavailable,
        "declared_hash_mismatch_count": len(declared_mismatches),
        "declared_hash_mismatches": declared_mismatches,
        "note": (
            "Exact file-content hashes are computed only for accessible image paths; "
            "missing paths are reported and never replaced with assumed hashes."
        ),
    }


def _near_duplicate_scan(
    records: Sequence[DatasetRecord],
    *,
    enabled: bool,
    max_hamming_distance: int,
    max_comparisons: int,
) -> dict[str, object]:
    if not enabled:
        return {
            "requested": False,
            "status": "NOT_REQUESTED",
            "max_hamming_distance": max_hamming_distance,
            "max_comparisons": max_comparisons,
            "comparisons": 0,
            "candidates": [],
            "unavailable_samples": [],
        }
    if not 0 <= max_hamming_distance <= 64:
        raise DatasetIndependenceError("near-duplicate Hamming distance must be between 0 and 64.")
    if max_comparisons <= 0:
        raise DatasetIndependenceError("near-duplicate maximum comparisons must be positive.")

    fingerprints: dict[int, int] = {}
    unavailable: list[dict[str, str]] = []
    for index, record in enumerate(records):
        path = record.resolved_image_path
        if path is None or not path.is_file():
            unavailable.append({"sample": record.reference, "reason": "image_path is unavailable"})
            continue
        try:
            fingerprints[index] = _hash_image(path)
        except Exception as exc:  # Pillow decoder boundary; report, never delete.
            unavailable.append({"sample": record.reference, "reason": f"could not fingerprint image: {exc}"})

    comparisons = 0
    stopped_for_limit = False
    candidates: list[dict[str, object]] = []
    for first, second in combinations(sorted(fingerprints), 2):
        if records[first].split == records[second].split:
            continue
        if records[first].split not in SPLITS or records[second].split not in SPLITS:
            continue
        if comparisons >= max_comparisons:
            stopped_for_limit = True
            break
        comparisons += 1
        distance = (fingerprints[first] ^ fingerprints[second]).bit_count()
        if distance <= max_hamming_distance:
            candidates.append(
                {
                    "left": records[first].reference,
                    "right": records[second].reference,
                    "left_index": first,
                    "right_index": second,
                    "hamming_distance": distance,
                    "kind": "PERCEPTUAL_HASH_CANDIDATE",
                }
            )
    status = "COMPLETE"
    if unavailable or stopped_for_limit:
        status = "INCOMPLETE"
    return {
        "requested": True,
        "status": status,
        "max_hamming_distance": max_hamming_distance,
        "max_comparisons": max_comparisons,
        "comparisons": comparisons,
        "stopped_for_comparison_limit": stopped_for_limit,
        "fingerprinted_samples": len(fingerprints),
        "candidates": candidates,
        "unavailable_samples": unavailable,
    }


def _pairs_for_splits(splits: Iterable[str]) -> list[str]:
    result: list[str] = []
    members = set(splits)
    for first, second, name in (
        ("train", "validation", "train_validation"),
        ("train", "test", "train_test"),
        ("validation", "test", "validation_test"),
    ):
        if first in members and second in members:
            result.append(name)
    return result


def _component_payload(component: _Component, records: Sequence[DatasetRecord]) -> dict[str, object]:
    members = [records[index] for index in component.record_indexes]
    return {
        "source_group_id": component.group_id,
        "splits": sorted({record.split for record in members}),
        "samples": [
            {"sample_id": record.sample_id, "reference": record.reference, "split": record.split}
            for record in members
        ],
        "evidence": list(component.evidence),
    }


def _pair_overlap_payload(components: Sequence[_Component], records: Sequence[DatasetRecord]) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {
        "train_validation": [],
        "train_test": [],
        "validation_test": [],
    }
    for component in components:
        members = [records[index] for index in component.record_indexes]
        for pair in _pairs_for_splits(record.split for record in members):
            result[pair].append(_component_payload(component, records))
    return result


def _cross_split_token_payload(
    graph: _RelationshipGraph,
    *,
    token_prefix: str,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for token, indexes in sorted(graph.token_members.items()):
        if not token.startswith(token_prefix):
            continue
        samples = [graph.records[index] for index in indexes]
        if len({sample.split for sample in samples} & set(SPLITS)) < 2:
            continue
        _, _, value = token.partition(":")
        result.append(
            {
                "value": value,
                "splits": sorted({sample.split for sample in samples}),
                "samples": [sample.reference for sample in samples],
            }
        )
    return result


def _status_for_scope(
    *,
    target: str,
    records: Sequence[DatasetRecord],
    overlaps: Mapping[str, Sequence[Mapping[str, object]]],
    near_scan: Mapping[str, object],
    model_training_lineage_status: str,
    exact_file_hash_scan: Mapping[str, object],
) -> dict[str, object]:
    if target == "validation":
        required_splits = {"train", "validation"}
        pairs = ("train_validation",)
    elif target == "test":
        required_splits = {"train", "test"}
        pairs = ("train_test", "validation_test")
    else:  # pragma: no cover - caller is internal and fixed.
        raise ValueError(target)

    reasons: list[str] = []
    present = {record.split for record in records}
    missing_required = sorted(required_splits - present)
    if missing_required:
        reasons.append("Missing required split(s): " + ", ".join(missing_required) + ".")
    if model_training_lineage_status != "PROVIDED":
        lineage_reason = (
            "Model-training lineage was not supplied."
            if model_training_lineage_status == "MISSING"
            else "Model-training lineage was supplied but contains no records."
        )
        reasons.append(f"{lineage_reason} Deployed-model training exposure cannot be verified.")
    missing_provenance = [
        record.reference
        for record in records
        if record.split in required_splits and not record.has_explicit_provenance
    ]
    if missing_provenance:
        reasons.append(f"Missing explicit provenance for {len(missing_provenance)} relevant sample(s).")
    malformed = [record.reference for record in records if record.missing_sample_id or record.missing_split]
    if malformed:
        reasons.append(f"Missing sample ID or recognized split for {len(malformed)} sample(s).")

    confirmed: list[Mapping[str, object]] = []
    candidate_only: list[Mapping[str, object]] = []
    for pair in pairs:
        for overlap in overlaps.get(pair, []):
            evidence = overlap.get("evidence", [])
            evidence_types = {
                str(item.get("type"))
                for item in evidence
                if isinstance(item, Mapping) and item.get("cross_split")
            }
            if evidence_types & CONFIRMED_RELATION_TYPES:
                confirmed.append(overlap)
            else:
                candidate_only.append(overlap)
    if confirmed:
        reasons.append(f"Confirmed cross-split lineage overlap in {len(confirmed)} source group(s).")
    if candidate_only:
        reasons.append(f"Unresolved filename or perceptual-duplicate overlap in {len(candidate_only)} source group(s).")
    if near_scan.get("requested") and near_scan.get("status") != "COMPLETE":
        reasons.append("Requested near-duplicate scan did not complete for every available comparison.")
    mismatch_samples = exact_file_hash_scan.get("declared_hash_mismatches", [])
    if isinstance(mismatch_samples, Sequence) and not isinstance(mismatch_samples, (str, bytes)):
        relevant_mismatches = [
            mismatch
            for mismatch in mismatch_samples
            if isinstance(mismatch, Mapping)
            and any(
                record.reference == str(mismatch.get("sample", "")) and record.split in required_splits
                for record in records
            )
        ]
        if relevant_mismatches:
            reasons.append(f"Declared file SHA-256 disagrees with accessible content for {len(relevant_mismatches)} relevant sample(s).")

    if confirmed:
        status = "FAIL"
    elif reasons:
        status = "BLOCKED"
    else:
        status = "PASS"
    return {
        "status": status,
        "reasons": reasons,
        "missing_provenance_samples": missing_provenance,
        "confirmed_overlap_groups": len(confirmed),
        "unresolved_overlap_groups": len(candidate_only),
        "model_training_lineage_status": model_training_lineage_status,
    }


def _manifest_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset_independence(
    records: Iterable[Mapping[str, object] | DatasetRecord],
    *,
    model_training_records: Iterable[Mapping[str, object] | DatasetRecord] | None = None,
    image_root: Path | str | None = None,
    near_duplicates: bool = False,
    near_hamming_distance: int = 4,
    near_max_comparisons: int = 250_000,
    manifest_path: Path | None = None,
    model_training_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Audit lineage and duplicate evidence without changing any input files.

    Training-lineage rows are always assigned the ``train`` role, even when a
    historical CSV calls its rows ``validation``.  This prevents a checkpoint's
    prior exposure from being hidden by a misleading column value.
    """

    root = Path(image_root) if image_root is not None else None
    study_records = normalize_records(records, source_name="study", image_root=root)
    if model_training_records is None:
        lineage_records: list[DatasetRecord] = []
        model_training_lineage_status = "MISSING"
    else:
        # Materialize once so an explicitly empty iterator can be reported as
        # incomplete lineage rather than silently becoming equivalent to a
        # complete training record.  The study manifest itself remains
        # untouched.
        supplied_lineage_records = list(model_training_records)
        if supplied_lineage_records:
            lineage_records = normalize_records(
                supplied_lineage_records,
                source_name="model_training",
                image_root=root,
                forced_split="train",
            )
            model_training_lineage_status = "PROVIDED"
        else:
            lineage_records = []
            model_training_lineage_status = "EMPTY"
    all_records = study_records + lineage_records
    file_content_hashes, exact_file_hash_scan = _exact_file_hash_scan(all_records)
    graph = _build_graph(all_records, file_content_hashes=file_content_hashes)
    near_scan = _near_duplicate_scan(
        all_records,
        enabled=near_duplicates,
        max_hamming_distance=near_hamming_distance,
        max_comparisons=near_max_comparisons,
    )
    for candidate_number, candidate in enumerate(near_scan["candidates"], start=1):
        assert isinstance(candidate, Mapping)
        graph.add_token(f"near_duplicate:{candidate_number:06d}", int(candidate["left_index"]))
        graph.add_token(f"near_duplicate:{candidate_number:06d}", int(candidate["right_index"]))

    components = _components(graph)
    overlaps = _pair_overlap_payload(components, all_records)
    samples_by_split = {split: sum(record.split == split for record in all_records) for split in SPLITS}
    groups_by_split = {
        split: sum(any(all_records[index].split == split for index in component.record_indexes) for component in components)
        for split in SPLITS
    }
    missing_provenance = [record.reference for record in all_records if not record.has_explicit_provenance]
    malformed = [record.reference for record in all_records if record.missing_sample_id or record.missing_split]
    validation = _status_for_scope(
        target="validation",
        records=all_records,
        overlaps=overlaps,
        near_scan=near_scan,
        model_training_lineage_status=model_training_lineage_status,
        exact_file_hash_scan=exact_file_hash_scan,
    )
    test = _status_for_scope(
        target="test",
        records=all_records,
        overlaps=overlaps,
        near_scan=near_scan,
        model_training_lineage_status=model_training_lineage_status,
        exact_file_hash_scan=exact_file_hash_scan,
    )
    certification = "PASS" if validation["status"] == "PASS" and test["status"] == "PASS" else "BLOCKED"
    if test["status"] == "PASS":
        required_label = "independent test performance"
    elif validation["status"] == "PASS":
        required_label = "independent validation result; locked test not yet eligible"
    else:
        required_label = "development/compatibility result only"

    study_manifest_sha256 = _manifest_sha256(manifest_path)
    model_training_manifest_sha256 = _manifest_sha256(model_training_manifest_path)
    normalized_study_records_sha256 = _records_sha256(study_records)

    return {
        "schema_version": 1,
        "audit_type": "DATASET_INDEPENDENCE_AUDIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "study_manifest": str(manifest_path) if manifest_path is not None else None,
            "study_manifest_sha256": study_manifest_sha256,
            "model_training_manifest": str(model_training_manifest_path) if model_training_manifest_path is not None else None,
            "model_training_manifest_sha256": model_training_manifest_sha256,
            "model_lineage_included": bool(lineage_records),
            "model_training_lineage_status": model_training_lineage_status,
            "model_training_lineage_required_for_pass": True,
        },
        "evidence_binding": {
            "algorithm": "sha256",
            "audited_study_artifact_sha256": study_manifest_sha256 or normalized_study_records_sha256,
            "audited_study_artifact_kind": "manifest_file" if study_manifest_sha256 else "normalized_records",
            "study_manifest_sha256": study_manifest_sha256,
            "normalized_study_records_sha256": normalized_study_records_sha256,
            "note": (
                "Bind a locked validation session to audited_study_artifact_sha256. "
                "When a readable manifest path is supplied this is its byte hash; otherwise it is a "
                "canonical hash of the normalized audited study records."
            ),
        },
        "counts": {
            "samples": len(all_records),
            "study_samples": len(study_records),
            "model_training_samples": len(lineage_records),
            "samples_by_split": samples_by_split,
            "groups_by_split": groups_by_split,
            "connected_source_groups": len(components),
        },
        "provenance": {
            "missing_explicit_provenance_count": len(missing_provenance),
            "missing_explicit_provenance_samples": missing_provenance,
            "malformed_sample_count": len(malformed),
            "malformed_samples": malformed,
            "accepted_relationships": sorted(
                {
                    alias
                    for relation in (
                        "source_group", "original_source", "augmentation_parent",
                        "physical_fish", "recording_session", "capture_session", "source_folder",
                    )
                    for alias in FIELD_ALIASES[relation]
                }
            ),
            "note": "Filenames and hashes can reveal duplicate evidence but do not satisfy the explicit-provenance requirement by themselves.",
        },
        "cross_split_overlap": {
            name: {
                "group_count": len(entries),
                "groups": entries,
            }
            for name, entries in overlaps.items()
        },
        "duplicates": {
            "duplicate_filenames_across_splits": _cross_split_token_payload(graph, token_prefix="filename:"),
            "identical_hashes_across_splits": _cross_split_token_payload(graph, token_prefix="sha256:"),
            "identical_file_contents_across_splits": _cross_split_token_payload(graph, token_prefix="file_content_sha256:"),
            "exact_file_hash_scan": exact_file_hash_scan,
            "near_duplicates": near_scan,
        },
        "independence": {
            "validation": validation,
            "test": test,
        },
        "FINAL_PERFORMANCE_CERTIFICATION": certification,
        "reporting_guard": {
            "may_label_independent_validation": validation["status"] == "PASS",
            "may_label_independent_test": test["status"] == "PASS",
            "required_performance_label": required_label,
            "rule": "Do not call a result independent unless the matching dataset-independence gate is PASS.",
        },
        "limitations": [
            "This audit cannot prove that separately named samples show different physical fish; collection metadata and human review remain required.",
            "Near-duplicate fingerprints are conservative review candidates, not automatic deletion or relabeling instructions.",
            "The audit does not calculate model accuracy, tune thresholds, load checkpoints, or change production inspection behavior.",
        ],
    }


def render_human_report(report: Mapping[str, object]) -> str:
    """Render the machine-readable audit as a concise operator/research record."""

    counts = report.get("counts", {}) if isinstance(report.get("counts"), Mapping) else {}
    sample_counts = counts.get("samples_by_split", {}) if isinstance(counts.get("samples_by_split"), Mapping) else {}
    group_counts = counts.get("groups_by_split", {}) if isinstance(counts.get("groups_by_split"), Mapping) else {}
    overlap = report.get("cross_split_overlap", {}) if isinstance(report.get("cross_split_overlap"), Mapping) else {}
    independence = report.get("independence", {}) if isinstance(report.get("independence"), Mapping) else {}
    provenance = report.get("provenance", {}) if isinstance(report.get("provenance"), Mapping) else {}
    duplicates = report.get("duplicates", {}) if isinstance(report.get("duplicates"), Mapping) else {}
    inputs = report.get("inputs", {}) if isinstance(report.get("inputs"), Mapping) else {}
    evidence_binding = report.get("evidence_binding", {}) if isinstance(report.get("evidence_binding"), Mapping) else {}
    exact_scan = duplicates.get("exact_file_hash_scan", {}) if isinstance(duplicates.get("exact_file_hash_scan"), Mapping) else {}
    near_scan = duplicates.get("near_duplicates", {}) if isinstance(duplicates.get("near_duplicates"), Mapping) else {}
    near_status = str(near_scan.get("status") or "NOT_REPORTED")
    if near_status == "NOT_REQUESTED":
        near_detail = "not run; suspicious near duplicates were not scanned"
    elif near_status == "INCOMPLETE":
        near_detail = "incomplete; review unavailable samples or the comparison limit before relying on this check"
    elif near_status == "COMPLETE":
        near_detail = (
            f"complete; {near_scan.get('comparisons', 0)} cross-split comparison(s), "
            f"{len(near_scan.get('candidates', [])) if isinstance(near_scan.get('candidates'), Sequence) else 0} candidate(s)"
        )
    else:
        near_detail = "status was not recognized; do not assume suspicious near duplicates were checked"
    lines = [
        "DATASET INDEPENDENCE AUDIT",
        "",
        f"Samples: {counts.get('samples', 0)}",
        "Lineage-connected groups combine every recorded relationship and may differ from a native source-group column.",
        f"Train lineage-connected groups: {group_counts.get('train', 0)} (samples: {sample_counts.get('train', 0)})",
        f"Validation lineage-connected groups: {group_counts.get('validation', 0)} (samples: {sample_counts.get('validation', 0)})",
        f"Test lineage-connected groups: {group_counts.get('test', 0)} (samples: {sample_counts.get('test', 0)})",
        f"Model-training lineage: {inputs.get('model_training_lineage_status', 'NOT_REPORTED')}",
        f"Exact file-content hash scan: {exact_scan.get('status', 'NOT_REPORTED')} ({exact_scan.get('fingerprinted_samples', 0)} sample(s) fingerprinted)",
        f"Audited study artifact SHA-256: {evidence_binding.get('audited_study_artifact_sha256', 'NOT_REPORTED')}",
        "",
    ]
    for key, label in (
        ("train_validation", "Train <-> Validation overlap"),
        ("train_test", "Train <-> Test overlap"),
        ("validation_test", "Validation <-> Test overlap"),
    ):
        item = overlap.get(key, {}) if isinstance(overlap.get(key), Mapping) else {}
        lines.append(f"{label}: {item.get('group_count', 0)} group(s)")
    lines.extend(
        [
            "",
            f"Missing explicit provenance: {provenance.get('missing_explicit_provenance_count', 0)} sample(s)",
            f"Near-duplicate scan: {near_status} ({near_detail})",
            f"Independent validation: {((independence.get('validation') or {}) if isinstance(independence.get('validation'), Mapping) else {}).get('status', 'BLOCKED')}",
            f"Independent test: {((independence.get('test') or {}) if isinstance(independence.get('test'), Mapping) else {}).get('status', 'BLOCKED')}",
            f"FINAL PERFORMANCE CERTIFICATION = {report.get('FINAL_PERFORMANCE_CERTIFICATION', 'BLOCKED')}",
        ]
    )
    guard = report.get("reporting_guard") if isinstance(report.get("reporting_guard"), Mapping) else {}
    if guard:
        lines.extend(["", f"Required performance label: {guard.get('required_performance_label', 'development/compatibility result only')}"])
    return "\n".join(lines) + "\n"


def _parse_ratios(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split(","):
        key, separator, raw_number = item.partition("=")
        if not separator:
            raise DatasetIndependenceError("Split ratios must use train=0.70,validation=0.15,test=0.15 format.")
        split = _canonical_split(key)
        if split not in SPLITS or split in result:
            raise DatasetIndependenceError("Split ratios must specify train, validation, and test exactly once.")
        try:
            number = float(raw_number)
        except ValueError as exc:
            raise DatasetIndependenceError(f"Invalid split ratio for {key!r}.") from exc
        if number <= 0:
            raise DatasetIndependenceError("Every split ratio must be positive.")
        result[split] = number
    if set(result) != set(SPLITS) or abs(sum(result.values()) - 1.0) > 1e-9:
        raise DatasetIndependenceError("Split ratios must contain train, validation, test and sum to 1.0.")
    return result


def propose_group_aware_split_manifest(
    records: Iterable[Mapping[str, object] | DatasetRecord],
    *,
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
    image_root: Path | str | None = None,
) -> list[dict[str, str]]:
    """Create a review-only split proposal that keeps whole lineage groups intact.

    No source image or existing split is touched.  The proposal prioritizes
    target sample counts, not class balance; reviewers must still inspect class
    representation and study design before materializing any dataset.
    """

    normalized = normalize_records(records, source_name="study", image_root=Path(image_root) if image_root is not None else None)
    missing = [record.reference for record in normalized if record.missing_sample_id or not record.has_explicit_provenance]
    if missing:
        raise DatasetIndependenceError(
            "Cannot propose a group-aware split with missing sample IDs or explicit provenance: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    split_ratios = dict(ratios or {"train": 0.70, "validation": 0.15, "test": 0.15})
    if set(split_ratios) != set(SPLITS) or any(value <= 0 for value in split_ratios.values()) or abs(sum(split_ratios.values()) - 1.0) > 1e-9:
        raise DatasetIndependenceError("Proposal ratios must contain positive train, validation, and test values that sum to 1.0.")
    graph = _build_graph(normalized)
    components = _components(graph)
    if len(components) < len(SPLITS):
        raise DatasetIndependenceError("At least three independent source groups are required for a train/validation/test proposal.")

    randomized = list(components)
    random.Random(seed).shuffle(randomized)
    total_samples = len(normalized)
    targets = {split: total_samples * split_ratios[split] for split in SPLITS}
    assigned_counts = {split: 0 for split in SPLITS}
    assignments: dict[int, str] = {}
    empty_splits = list(SPLITS)
    for position, component in enumerate(randomized):
        remaining_groups = len(randomized) - position
        if remaining_groups == len(empty_splits):
            split = empty_splits.pop(0)
        else:
            split = min(
                SPLITS,
                key=lambda candidate: (
                    assigned_counts[candidate] / max(targets[candidate], 1.0),
                    assigned_counts[candidate],
                    candidate,
                ),
            )
            if split in empty_splits:
                empty_splits.remove(split)
        assignments[component.root] = split
        assigned_counts[split] += len(component.record_indexes)

    component_by_index = {
        index: component
        for component in components
        for index in component.record_indexes
    }
    proposal: list[dict[str, str]] = []
    for index, record in enumerate(normalized):
        component = component_by_index[index]
        proposal.append(
            {
                "sample_id": record.sample_id,
                "source_group_id": component.group_id,
                "image_path": record.image_path,
                "annotation_path": record.annotation_path,
                "class_information": record.class_information,
                "split": assignments[component.root],
                "original_source": record.original_source,
            }
        )
    return sorted(proposal, key=lambda row: (row["split"], row["source_group_id"], row["sample_id"]))


def _write_text(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise DatasetIndependenceError(f"Refusing to overwrite existing output: {path}. Use --overwrite-output after review.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_proposed_split_manifest(path: Path | str, rows: Sequence[Mapping[str, str]], *, overwrite: bool = False) -> None:
    """Persist a proposal CSV only after the caller explicitly selected a path."""

    output = Path(path)
    if output.exists() and not overwrite:
        raise DatasetIndependenceError(f"Refusing to overwrite existing proposal: {output}. Use --overwrite-output after review.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PROPOSAL_COLUMNS)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in PROPOSAL_COLUMNS} for row in rows)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV or JSON study manifest to audit")
    parser.add_argument("--model-training-manifest", type=Path, default=None, help="Optional CSV/JSON lineage manifest for the deployed model; every row is treated as training exposure")
    parser.add_argument("--image-root", type=Path, default=None, help="Base directory for relative image paths when --near-duplicates is used")
    parser.add_argument("--near-duplicates", action="store_true", help="Optionally scan cross-split image paths with a small perceptual hash")
    parser.add_argument("--near-hamming-distance", type=int, default=4, help="Maximum 64-bit average-hash distance for a review candidate (default: 4)")
    parser.add_argument("--near-max-comparisons", type=int, default=250_000, help="Safety cap for optional cross-split perceptual comparisons")
    parser.add_argument("--output", type=Path, required=True, help="Machine-readable JSON audit output")
    parser.add_argument("--text-output", type=Path, default=None, help="Human-readable audit output (default: alongside --output)")
    parser.add_argument("--proposed-split-manifest", type=Path, default=None, help="Optional review-only group-aware CSV proposal")
    parser.add_argument("--split-ratios", default="train=0.70,validation=0.15,test=0.15", help="Ratios for an optional proposed split")
    parser.add_argument("--seed", type=int, default=42, help="Seed for an optional proposed split")
    parser.add_argument("--overwrite-output", action="store_true", help="Allow replacement of an existing audit/proposal output after review")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        image_root = args.image_root or args.manifest.parent
        study_records = load_manifest(args.manifest, source_name="study", image_root=image_root)
        lineage_records = (
            load_manifest(
                args.model_training_manifest,
                source_name="model_training",
                image_root=image_root,
                forced_split="train",
            )
            if args.model_training_manifest is not None
            else None
        )
        report = audit_dataset_independence(
            study_records,
            model_training_records=lineage_records,
            image_root=image_root,
            near_duplicates=args.near_duplicates,
            near_hamming_distance=args.near_hamming_distance,
            near_max_comparisons=args.near_max_comparisons,
            manifest_path=args.manifest,
            model_training_manifest_path=args.model_training_manifest,
        )
        text = render_human_report(report)
        text_output = args.text_output or args.output.with_suffix(".txt")
        _write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n", overwrite=args.overwrite_output)
        _write_text(text_output, text, overwrite=args.overwrite_output)
        if args.proposed_split_manifest is not None:
            proposal = propose_group_aware_split_manifest(
                study_records,
                seed=args.seed,
                ratios=_parse_ratios(args.split_ratios),
                image_root=image_root,
            )
            write_proposed_split_manifest(args.proposed_split_manifest, proposal, overwrite=args.overwrite_output)
        print(text, end="")
        print(f"JSON audit written: {args.output}")
        print(f"Text audit written: {text_output}")
        if args.proposed_split_manifest is not None:
            print(f"Review-only proposed split manifest written: {args.proposed_split_manifest}")
        return 0
    except DatasetIndependenceError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
