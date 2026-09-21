"""Durable SQLite storage for inspection sessions and completed fish events.

The live tracker intentionally remains in memory for low-latency overlays.  This
module is its durable counterpart: it stores one immutable-ish session snapshot
and upserts the latest evidence for each ``(session_id, track_id)`` inspection
event.  Keeping the original JSON event alongside indexed columns makes history
and exports both convenient and reproducible without coupling persistence to a
particular grading-engine revision.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Iterator, Mapping
from uuid import uuid4


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json(value: object) -> str:
    """Serialize metadata predictably while retaining diagnostic values safely."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _decoded(value: object, default: object) -> object:
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def grading_config_hash(config: Mapping[str, object] | None) -> str | None:
    """Return a stable policy fingerprint, or ``None`` when no policy is known."""

    if not config:
        return None
    return sha256(_json(dict(config)).encode("utf-8")).hexdigest()


class InspectionDatabase:
    """Thread-safe, standard-library SQLite store for the canonical web app."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._initialized = False

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.initialize()
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the small durable schema lazily, so import-time has no writes."""

        with self._lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.path), timeout=10.0)
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                # WAL lets the camera worker insert an event while a dashboard
                # request is reading history, without requiring a second service.
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS inspection_sessions (
                        session_id TEXT PRIMARY KEY,
                        started_at TEXT NOT NULL,
                        ended_at TEXT,
                        git_commit TEXT,
                        model1_path TEXT,
                        model1_sha256 TEXT,
                        model2_path TEXT,
                        model2_sha256 TEXT,
                        tracker_backend TEXT,
                        grading_config_json TEXT NOT NULL DEFAULT '{}',
                        grading_config_hash TEXT,
                        camera_settings_json TEXT NOT NULL DEFAULT '{}',
                        runtime_settings_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS fish_inspections (
                        session_id TEXT NOT NULL,
                        track_id INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        final_grade TEXT NOT NULL,
                        final_support REAL,
                        second_grade TEXT,
                        second_support REAL,
                        grade_margin REAL,
                        evidence_coverage REAL,
                        observed_regions_json TEXT NOT NULL DEFAULT '[]',
                        reason_codes_json TEXT NOT NULL DEFAULT '[]',
                        model1_confidence REAL,
                        weighted_scores_json TEXT NOT NULL DEFAULT '{}',
                        part_results_json TEXT NOT NULL DEFAULT '{}',
                        color_measurements_json TEXT NOT NULL DEFAULT '{}',
                        analysis_json TEXT NOT NULL DEFAULT '{}',
                        processing_time_ms REAL,
                        event_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, track_id),
                        FOREIGN KEY (session_id) REFERENCES inspection_sessions(session_id)
                            ON DELETE RESTRICT
                    );

                    CREATE INDEX IF NOT EXISTS idx_fish_inspections_timestamp
                        ON fish_inspections(timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_fish_inspections_session_timestamp
                        ON fish_inspections(session_id, timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_fish_inspections_grade
                        ON fish_inspections(final_grade);
                    """
                )
                connection.commit()
            finally:
                connection.close()
            self._initialized = True

    @staticmethod
    def _session_columns(metadata: Mapping[str, object]) -> dict[str, object]:
        grading = _mapping(metadata.get("grading_config"))
        return {
            "started_at": str(metadata.get("started_at") or _now()),
            "ended_at": metadata.get("ended_at"),
            "git_commit": metadata.get("git_commit"),
            "model1_path": metadata.get("model1_path"),
            "model1_sha256": metadata.get("model1_sha256"),
            "model2_path": metadata.get("model2_path"),
            "model2_sha256": metadata.get("model2_sha256"),
            "tracker_backend": metadata.get("tracker_backend"),
            "grading_config_json": _json(grading),
            "grading_config_hash": metadata.get("grading_config_hash") or grading_config_hash(grading),
            "camera_settings_json": _json(_mapping(metadata.get("camera_settings"))),
            "runtime_settings_json": _json(_mapping(metadata.get("runtime_settings"))),
        }

    def create_session(self, metadata: Mapping[str, object] | None = None, *, session_id: str | None = None) -> str:
        """Create one durable configuration snapshot and return its opaque ID."""

        values = self._session_columns(_mapping(metadata))
        identifier = session_id or uuid4().hex
        now = _now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO inspection_sessions (
                    session_id, started_at, ended_at, git_commit, model1_path,
                    model1_sha256, model2_path, model2_sha256, tracker_backend,
                    grading_config_json, grading_config_hash, camera_settings_json,
                    runtime_settings_json, created_at, updated_at
                ) VALUES (
                    :session_id, :started_at, :ended_at, :git_commit, :model1_path,
                    :model1_sha256, :model2_path, :model2_sha256, :tracker_backend,
                    :grading_config_json, :grading_config_hash, :camera_settings_json,
                    :runtime_settings_json, :created_at, :updated_at
                )
                """,
                {"session_id": identifier, **values, "created_at": now, "updated_at": now},
            )
        return identifier

    def update_session(self, session_id: str, metadata: Mapping[str, object]) -> None:
        """Refresh known session readbacks without replacing unavailable fields."""

        if not session_id:
            return
        source = _mapping(metadata)
        # A partial camera readback must not erase a useful snapshot captured at
        # start.  Only explicitly supplied metadata members are updated.
        scalar_columns = (
            "ended_at",
            "git_commit",
            "model1_path",
            "model1_sha256",
            "model2_path",
            "model2_sha256",
            "tracker_backend",
            "grading_config_hash",
        )
        assignments: list[str] = []
        parameters: dict[str, object] = {"session_id": session_id, "updated_at": _now()}
        for key in scalar_columns:
            value = source.get(key)
            if value is None:
                continue
            assignments.append(f"{key} = :{key}")
            parameters[key] = value
        if "grading_config" in source:
            grading = _mapping(source.get("grading_config"))
            assignments.append("grading_config_json = :grading_config_json")
            parameters["grading_config_json"] = _json(grading)
            if "grading_config_hash" not in source:
                assignments.append("grading_config_hash = :computed_grading_config_hash")
                parameters["computed_grading_config_hash"] = grading_config_hash(grading)
        if "camera_settings" in source:
            assignments.append("camera_settings_json = :camera_settings_json")
            parameters["camera_settings_json"] = _json(_mapping(source.get("camera_settings")))
        if "runtime_settings" in source:
            assignments.append("runtime_settings_json = :runtime_settings_json")
            parameters["runtime_settings_json"] = _json(_mapping(source.get("runtime_settings")))
        if not assignments:
            return
        assignments.append("updated_at = :updated_at")
        with self._lock, self._connection() as connection:
            connection.execute(
                f"UPDATE inspection_sessions SET {', '.join(assignments)} WHERE session_id = :session_id",
                parameters,
            )

    def end_session(self, session_id: str, *, ended_at: str | None = None) -> None:
        if not session_id:
            return
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE inspection_sessions SET ended_at = COALESCE(ended_at, :ended_at), updated_at = :updated_at WHERE session_id = :session_id",
                {"session_id": session_id, "ended_at": ended_at or _now(), "updated_at": _now()},
            )

    @staticmethod
    def _event_columns(event: Mapping[str, object]) -> dict[str, object]:
        analysis = _mapping(event.get("analysis"))
        quality = event.get("quality") or analysis.get("final_grade") or "Ungraded"
        weighted_scores = _mapping(analysis.get("weighted_scores"))
        part_results = _mapping(analysis.get("part_results"))
        colors = _mapping(analysis.get("color"))
        observed = analysis.get("observed_regions")
        observed_regions = list(observed) if isinstance(observed, (list, tuple)) else []
        reason_values = analysis.get("reason_codes")
        if isinstance(reason_values, (list, tuple)):
            reason_codes = [str(value) for value in reason_values if value]
        elif analysis.get("verdict_reason_code"):
            reason_codes = [str(analysis["verdict_reason_code"])]
        else:
            reason_codes = []
        model1 = _mapping(analysis.get("model1_detection"))
        final_support = _finite(analysis.get("final_score"))
        if final_support is None:
            final_support = _finite(event.get("quality_confidence"))
        return {
            "timestamp": str(event.get("timestamp") or _now()),
            "final_grade": str(quality),
            "final_support": final_support,
            "second_grade": analysis.get("second_grade"),
            "second_support": _finite(analysis.get("second_support")),
            "grade_margin": _finite(analysis.get("grade_margin")),
            "evidence_coverage": _finite(analysis.get("original_weight_coverage", analysis.get("evidence_coverage"))),
            "observed_regions_json": _json(observed_regions),
            "reason_codes_json": _json(reason_codes),
            "model1_confidence": _finite(model1.get("confidence", event.get("final_confidence"))),
            "weighted_scores_json": _json(weighted_scores),
            "part_results_json": _json(part_results),
            "color_measurements_json": _json(colors),
            "analysis_json": _json(analysis),
            "processing_time_ms": _finite(event.get("processing_time_ms")),
            "event_json": _json(dict(event)),
        }

    def upsert_event(self, session_id: str, event: Mapping[str, object]) -> None:
        """Persist a completed event or its late grade/review update exactly once."""

        if not session_id:
            raise ValueError("A durable inspection session is required before storing an event.")
        try:
            track_id = int(event["track_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A persisted inspection event requires an integer track_id.") from exc
        values = self._event_columns(event)
        now = _now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO fish_inspections (
                    session_id, track_id, timestamp, final_grade, final_support,
                    second_grade, second_support, grade_margin, evidence_coverage,
                    observed_regions_json, reason_codes_json, model1_confidence,
                    weighted_scores_json, part_results_json, color_measurements_json,
                    analysis_json, processing_time_ms, event_json, updated_at
                ) VALUES (
                    :session_id, :track_id, :timestamp, :final_grade, :final_support,
                    :second_grade, :second_support, :grade_margin, :evidence_coverage,
                    :observed_regions_json, :reason_codes_json, :model1_confidence,
                    :weighted_scores_json, :part_results_json, :color_measurements_json,
                    :analysis_json, :processing_time_ms, :event_json, :updated_at
                ) ON CONFLICT(session_id, track_id) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    final_grade = excluded.final_grade,
                    final_support = excluded.final_support,
                    second_grade = excluded.second_grade,
                    second_support = excluded.second_support,
                    grade_margin = excluded.grade_margin,
                    evidence_coverage = excluded.evidence_coverage,
                    observed_regions_json = excluded.observed_regions_json,
                    reason_codes_json = excluded.reason_codes_json,
                    model1_confidence = excluded.model1_confidence,
                    weighted_scores_json = excluded.weighted_scores_json,
                    part_results_json = excluded.part_results_json,
                    color_measurements_json = excluded.color_measurements_json,
                    analysis_json = excluded.analysis_json,
                    processing_time_ms = excluded.processing_time_ms,
                    event_json = excluded.event_json,
                    updated_at = excluded.updated_at
                """,
                {"session_id": session_id, "track_id": track_id, **values, "updated_at": now},
            )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict[str, object]:
        event = _decoded(row["event_json"], {})
        result = dict(event) if isinstance(event, Mapping) else {}
        result["session_id"] = row["session_id"]
        result.setdefault("track_id", row["track_id"])
        result.setdefault("timestamp", row["timestamp"])
        if not result.get("quality"):
            result["quality"] = row["final_grade"]
        if not isinstance(result.get("analysis"), Mapping):
            result["analysis"] = _decoded(row["analysis_json"], {})
        return result

    def events(
        self,
        *,
        session_id: str | None = None,
        range_name: str = "all",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, object]]:
        """Read durable events for all, one session, today, or a date range."""

        if range_name not in {"all", "current_session", "today", "custom"}:
            raise ValueError("Unsupported history range.")
        if range_name == "current_session" and not session_id:
            return []
        try:
            lower = date.fromisoformat(start_date) if range_name == "custom" and start_date else None
            upper = date.fromisoformat(end_date) if range_name == "custom" and end_date else None
        except ValueError as exc:
            raise ValueError("Custom history dates must use YYYY-MM-DD.") from exc
        if lower and upper and lower > upper:
            raise ValueError("The start date must not be after the end date.")
        clauses: list[str] = []
        params: dict[str, object] = {}
        if session_id:
            clauses.append("session_id = :session_id")
            params["session_id"] = session_id
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM fish_inspections" + where + " ORDER BY timestamp DESC, updated_at DESC",
                params,
            ).fetchall()
        values = [self._event_from_row(row) for row in rows]
        if range_name == "all" or range_name == "current_session":
            return values
        if range_name == "today":
            today = date.today()
            return [event for event in values if (parsed := _parse_timestamp(event.get("timestamp"))) and parsed.date() == today]
        return [
            event
            for event in values
            if (parsed := _parse_timestamp(event.get("timestamp"))) is not None
            and (lower is None or parsed.date() >= lower)
            and (upper is None or parsed.date() <= upper)
        ]

    def sessions(self, *, limit: int = 100) -> list[dict[str, object]]:
        """Return recent session metadata for session-aware history selectors."""

        bounded = max(1, min(int(limit), 1000))
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM inspection_sessions ORDER BY started_at DESC LIMIT ?", (bounded,)
            ).fetchall()
        values: list[dict[str, object]] = []
        for row in rows:
            values.append(
                {
                    "session_id": row["session_id"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "git_commit": row["git_commit"],
                    "model1_path": row["model1_path"],
                    "model1_sha256": row["model1_sha256"],
                    "model2_path": row["model2_path"],
                    "model2_sha256": row["model2_sha256"],
                    "tracker_backend": row["tracker_backend"],
                    "grading_config": _decoded(row["grading_config_json"], {}),
                    "grading_config_hash": row["grading_config_hash"],
                    "camera_settings": _decoded(row["camera_settings_json"], {}),
                    "runtime_settings": _decoded(row["runtime_settings_json"], {}),
                }
            )
        return values

    def session(self, session_id: str) -> dict[str, object] | None:
        for value in self.sessions(limit=1000):
            if value["session_id"] == session_id:
                return value
        return None

    def apply_manual_review(self, session_id: str, track_id: int, manual_grade: str | None) -> dict[str, object] | None:
        """Persist a review for an event retained after the live process restarts."""

        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM fish_inspections WHERE session_id = ? AND track_id = ?",
                (session_id, int(track_id)),
            ).fetchone()
        if row is None:
            return None
        event = self._event_from_row(row)
        event["manual_override"] = manual_grade is not None
        event["manual_grade"] = manual_grade
        event["review_timestamp"] = _now() if manual_grade is not None else None
        self.upsert_event(session_id, event)
        event["session_id"] = session_id
        return event


class InspectionSessionContext:
    """Keep the current durable session ID coherent across API and camera threads."""

    def __init__(self, database: InspectionDatabase) -> None:
        self.database = database
        self._lock = RLock()
        self._session_id: str | None = None
        self._active = False

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def start(self, metadata: Mapping[str, object]) -> str:
        with self._lock:
            if self._active and self._session_id:
                return self._session_id
            self._session_id = self.database.create_session(metadata)
            self._active = True
            return self._session_id

    def close(self) -> None:
        with self._lock:
            if self._active and self._session_id:
                self.database.end_session(self._session_id)
            self._active = False

    def rotate(self, metadata: Mapping[str, object]) -> str:
        with self._lock:
            if self._active and self._session_id:
                self.database.end_session(self._session_id)
            self._session_id = self.database.create_session(metadata)
            self._active = True
            return self._session_id

    def event_session(self, metadata: Mapping[str, object]) -> str:
        """Use the last session for late updates; create one only when necessary."""

        with self._lock:
            if self._session_id:
                return self._session_id
        return self.start(metadata)


def _parse_timestamp(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
