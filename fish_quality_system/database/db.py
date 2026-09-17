"""SQLite repository with one durable row per tracked fish."""

from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Iterable

from .models import FishInspectionRecord


class InspectionDatabase:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fish_inspections (
                    batch_id TEXT NOT NULL,
                    fish_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    grade TEXT,
                    result TEXT NOT NULL,
                    crack_present INTEGER NOT NULL,
                    crack_count INTEGER NOT NULL,
                    crack_area_px INTEGER NOT NULL,
                    crack_percentage REAL NOT NULL,
                    yellow_present INTEGER NOT NULL,
                    yellow_area_px INTEGER NOT NULL,
                    yellow_percentage REAL NOT NULL,
                    shiny_percentage REAL NOT NULL,
                    dark_discoloration_percentage REAL NOT NULL,
                    fish_length_px REAL NOT NULL,
                    fish_width_px REAL NOT NULL,
                    fish_area_px INTEGER NOT NULL,
                    body_length_px REAL,
                    body_width_px REAL,
                    body_area_px INTEGER,
                    body_width_length_ratio REAL,
                    straightness REAL,
                    curvature REAL,
                    head_status TEXT NOT NULL,
                    body_status TEXT NOT NULL,
                    tail_status TEXT NOT NULL,
                    model1_confidence REAL NOT NULL,
                    model2_confidence REAL NOT NULL,
                    review_reason TEXT,
                    orientation_degrees REAL,
                    wide_or_abnormal_body INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (batch_id, fish_id)
                )
                """
            )

    def save_record(self, record: FishInspectionRecord) -> None:
        row = record.to_row()
        columns = list(row)
        placeholders = ", ".join(f":{column}" for column in columns)
        assignments = ", ".join(f"{column}=excluded.{column}" for column in columns if column not in {"batch_id", "fish_id"})
        sql = f"""
            INSERT INTO fish_inspections ({", ".join(columns)}) VALUES ({placeholders})
            ON CONFLICT(batch_id, fish_id) DO UPDATE SET {assignments}
        """
        with closing(self._connect()) as connection, connection:
            connection.execute(sql, row)

    def records_for_batch(self, batch_id: str) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM fish_inspections WHERE batch_id = ? ORDER BY fish_id", (batch_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def summarize_rows(rows: Iterable[dict[str, object]]) -> dict[str, float | int]:
        records = list(rows)
        total = len(records)
        grades = Counter(record["grade"] for record in records)
        results = Counter(record["result"] for record in records)
        cracked = sum(bool(record["crack_present"]) for record in records)
        yellow = sum(bool(record["yellow_present"]) for record in records)
        wide = sum(bool(record["wide_or_abnormal_body"]) for record in records)
        accepted = results["ACCEPT"]
        rejected = results["REJECT"]
        return {
            "total_unique_fish": total,
            "grade_a_count": grades["A"],
            "grade_b_count": grades["B"],
            "grade_c_count": grades["C"],
            "grade_d_count": grades["D"],
            "cracked_fish_count": cracked,
            "yellow_fish_count": yellow,
            "wide_or_abnormal_body_count": wide,
            "rejected_count": rejected,
            "needs_review_count": results["NEEDS_REVIEW"],
            "acceptance_rate": 0.0 if total == 0 else 100.0 * accepted / total,
            "rejection_rate": 0.0 if total == 0 else 100.0 * rejected / total,
        }

    def batch_summary(self, batch_id: str) -> dict[str, float | int]:
        return self.summarize_rows(self.records_for_batch(batch_id))
