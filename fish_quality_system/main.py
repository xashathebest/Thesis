"""Command-line entry point for local fish inspection batches."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .config import InspectionConfig
from .database.db import InspectionDatabase
from .models.mock import SimulatedPartsModel, SimulatedSurfaceModel
from .models.model1_surface import LocalSurfaceModel
from .models.model2_parts import LocalPartsModel
from .pipeline.inspection_pipeline import InspectionPipeline, create_batch_id
from .reports.excel_export import export_batch_excel


def _demo_image() -> np.ndarray:
    """Create a textured test frame for the explicitly simulated demonstration."""
    image = np.full((480, 800, 3), 145, dtype=np.uint8)
    for x in range(0, image.shape[1], 24):
        cv2.line(image, (x, 0), (x, image.shape[0]), (135, 135, 135), 1)
    cv2.putText(image, "SIMULATED TEST FRAME", (225, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
    return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local dried tamban inspection pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="One input image")
    source.add_argument("--video", type=Path, help="Conveyor video")
    source.add_argument("--demo", action="store_true", help="Run a clearly simulated image/model demonstration")
    parser.add_argument("--batch-id", help="e.g. BATCH-2026-09-06-001")
    parser.add_argument("--batch-sequence", type=int, default=1)
    parser.add_argument("--database", type=Path, help="Override the default SQLite path")
    parser.add_argument("--output-dir", type=Path, help="Override the default export directory")
    parser.add_argument("--mock-mode", action="store_true", help="Use simulated model output; never for production inspection")
    parser.add_argument("--no-export", action="store_true", help="Do not write the XLSX/CSV after this run")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.demo and not args.mock_mode:
        raise SystemExit("--demo requires --mock-mode because its detections are simulated.")
    config = InspectionConfig()
    if args.database:
        config = replace(config, database_path=args.database)
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    database = InspectionDatabase(config.database_path)
    batch_id = args.batch_id or create_batch_id(args.batch_sequence)
    if args.mock_mode:
        surface_model, parts_model = SimulatedSurfaceModel(), SimulatedPartsModel()
    else:
        surface_model, parts_model = LocalSurfaceModel(config.surface_weights), LocalPartsModel(config.parts_weights)
    pipeline = InspectionPipeline(surface_model, parts_model, database, config, batch_id)
    if args.demo:
        pipeline.process_frame(_demo_image(), frame_index=0)
        records = pipeline.finish_batch()
    elif args.image:
        records = pipeline.process_image(args.image)
    else:
        records = pipeline.process_video(args.video)
    print(f"Finalized {len(records)} unique fish for {batch_id}.")
    summary = database.batch_summary(batch_id)
    print(summary)
    if not args.no_export:
        workbook, csv_file = export_batch_excel(database, batch_id, config.output_dir)
        print(f"Wrote {workbook} and {csv_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
