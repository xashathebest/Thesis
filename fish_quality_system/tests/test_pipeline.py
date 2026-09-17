from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from fish_quality_system.config import InspectionConfig
from fish_quality_system.database.db import InspectionDatabase
from fish_quality_system.grading.grader import Grade, Result
from fish_quality_system.models.mock import SimulatedPartsModel, SimulatedSurfaceModel
from fish_quality_system.pipeline.inspection_pipeline import InspectionPipeline


class PipelineTests(unittest.TestCase):
    def test_simulated_frames_create_one_persistent_fish_record(self) -> None:
        # Textured frame passes the real image-quality gate; only model output is simulated.
        generator = np.random.default_rng(7)
        image = generator.integers(80, 180, size=(180, 300, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            config = replace(InspectionConfig(), database_path=Path(directory) / "inspection.sqlite")
            database = InspectionDatabase(config.database_path)
            pipeline = InspectionPipeline(SimulatedSurfaceModel(), SimulatedPartsModel(), database, config, "BATCH-TEST-001")
            self.assertEqual(pipeline.process_frame(image, frame_index=0), [])
            self.assertEqual(pipeline.process_frame(image, frame_index=1), [])
            records = pipeline.finish_batch()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].fish_id, "F0001")
            self.assertEqual((records[0].grade, records[0].result), (Grade.A, Result.ACCEPT))
            self.assertEqual(database.batch_summary("BATCH-TEST-001")["total_unique_fish"], 1)


if __name__ == "__main__":
    unittest.main()
