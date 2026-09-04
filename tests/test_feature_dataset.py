from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.training.feature_dataset import load_feature_table


HEADER = "instance_id,specimen_id,batch_id,final_class,split,gradable,occluded,truncated,width_length_ratio,skeleton_converged,head_present\n"


class FeatureTableTests(unittest.TestCase):
    def _write(self, content: str) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "features.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_numeric_and_boolean_features(self) -> None:
        path = self._write(HEADER + "i1,s1,b1,Class A,train,true,false,false,0.2,yes,yes\n" + "i2,s2,b2,Class B,validation,true,false,false,0.3,no,no\n")
        table = load_feature_table(path)
        self.assertEqual(table.feature_names, ["width_length_ratio", "skeleton_converged"])
        self.assertEqual(table.values.tolist(), [[0.2, 1.0], [0.3, 0.0]])

    def test_rejects_specimen_leakage(self) -> None:
        path = self._write(HEADER + "i1,s1,b1,Class A,train,true,false,false,0.2,yes,yes\n" + "i2,s1,b2,Class A,test,true,false,false,0.2,yes,yes\n")
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_feature_table(path)

    def test_excludes_ungradable_instances(self) -> None:
        path = self._write(HEADER + "i1,s1,b1,Class A,train,false,false,false,0.2,yes,yes\n" + "i2,s2,b2,Class B,validation,true,false,false,0.3,no,no\n")
        table = load_feature_table(path)
        self.assertEqual(table.instance_ids.tolist(), ["i2"])

    def test_excludes_occluded_instances(self) -> None:
        path = self._write(HEADER + "i1,s1,b1,Class A,train,true,true,false,0.2,yes,yes\n" + "i2,s2,b2,Class B,validation,true,false,false,0.3,no,no\n")
        table = load_feature_table(path)
        self.assertEqual(table.instance_ids.tolist(), ["i2"])

    def test_rejects_batch_leakage_even_when_specimens_differ(self) -> None:
        path = self._write(HEADER + "i1,s1,b1,Class A,train,true,false,false,0.2,yes,yes\n" + "i2,s2,b1,Class B,test,true,false,false,0.3,no,no\n")
        with self.assertRaisesRegex(ValueError, "batch_id:b1"):
            load_feature_table(path)


if __name__ == "__main__":
    unittest.main()
