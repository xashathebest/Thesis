import json
import tempfile
import unittest
from pathlib import Path
from src.evaluation.comparison_protocol import evaluate_boxes, MODELS, PROTOCOL, write_json
from src.evaluation.compare_models import load_result, result_path, generate, FIELDS
from src.preprocessing.dataset_utils import project_root


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.truth = [dict(image='a',boxes=[[0,0,10,10]],labels=[0])]

    def test_perfect_and_missing_classes(self):
        result = evaluate_boxes(self.truth,[dict(image='a',boxes=[[0,0,10,10]],labels=[0],scores=[.9])])
        self.assertEqual(result['overall']['mAP50_95'],1.)
        self.assertIsNone(result['per_class'][1]['mAP50'])

    def test_no_predictions_are_measured_zero(self):
        result = evaluate_boxes(self.truth,[dict(image='a',boxes=[],labels=[],scores=[])])
        self.assertEqual(result['overall']['Recall'],0)
        self.assertEqual(result['errors'][0]['missed_instances'],1)

    def test_duplicate_false_positive(self):
        result = evaluate_boxes(self.truth,[dict(image='a',boxes=[[0,0,10,10]]*2,labels=[0,0],scores=[.9,.8])])
        self.assertEqual(result['overall']['Precision'],.5)
        self.assertEqual(result['overall']['Recall'],1)

    def test_wrong_class(self):
        result = evaluate_boxes(self.truth,[dict(image='a',boxes=[[0,0,10,10]],labels=[1],scores=[.9])])
        self.assertEqual(result['overall']['mAP50'],0)

    def test_membership_guard(self):
        with self.assertRaises(ValueError):
            evaluate_boxes(self.truth,[])

    def test_missing_result_and_path(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(load_result(temp,'yolo','abc'))
            self.assertEqual(result_path(temp,'yolo'),Path(temp)/'yolo/metrics.json')
            with self.assertRaises(ValueError):
                result_path(temp,'../bad')

    def test_reject_unproven_results(self):
        with tempfile.TemporaryDirectory() as temp:
            write_json(result_path(temp,'yolo'),dict(model='yolo',status='completed',metrics={'mAP50':.9}))
            with self.assertRaises(ValueError):
                load_result(temp,'yolo','abc')
            write_json(result_path(temp,'yolo'),dict(model='yolo',status='not_trained',metrics={'mAP50':0}))
            with self.assertRaises(ValueError):
                load_result(temp,'yolo','abc')

    def test_load_measured_zero_and_reject_wrong_split(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = dict(model='yolo',status='completed',protocol=PROTOCOL,evaluation_split='test',
                           dataset_fingerprint='abc',checkpoint_sha256='verified-test-fixture',
                           hardware={'device':'cpu'},timing_definition='test fixture',metrics={'mAP50':0.0})
            write_json(result_path(temp,'yolo'),payload)
            self.assertEqual(load_result(temp,'yolo','abc')['metrics']['mAP50'],0.0)
            payload['evaluation_split'] = 'val'
            write_json(result_path(temp,'yolo'),payload)
            with self.assertRaises(ValueError):
                load_result(temp,'yolo','abc')

    def test_iou_thresholds_and_fixed_confidence(self):
        result = evaluate_boxes(self.truth,[dict(image='a',boxes=[[0,0,10,6]],labels=[0],scores=[.2])])
        self.assertEqual(result['overall']['mAP50'],1.)
        self.assertAlmostEqual(result['overall']['mAP50_95'],.3)
        self.assertEqual(result['overall']['Recall'],0.)

    def test_generation_missing_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            data = project_root()/'dataset/canonical_v7_parts'
            rows = generate(temp,data,{})
            self.assertEqual(len(rows),3)
            self.assertTrue(all(row['mAP50'] is None for row in rows))
            report = (Path(temp)/'MODEL_COMPARISON.md').read_bytes()
            generate(temp,data,{})
            self.assertEqual(report,(Path(temp)/'MODEL_COMPARISON.md').read_bytes())
            self.assertEqual((Path(temp)/'model_comparison.csv').read_text().splitlines()[0],','.join(FIELDS))


if __name__ == '__main__':
    unittest.main()
