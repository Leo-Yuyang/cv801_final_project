import unittest

from CV801.scripts.evaluate_coco_predictions import (
    bbox_iou_xywh,
    best_f1_threshold,
    greedy_match_outcomes,
    summarize_operating_point,
)


class DetectionOperatingPointTests(unittest.TestCase):
    def setUp(self):
        self.annotations = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10]},
            {"image_id": 2, "category_id": 2, "bbox": [0, 0, 5, 5]},
        ]

    def test_bbox_iou_xywh(self):
        self.assertAlmostEqual(bbox_iou_xywh([0, 0, 10, 10], [5, 0, 10, 10]), 1 / 3)

    def test_frozen_operating_point(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
            {"image_id": 2, "category_id": 2, "bbox": [0, 0, 5, 5], "score": 0.7},
            {"image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10], "score": 0.4},
        ]
        outcomes, ground_truth_count = greedy_match_outcomes(self.annotations, predictions)
        result = summarize_operating_point(outcomes, ground_truth_count, score_threshold=0.5)
        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["mean_matched_iou"], 1.0)

    def test_best_threshold_includes_equal_score_group(self):
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
            {"image_id": 1, "category_id": 1, "bbox": [20, 20, 10, 10], "score": 0.7},
            {"image_id": 2, "category_id": 2, "bbox": [0, 0, 5, 5], "score": 0.7},
        ]
        result = best_f1_threshold(self.annotations, predictions)
        self.assertAlmostEqual(result["score_threshold"], 0.7)
        self.assertEqual(result["true_positives"], 3)
        self.assertEqual(result["false_positives"], 1)
        self.assertAlmostEqual(result["f1"], 6 / 7)


if __name__ == "__main__":
    unittest.main()
