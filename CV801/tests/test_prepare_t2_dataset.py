import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_t2_dataset import build_split, greedy_match_count, normalized_box


class PrepareT2DatasetTest(unittest.TestCase):
    def test_box_helpers(self):
        self.assertEqual(normalized_box([10, 20, 30, 40], 100, 100), [100, 200, 400, 600])
        self.assertEqual(greedy_match_count([[0, 0, 10, 10]], [[0, 0, 10, 10]], 0.5), 1)

    def test_build_split_changes_only_user_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            t1_root = root / "t1"
            detection_root = root / "detection"
            (t1_root / "manifests").mkdir(parents=True)
            (detection_root / "coco").mkdir(parents=True)
            target = '{"crop":"tomato","objects":[{"object_id":"object_001"}]}'
            sample = {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "<image> analyze"},
                    {"role": "assistant", "content": target},
                ],
                "images": ["datas/cv801_detection_v1/images/train/Fruit/Tomato/a.jpg"],
            }
            manifest = {
                "image_id": "Fruit/Tomato/a",
                "image_path": sample["images"][0],
                "crop": "tomato",
                "object_count": 1,
                "reference_boxes": [[10, 10, 60, 60]],
                "boxes_exposed_to_t1": False,
            }
            (t1_root / "t1_train.jsonl").write_text(json.dumps(sample) + "\n")
            (t1_root / "manifests" / "t1_train.jsonl").write_text(json.dumps(manifest) + "\n")
            coco = {
                "images": [{
                    "id": 7,
                    "file_name": "images/train/Fruit/Tomato/a.jpg",
                    "width": 100,
                    "height": 100,
                }],
                "categories": [{"id": 11, "name": "tomato"}],
                "annotations": [],
            }
            (detection_root / "coco" / "instances_train.json").write_text(json.dumps(coco))
            predictions = root / "predictions.json"
            predictions.write_text(json.dumps([
                {"image_id": 7, "category_id": 11, "bbox": [10, 10, 50, 50], "score": 0.9},
                {"image_id": 7, "category_id": 11, "bbox": [0, 0, 5, 5], "score": 0.1},
            ]))

            samples, manifests, summary = build_split(
                "train", t1_root, detection_root, predictions, 0.25, 20, 0.5
            )
            self.assertEqual(samples[0]["messages"][-1]["content"], target)
            self.assertIn('"proposal_id":"box_001"', samples[0]["messages"][1]["content"])
            self.assertNotIn("reference_boxes", samples[0]["messages"][1]["content"])
            self.assertEqual(len(manifests[0]["predicted_proposals"]), 1)
            self.assertTrue(manifests[0]["boxes_exposed_to_t2"])
            self.assertEqual(summary["reference_coverage"], 1.0)
            self.assertEqual(summary["assistant_targets_changed"], 0)


if __name__ == "__main__":
    unittest.main()
