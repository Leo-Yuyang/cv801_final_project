import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_vlm_dataset import build_dataset


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def annotation(image_id: str, annotation_id: str, bbox: list[int], *, valid: bool = True) -> dict:
    return {
        "image_id": image_id,
        "annotation_id": annotation_id,
        "image_type": "fruit",
        "crop": "tomato",
        "bbox": bbox,
        "localization_decision": "accept",
        "ai_analysis_status": "complete",
        "ai_target_valid": valid,
        "ai_caption": {
            "stage": "ripe",
            "evidence": f"Visible evidence for {annotation_id}.",
            "fruit_disease": "none",
            "severity": "none",
            "image_quality": "good",
            "confidence": 0.8,
            "action": "Harvest-ready.",
        },
    }


class PrepareVlmDatasetTest(unittest.TestCase):
    def test_aggregates_objects_and_keeps_golden_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = [
                {
                    "image_id": "train.jpg", "image_type": "fruit", "crop": "tomato",
                    "image_path": "images/train/train.jpg.jpg", "split": "train",
                },
                {
                    "image_id": "val.jpg", "image_type": "fruit", "crop": "tomato",
                    "image_path": "images/val/val.jpg.jpg", "split": "val",
                },
                {
                    "image_id": "test.jpg", "image_type": "fruit", "crop": "tomato",
                    "image_path": "images/test/test.jpg.jpg", "split": "test",
                },
            ]
            annotations = [
                annotation("train.jpg", "right", [100, 100, 200, 200]),
                annotation("train.jpg", "top", [10, 0, 20, 20]),
                annotation("train.jpg", "excluded", [0, 0, 1, 1], valid=False),
                annotation("val.jpg", "val", [0, 0, 20, 20]),
                annotation("test.jpg", "test", [0, 0, 20, 20]),
            ]
            manifest_path = root / "images.jsonl"
            annotations_path = root / "annotations.jsonl"
            output = root / "output"
            write_jsonl(manifest_path, manifest)
            write_jsonl(annotations_path, annotations)

            summary = build_dataset(annotations_path, manifest_path, output, 1, 1)

            train = json.loads((output / "t1_train.jsonl").read_text().strip())
            target = json.loads(train["messages"][-1]["content"])
            self.assertEqual([item["caption"] for item in target["objects"]], [
                "Visible evidence for top.", "Visible evidence for right.",
            ])
            self.assertNotIn("bbox", train["messages"][1]["content"])
            self.assertNotIn("coordinates", train["messages"][-1]["content"])
            self.assertFalse((output / "t1_test.jsonl").exists())
            stress = json.loads((output / "t1_train_stress.jsonl").read_text().strip())
            self.assertEqual(stress["images"], train["images"])
            self.assertEqual(summary["t1"]["train"], {
                "images": 1, "objects": 2, "multi_object_images": 1,
            })
            self.assertEqual(summary["golden_test"]["eligible_images_present_but_not_exported"], 1)
            self.assertEqual(summary["excluded_annotations"]["ineligible_train"], 1)

    def test_rejects_non_frozen_split_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "images.jsonl"
            annotations_path = root / "annotations.jsonl"
            write_jsonl(manifest_path, [{
                "image_id": "train.jpg", "image_type": "fruit", "crop": "tomato",
                "image_path": "images/train/train.jpg.jpg", "split": "train",
            }])
            write_jsonl(annotations_path, [])
            with self.assertRaisesRegex(ValueError, "Expected frozen train/val/test"):
                build_dataset(annotations_path, manifest_path, root / "output", 1, 1)


if __name__ == "__main__":
    unittest.main()
