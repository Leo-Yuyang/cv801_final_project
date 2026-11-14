import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_golden_vlm_references import export_references


class PrepareGoldenVlmReferencesTest(unittest.TestCase):
    def test_exports_in_safe_manifest_order_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_manifest = root / "images.jsonl"
            annotations = root / "annotations.jsonl"
            safe_manifest = root / "safe.jsonl"
            output = root / "references.jsonl"
            image_manifest.write_text(json.dumps({
                "image_id": "fruit/a.jpg",
                "image_path": "images/test/a.jpg",
                "image_type": "fruit",
                "crop": "tomato",
                "split": "test",
            }) + "\n")
            annotations.write_text(json.dumps({
                "image_id": "fruit/a.jpg",
                "image_type": "fruit",
                "crop": "tomato",
                "bbox": [0, 0, 10, 10],
                "annotation_id": "ann-1",
                "localization_decision": "accept",
                "ai_analysis_status": "complete",
                "ai_target_valid": True,
                "ai_caption": {
                    "evidence": "A red fruit is visible.",
                    "stage": "ripe",
                    "fruit_disease": "none",
                    "severity": "none",
                    "image_quality": "good",
                    "confidence": 0.9,
                    "action": "Harvest when needed.",
                },
            }) + "\n")
            safe_manifest.write_text(json.dumps({
                "sample_id": "sample-a",
                "image_id": "fruit/a.jpg",
                "image_path": "datas/cv801_detection_v1/images/test/a.jpg",
            }) + "\n")

            result = export_references(
                annotations,
                image_manifest,
                safe_manifest,
                output,
                expected_images=1,
                expected_objects=1,
            )
            self.assertEqual(result, {"images": 1, "objects": 1})
            record = json.loads(output.read_text())
            self.assertEqual(record["sample_id"], "sample-a")
            self.assertEqual(record["object_count"], 1)
            self.assertIn('"object_id":"object_001"', record["labels"])
            with self.assertRaises(FileExistsError):
                export_references(
                    annotations,
                    image_manifest,
                    safe_manifest,
                    output,
                    expected_images=1,
                    expected_objects=1,
                )


if __name__ == "__main__":
    unittest.main()
