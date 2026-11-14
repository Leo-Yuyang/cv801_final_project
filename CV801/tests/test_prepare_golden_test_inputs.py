import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_golden_test_inputs import (
    audit_inputs,
    audit_writer_predictions,
    build_base,
    build_t2,
)


class PrepareGoldenTestInputsTest(unittest.TestCase):
    def _base_args(self, root: Path) -> argparse.Namespace:
        detection_root = root / "detection"
        rows = []
        for index, crop in enumerate(("tomato", "lemon"), 1):
            relative = f"images/test/Fruit/{crop}/{index}.jpg"
            image = detection_root / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{index}".encode())
            rows.append({
                "image_id": f"Fruit/{crop}/{index}",
                "image_path": relative,
                "split": "test",
                "crop": crop,
                "image_type": "fruit",
                "width": 100,
                "height": 200,
                "file_sha256": f"sha-{index}",
                "object_count": 99,
                "label_path": "MUST_NOT_LEAK",
            })
        manifest = root / "images.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        classes = root / "classes.json"
        classes.write_text(json.dumps({"names": ["lemon", "tomato"]}), encoding="utf-8")
        return argparse.Namespace(
            image_manifest=manifest,
            classes=classes,
            detection_root=detection_root,
            output_root=root / "output",
            expected_images=2,
        )

    def test_base_and_t2_inputs_are_reference_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_args = self._base_args(root)
            summary = build_base(base_args)
            self.assertEqual(summary["images"], 2)
            output = base_args.output_root
            serialized = (output / "writer_t0_t1.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("MUST_NOT_LEAK", serialized)
            self.assertNotIn("object_count", serialized)
            writer = json.loads(serialized.splitlines()[0])
            self.assertEqual([message["role"] for message in writer["messages"]], ["system", "user"])
            detector = json.loads((output / "detector_input.coco.json").read_text())
            self.assertNotIn("annotations", detector)

            source_rows = [json.loads(line) for line in serialized.splitlines()]
            writer_predictions = root / "writer_predictions.jsonl"
            writer_predictions.write_text("".join(
                json.dumps({
                    "messages": row["messages"],
                    "images": [{"path": row["images"][0]}],
                    "labels": None,
                    "response": "{}",
                }) + "\n"
                for row in source_rows
            ))
            prediction_audit = audit_writer_predictions(
                writer_predictions, output, "t0_t1", expected_images=2
            )
            self.assertTrue(prediction_audit["aligned"])

            appended_rows = []
            for row in source_rows:
                response = '{"ok": true}'
                appended_rows.append({
                    "messages": row["messages"] + [{"role": "assistant", "content": response}],
                    "images": [{"bytes": None, "path": row["images"][0]}],
                    "labels": None,
                    "response": response,
                })
            writer_predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in appended_rows), encoding="utf-8"
            )
            prediction_audit = audit_writer_predictions(
                writer_predictions, output, "t0_t1", expected_images=2
            )
            self.assertTrue(prediction_audit["aligned"])

            appended_rows[0]["messages"][0]["content"] = "TAMPERED"
            writer_predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in appended_rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "source messages changed"):
                audit_writer_predictions(writer_predictions, output, "t0_t1", expected_images=2)

            predictions = root / "predictions.json"
            predictions.write_text(json.dumps([
                {"image_id": 1, "category_id": 2, "bbox": [10, 20, 30, 40], "score": 0.9},
                {"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.1},
            ]))
            model = root / "best.pt"
            model.write_bytes(b"frozen-model")
            t2_args = argparse.Namespace(
                input_root=output,
                predictions=predictions,
                locator_model=model,
                score_threshold=0.25,
                max_proposals=20,
                expected_images=2,
            )
            result = build_t2(t2_args)
            self.assertEqual(result["empty_proposal_images"], 1)
            t2_writer = json.loads((output / "writer_t2.jsonl").read_text().splitlines()[0])
            self.assertEqual([message["role"] for message in t2_writer["messages"]], ["system", "user"])
            self.assertIn("Predicted localization hints", t2_writer["messages"][-1]["content"])
            self.assertTrue(audit_inputs(output, 2, require_t2=True)["t2_reference_free"])

    def test_audit_rejects_assistant_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._base_args(root)
            build_base(args)
            path = args.output_root / "writer_t0_t1.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["messages"].append({"role": "assistant", "content": "HIDDEN_REFERENCE"})
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "not reference-free"):
                audit_inputs(args.output_root, 2, require_t2=False)


if __name__ == "__main__":
    unittest.main()
