import json
import tempfile
import unittest
from pathlib import Path

from scripts.attach_vlm_eval_labels import attach_labels
from scripts.prepare_agent_review_inputs import (
    prepare_judge,
    prepare_recheck,
    prepare_reviewers,
    prepare_revision,
    prepare_revision_round2,
)


class PrepareAgentReviewInputsTest(unittest.TestCase):
    def test_smoke_labels_are_attached_only_after_alignment_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.jsonl"
            references = root / "references.jsonl"
            output = root / "with_labels.jsonl"
            predictions.write_text(
                json.dumps({"response": "{}", "images": ["a.jpg"], "labels": None}) + "\n",
                encoding="utf-8",
            )
            references.write_text(
                "\n".join([
                    json.dumps({"labels": "A", "images": [{"path": "a.jpg"}]}),
                    json.dumps({"labels": "B", "images": [{"path": "b.jpg"}]}),
                ]) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(attach_labels(predictions, references, output, max_samples=1), 1)
            attached = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(attached["labels"], "A")
            self.assertTrue(attached["labels_attached_post_inference"])

    def test_references_do_not_enter_review_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer_path = root / "writer.jsonl"
            writer_path.write_text(json.dumps({
                "response": '<think>private</think>{"objects":[]}',
                "labels": "HIDDEN_REFERENCE",
                "messages": [
                    {"role": "system", "content": "Return JSON."},
                    {"role": "user", "content": "<image> Analyze the visible object."},
                ],
                "images": [{"path": "image.jpg"}],
            }) + "\n")
            review_dir = root / "review"
            prepare_reviewers(writer_path, review_dir)

            visual_text = (review_dir / "visual_reviewer_inputs.jsonl").read_text()
            domain_text = (review_dir / "domain_reviewer_inputs.jsonl").read_text()
            manifest_text = (review_dir / "review_manifest.jsonl").read_text()
            self.assertNotIn("HIDDEN_REFERENCE", visual_text + domain_text + manifest_text)
            self.assertNotIn("private", visual_text + domain_text + manifest_text)
            self.assertIn('{\\"objects\\":[]}', visual_text)

            visual_predictions = root / "visual_predictions.jsonl"
            domain_predictions = root / "domain_predictions.jsonl"
            visual_predictions.write_text(json.dumps({"response": '{"verdict":"pass"}'}) + "\n")
            domain_predictions.write_text(json.dumps({"response": '{"verdict":"revise"}'}) + "\n")
            judge_path = root / "judge.jsonl"
            prepare_judge(
                review_dir / "review_manifest.jsonl",
                visual_predictions,
                domain_predictions,
                judge_path,
            )
            judge = json.loads(judge_path.read_text())
            self.assertNotIn("images", judge)
            self.assertNotIn("HIDDEN_REFERENCE", judge_path.read_text())
            self.assertIn("VISUAL_REVIEW", judge["messages"][1]["content"])

            revision_path = root / "revision.jsonl"
            prepare_revision(
                writer_path,
                "judge",
                revision_path,
                judge_predictions=judge_path,
            )
            revision = json.loads(revision_path.read_text())
            self.assertNotIn("HIDDEN_REFERENCE", revision_path.read_text())
            self.assertIn("JUDGE_FEEDBACK", revision["messages"][-1]["content"])
            self.assertEqual(revision["images"], ["image.jpg"])

            revised_predictions = root / "revised_predictions.jsonl"
            revised_predictions.write_text(json.dumps({
                "response": '{"objects":[]}',
                "messages": revision["messages"] + [
                    {"role": "assistant", "content": '{"objects":[]}'},
                ],
                "images": revision["images"],
                "labels": "HIDDEN_REFERENCE",
            }) + "\n")
            recheck_path = root / "recheck.jsonl"
            prepare_recheck(
                review_dir / "review_manifest.jsonl",
                visual_predictions,
                domain_predictions,
                judge_path,
                revised_predictions,
                recheck_path,
            )
            recheck = json.loads(recheck_path.read_text())
            self.assertNotIn("images", recheck)
            self.assertNotIn("HIDDEN_REFERENCE", recheck_path.read_text())
            self.assertIn("REVISED_RESPONSE", recheck["messages"][1]["content"])

            recheck_predictions = root / "recheck_predictions.jsonl"
            recheck_predictions.write_text(json.dumps({
                "response": '{"decision":"revise_again","revision_feedback":["Be conservative."]}'
            }) + "\n")
            revision_round2_path = root / "revision_round2.jsonl"
            prepare_revision_round2(
                revised_predictions,
                recheck_predictions,
                revision_round2_path,
            )
            revision_round2 = json.loads(revision_round2_path.read_text())
            self.assertNotIn("HIDDEN_REFERENCE", revision_round2_path.read_text())
            self.assertEqual(revision_round2["messages"][-1]["role"], "user")
            self.assertIn("RECHECK_FEEDBACK", revision_round2["messages"][-1]["content"])
            self.assertEqual(revision_round2["images"], ["image.jpg"])


if __name__ == "__main__":
    unittest.main()
