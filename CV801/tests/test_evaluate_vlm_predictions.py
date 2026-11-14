import json
import unittest

from scripts.evaluate_vlm_predictions import evaluate


class EvaluateVlmPredictionsTest(unittest.TestCase):
    def test_strips_think_and_scores_structured_output(self):
        target = {
            "crop": "tomato", "plant_part": "fruit",
            "objects": [{
                "object_id": "object_001", "caption": "red ripe fruit",
                "assessment": {"stage": "ripe", "disease": "none", "severity": "none"},
                "image_quality": "good", "confidence": 0.8, "action": "harvest now",
            }],
        }
        row = {
            "labels": json.dumps(target),
            "response": "<think>\n\n</think>\n" + json.dumps(target),
        }
        result = evaluate([row])
        self.assertEqual(result["json_valid_rate"], 1.0)
        self.assertEqual(result["schema_valid_rate"], 1.0)
        self.assertEqual(result["object_count_recall"], 1.0)
        self.assertEqual(result["categorical_exact"]["disease"], 1.0)
        self.assertEqual(result["object_id_link_f1"], 1.0)
        self.assertEqual(result["caption_rouge_l_f1"], 1.0)

    def test_invalid_schema_remains_json_valid(self):
        target = {"crop": "tomato", "plant_part": "leaf", "objects": []}
        result = evaluate([{"labels": json.dumps(target), "response": json.dumps(target)}])
        self.assertEqual(result["json_valid_rate"], 1.0)
        self.assertEqual(result["schema_valid_rate"], 0.0)

    def test_invalid_json_counts_all_reference_objects(self):
        target = {
            "crop": "tomato", "plant_part": "leaf",
            "objects": [{"object_id": "object_001"}, {"object_id": "object_002"}],
        }
        result = evaluate([{"labels": json.dumps(target), "response": "{truncated"}])
        self.assertEqual(result["json_valid_rate"], 0.0)
        self.assertEqual(result["counts"]["reference_objects"], 2)
        self.assertEqual(result["object_count_recall"], 0.0)
        self.assertEqual(result["object_count_mae"], 2.0)

    def test_caption_metrics_match_by_object_id_not_list_order(self):
        target = {
            "crop": "tomato", "plant_part": "fruit",
            "objects": [
                {
                    "object_id": "object_001", "caption": "bright red fruit on green stem",
                    "assessment": {"stage": "ripe", "disease": "none", "severity": "none"},
                    "image_quality": "good", "confidence": 0.9, "action": "harvest",
                },
                {
                    "object_id": "object_002", "caption": "small green fruit beside dry leaf",
                    "assessment": {"stage": "unripe", "disease": "none", "severity": "none"},
                    "image_quality": "good", "confidence": 0.8, "action": "monitor",
                },
            ],
        }
        prediction = dict(target)
        prediction["objects"] = list(reversed(target["objects"]))
        result = evaluate([{"labels": json.dumps(target), "response": json.dumps(prediction)}])
        self.assertEqual(result["caption_bleu4_corpus"], 1.0)
        self.assertEqual(result["caption_rouge_l_f1"], 1.0)
        self.assertEqual(result["object_id_link_precision"], 1.0)
        self.assertEqual(result["object_id_set_exact_rate"], 1.0)

    def test_extra_object_and_duplicate_id_are_only_structural_proxies(self):
        target_object = {
            "object_id": "object_001", "caption": "bright red fruit on green stem",
            "assessment": {"stage": "ripe", "disease": "none", "severity": "none"},
            "image_quality": "good", "confidence": 0.9, "action": "harvest",
        }
        target = {"crop": "tomato", "plant_part": "fruit", "objects": [target_object]}
        prediction = {"crop": "tomato", "plant_part": "fruit", "objects": [target_object, target_object]}
        result = evaluate([{"labels": json.dumps(target), "response": json.dumps(prediction)}])
        self.assertEqual(result["extra_object_hallucination_proxy_rate"], 0.5)
        self.assertEqual(result["samples_with_extra_object_rate"], 1.0)
        self.assertEqual(result["object_id_link_precision"], 0.5)
        self.assertEqual(result["object_id_link_recall"], 1.0)
        self.assertEqual(result["duplicate_object_id_rate"], 0.5)
        self.assertEqual(result["object_id_set_exact_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
