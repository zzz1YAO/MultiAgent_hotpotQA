from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hotpot_evaluate_em_f1 import evaluate, f1_score
from src.text_metrics import normalize_answer, token_f1


class TextMetricsTests(unittest.TestCase):
    def test_normalize_answer_and_token_f1(self) -> None:
        self.assertEqual(normalize_answer("The Quick, Brown Fox!"), "quick brown fox")
        self.assertEqual(token_f1("yes", "yes"), 1.0)
        self.assertEqual(round(token_f1("Author A", "Author B"), 4), 0.6667)

    def test_hotpot_evaluate_returns_expected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pred_path = Path(tmp_dir) / "pred.json"
            gold_path = Path(tmp_dir) / "gold.json"

            pred_path.write_text(json.dumps({"q1": "Author A", "q2": "yes"}), encoding="utf-8")
            gold_path.write_text(
                json.dumps(
                    [
                        {"_id": "q1", "answer": "Author A"},
                        {"_id": "q2", "answer": "yes"},
                    ]
                ),
                encoding="utf-8",
            )

            metrics = evaluate(pred_path, gold_path)

        self.assertEqual(metrics["count"], 2.0)
        self.assertEqual(metrics["em"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(f1_score("A writer", "writer"), 1.0)
