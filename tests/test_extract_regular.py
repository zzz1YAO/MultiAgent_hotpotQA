from __future__ import annotations

import unittest

from eval_other_models.extract_regular import extract_answer


class ExtractRegularTests(unittest.TestCase):
    def test_extract_answer_prefers_answer_tag(self) -> None:
        self.assertEqual(extract_answer("<answer>Final result</answer>"), "Final result")

    def test_extract_answer_falls_back_to_answer_prefix(self) -> None:
        text = "answer: concise result You are a helpful assistant"
        self.assertEqual(extract_answer(text), "concise result")

    def test_extract_answer_returns_original_when_no_pattern_matches(self) -> None:
        self.assertEqual(extract_answer("raw output"), "raw output")
