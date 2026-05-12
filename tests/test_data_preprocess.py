from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data_preprocess import (
    extract_entity_perspective,
    extract_info_from_response,
    extract_matching_info,
    load_hotpot_samples,
    preprocess_hotpot_file,
    refine_for_answer_agent,
    teacher_agent_judge,
)


class DataPreprocessTests(unittest.TestCase):
    def test_preprocess_hotpot_file_formats_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "hotpot.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "_id": "q1",
                            "question": "Who wrote the novel?",
                            "answer": "Author A",
                            "type": "bridge",
                            "level": "easy",
                            "supporting_facts": [["Novel", 0]],
                            "context": [["Novel", ["Author A wrote it.", "It was adapted later."]]],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            processed = preprocess_hotpot_file(input_path)
            samples = load_hotpot_samples(input_path)

        self.assertEqual(processed[0]["id"], "q1")
        self.assertEqual(processed[0]["model_input"]["question"], "Who wrote the novel?")
        self.assertEqual(
            processed[0]["model_input"]["context"],
            "Novel:\n Author A wrote it. It was adapted later.\n",
        )
        self.assertEqual(samples[0].sample_id, "q1")
        self.assertEqual(samples[0].formatted_context, processed[0]["model_input"]["context"])

    def test_extract_info_and_entity_helpers(self) -> None:
        self.assertEqual(
            extract_info_from_response("<Info>Key facts</Info><Answer>ignored</Answer>"),
            "Key facts",
        )
        self.assertEqual(
            extract_entity_perspective(
                'prefix {"entity": "Melanie Oudin", "perspective": "career"} suffix'
            ),
            ("Melanie Oudin", "career"),
        )

    def test_extract_matching_info_supports_exact_and_fuzzy_match(self) -> None:
        data_string = "Melanie Oudin:\n tennis player\n. Ted Schroeder:\n former champion"
        self.assertEqual(
            extract_matching_info(data_string, "Tell me about Melanie Oudin"),
            "Melanie Oudin:\ntennis player",
        )
        self.assertEqual(
            extract_matching_info(data_string, "Tell me about Ted Schroeeder"),
            "Ted Schroeder:\nformer champion",
        )

    def test_refine_for_answer_agent_uses_updated_prompt_wording(self) -> None:
        prompt = refine_for_answer_agent("Fact block", "What happened?")
        self.assertIn("Key information: Fact block", prompt)
        self.assertIn("Other information:", prompt)

    def test_teacher_agent_judge_is_fail_closed(self) -> None:
        self.assertEqual(teacher_agent_judge("yes"), "YES")
        self.assertEqual(teacher_agent_judge("no"), "NO")
        self.assertEqual(teacher_agent_judge("maybe"), "UNKNOWN")
        self.assertEqual(teacher_agent_judge("yes or no"), "UNKNOWN")
