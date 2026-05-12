from __future__ import annotations

import unittest

from src.workflow_runner import WorkflowConfig, WorkflowRunner, WorkflowRuntime
from src.workflow_schema import ContextDocument, QuestionSample


class ScriptedGenerator:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def __call__(self, prompt: str) -> str:
        if not self._responses:
            raise AssertionError(f"No scripted response left for prompt: {prompt}")
        return self._responses.pop(0)


class ConstantGenerator:
    def __init__(self, response: str) -> None:
        self.response = response

    def __call__(self, prompt: str) -> str:
        return self.response


class AgenticWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = QuestionSample(
            sample_id="q1",
            question="What is Melanie Oudin known for?",
            answer="tennis player",
            question_type="bridge",
            level="easy",
            supporting_facts=(("Melanie Oudin", 0),),
            context_documents=(
                ContextDocument(title="Melanie Oudin", sentences=("Melanie Oudin is a tennis player.",)),
                ContextDocument(title="Ted Schroeder", sentences=("Ted Schroeder is a former champion.",)),
            ),
        )

    def test_runner_executes_complete_agentic_loop(self) -> None:
        controller = ScriptedGenerator(
            [
                '{"action": "decompose_question", "arguments": {}, "reason": "split the task"}',
                '{"action": "retrieve_passages", "arguments": {"query": "Melanie Oudin"}, "reason": "find relevant passages"}',
                '{"action": "extract_evidence", "arguments": {}, "reason": "collect evidence"}',
                '{"action": "verify_evidence", "arguments": {}, "reason": "check sufficiency"}',
                '{"action": "stop_and_answer", "arguments": {}, "reason": "answer now"}',
            ]
        )
        evidence = ScriptedGenerator(
            [
                '{"subquestions": ["Who is Melanie Oudin?"], "entities": ["Melanie Oudin"]}',
                '{"evidence": [{"title": "Melanie Oudin", "content": "Melanie Oudin is a tennis player."}]}',
            ]
        )
        answer = ConstantGenerator("Melanie Oudin is known for being a tennis player.")
        verifier = ConstantGenerator(
            '{"status": "SUFFICIENT", "reason": "The evidence directly answers the question.", "missing_items": [], "confidence": 0.95}'
        )

        runtime = WorkflowRuntime(
            controller=controller,
            evidence=evidence,
            answer=answer,
            verifier=verifier,
        )
        runner = WorkflowRunner(runtime, WorkflowConfig(max_steps=6))
        result = runner.run(self.sample)

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.verification_status, "SUFFICIENT")
        self.assertEqual(result.final_answer, "Melanie Oudin is known for being a tennis player.")
        self.assertEqual([step.action for step in result.trace], [
            "decompose_question",
            "retrieve_passages",
            "extract_evidence",
            "verify_evidence",
            "stop_and_answer",
        ])
        self.assertEqual(result.evidence[0].title, "Melanie Oudin")

    def test_runner_falls_back_after_controller_error(self) -> None:
        controller = ScriptedGenerator(
            ['{"action": "unsupported_action", "arguments": {}, "reason": "broken"}']
        )
        runtime = WorkflowRuntime(
            controller=controller,
            evidence=ConstantGenerator("{}"),
            answer=ConstantGenerator("fallback answer"),
            verifier=ConstantGenerator('{"status": "UNKNOWN", "reason": "n/a", "missing_items": []}'),
        )
        runner = WorkflowRunner(runtime, WorkflowConfig(max_steps=3))
        result = runner.run(self.sample)

        self.assertEqual(result.status, "INTERRUPTED")
        self.assertEqual(result.final_answer, "fallback answer")
        self.assertEqual(result.trace[0].action, "controller_error")
