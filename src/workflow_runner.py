from __future__ import annotations

from dataclasses import dataclass

try:
    from src.workflow_actions import (
        LLMController,
        LLMVerifier,
        TextGenerator,
        build_budget_fallback_answer,
        decompose_question,
        expand_entity,
        extract_evidence,
        retrieve_passages,
        stop_and_answer,
    )
    from src.workflow_schema import (
        AgentObservation,
        AgentResult,
        AgentState,
        AgentTraceStep,
        QuestionSample,
        VerifierResult,
    )
except ImportError:
    from workflow_actions import (
        LLMController,
        LLMVerifier,
        TextGenerator,
        build_budget_fallback_answer,
        decompose_question,
        expand_entity,
        extract_evidence,
        retrieve_passages,
        stop_and_answer,
    )
    from workflow_schema import (
        AgentObservation,
        AgentResult,
        AgentState,
        AgentTraceStep,
        QuestionSample,
        VerifierResult,
    )


@dataclass(frozen=True)
class WorkflowRuntime:
    controller: TextGenerator
    evidence: TextGenerator
    answer: TextGenerator
    verifier: TextGenerator


@dataclass(frozen=True)
class WorkflowConfig:
    max_steps: int = 6


class WorkflowRunner:
    def __init__(self, runtime: WorkflowRuntime, config: WorkflowConfig | None = None) -> None:
        self.runtime = runtime
        self.config = config or WorkflowConfig()
        self.controller = LLMController(runtime.controller)
        self.verifier = LLMVerifier(runtime.verifier)

    def run(self, sample: QuestionSample) -> AgentResult:
        state = AgentState.from_sample(sample, step_budget=self.config.max_steps)

        for step_index in range(self.config.max_steps):
            try:
                action = self.controller.select_action(state)
                observation, verification = self._execute_action(state, action.name, action.arguments)
                state.trace.append(
                    AgentTraceStep(
                        step_index=step_index,
                        action=action.name,
                        arguments=action.arguments,
                        reason=action.reason,
                        observation_summary=observation.summary,
                        verification_status=verification.status,
                        verification_reason=verification.reason,
                    )
                )
                state.verification_status = verification.status
                state.verification_reason = verification.reason

                if action.name == "stop_and_answer" and state.final_answer:
                    return self._build_result(state, status="COMPLETED")
            except Exception as exc:
                state.trace.append(
                    AgentTraceStep(
                        step_index=step_index,
                        action="controller_error",
                        arguments={},
                        reason="controller or action execution failed",
                        observation_summary=str(exc),
                        verification_status=state.verification_status,
                        verification_reason=state.verification_reason,
                    )
                )
                break

        state.final_answer = build_budget_fallback_answer(state, self.runtime.answer)
        if not state.draft_answer:
            state.draft_answer = state.final_answer
        final_status = "BUDGET_EXHAUSTED" if len(state.trace) >= self.config.max_steps else "INTERRUPTED"
        return self._build_result(state, status=final_status)

    def _execute_action(
        self,
        state: AgentState,
        action_name: str,
        arguments: dict[str, object],
    ) -> tuple[AgentObservation, VerifierResult]:
        if action_name == "decompose_question":
            observation = decompose_question(state, self.runtime.evidence)
            verification = VerifierResult(status=state.verification_status, reason=state.verification_reason)
        elif action_name == "retrieve_passages":
            observation = retrieve_passages(state, arguments)
            verification = VerifierResult(status=state.verification_status, reason=state.verification_reason)
        elif action_name == "expand_entity":
            observation = expand_entity(state, self.runtime.evidence)
            verification = VerifierResult(status=state.verification_status, reason=state.verification_reason)
        elif action_name == "extract_evidence":
            observation = extract_evidence(state, self.runtime.evidence)
            verification = VerifierResult(status=state.verification_status, reason=state.verification_reason)
        elif action_name == "verify_evidence":
            verification = self.verifier.evaluate(state)
            state.verification_status = verification.status
            state.verification_reason = verification.reason
            observation = AgentObservation(summary=f"Verifier status: {verification.status}")
        elif action_name == "stop_and_answer":
            observation = stop_and_answer(state, self.runtime.answer)
            verification = self.verifier.evaluate(state)
            state.verification_status = verification.status
            state.verification_reason = verification.reason
        else:
            raise ValueError(f"Unknown action: {action_name}")

        return observation, verification

    def _build_result(self, state: AgentState, status: str) -> AgentResult:
        return AgentResult(
            sample_id=state.sample.sample_id,
            final_answer=state.final_answer,
            status=status,
            draft_answer=state.draft_answer,
            verification_status=state.verification_status,
            verification_reason=state.verification_reason,
            trace=tuple(state.trace),
            evidence=tuple(state.evidence_buffer),
        )
