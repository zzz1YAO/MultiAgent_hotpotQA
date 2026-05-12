from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

try:
    from src.data_preprocess import extract_matching_info, parse_context_string, teacher_agent_judge
    from src.workflow_schema import (
        AgentAction,
        AgentObservation,
        AgentState,
        EvidenceItem,
        VerifierResult,
    )
    from src.agent_prompts import (
        build_controller_prompt,
        build_decompose_prompt,
        build_expand_entity_prompt,
        build_extract_evidence_prompt,
        build_final_answer_prompt,
        build_verify_prompt,
    )
except ImportError:
    from data_preprocess import extract_matching_info, parse_context_string, teacher_agent_judge
    from workflow_schema import (
        AgentAction,
        AgentObservation,
        AgentState,
        EvidenceItem,
        VerifierResult,
    )
    from agent_prompts import (
        build_controller_prompt,
        build_decompose_prompt,
        build_expand_entity_prompt,
        build_extract_evidence_prompt,
        build_final_answer_prompt,
        build_verify_prompt,
    )


TextGenerator = Callable[[str], str]

DEFAULT_ACTIONS: tuple[str, ...] = (
    "decompose_question",
    "retrieve_passages",
    "expand_entity",
    "extract_evidence",
    "verify_evidence",
    "stop_and_answer",
)


class LLMController:
    def __init__(
        self,
        generator: TextGenerator,
        allowed_actions: Iterable[str] = DEFAULT_ACTIONS,
    ) -> None:
        self.generator = generator
        self.allowed_actions = tuple(allowed_actions)

    def select_action(self, state: AgentState) -> AgentAction:
        response = self.generator(build_controller_prompt(state, self.allowed_actions))
        payload = extract_json_payload(response)
        action_name = str(payload.get("action", "")).strip()
        if action_name not in self.allowed_actions:
            raise ValueError(f"Unsupported action selected by controller: {action_name!r}")

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Controller must return an object under `arguments`.")

        reason = str(payload.get("reason", "")).strip()
        return AgentAction(name=action_name, arguments=arguments, reason=reason)


class LLMVerifier:
    def __init__(self, generator: TextGenerator) -> None:
        self.generator = generator

    def evaluate(self, state: AgentState) -> VerifierResult:
        if not state.evidence_buffer:
            return VerifierResult(
                status="INSUFFICIENT",
                reason="No evidence has been collected yet.",
                missing_items=("supporting evidence",),
                confidence=0.0,
            )

        response = self.generator(build_verify_prompt(state))
        try:
            payload = extract_json_payload(response)
            status = str(payload.get("status", "UNKNOWN")).upper()
            reason = str(payload.get("reason", "")).strip()
            missing_items = payload.get("missing_items", [])
            confidence = payload.get("confidence")
            if not isinstance(missing_items, list):
                missing_items = []
            if status not in {"SUFFICIENT", "INSUFFICIENT", "CONFLICT", "UNKNOWN"}:
                status = "UNKNOWN"
            if confidence is not None:
                confidence = float(confidence)
            return VerifierResult(
                status=status,
                reason=reason,
                missing_items=tuple(str(item) for item in missing_items),
                confidence=confidence,
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            legacy_label = teacher_agent_judge(response)
            fallback_status = {
                "YES": "SUFFICIENT",
                "NO": "INSUFFICIENT",
                "UNKNOWN": "UNKNOWN",
            }[legacy_label]
            return VerifierResult(
                status=fallback_status,
                reason="Verifier returned an unstructured response.",
                confidence=None,
            )


def decompose_question(state: AgentState, generator: TextGenerator) -> AgentObservation:
    response = generator(build_decompose_prompt(state))
    payload = extract_json_payload(response)
    subquestions = [
        str(item).strip()
        for item in payload.get("subquestions", [])
        if str(item).strip()
    ]
    entities = [
        str(item).strip()
        for item in payload.get("entities", [])
        if str(item).strip()
    ]

    if not subquestions:
        subquestions = [state.question]

    _extend_unique(state.subquestions, subquestions)
    _extend_unique(state.candidate_entities, entities)

    return AgentObservation(
        summary=f"Decomposed into {len(subquestions)} subquestion(s).",
        payload={
            "subquestions": subquestions,
            "entities": entities,
        },
    )


def retrieve_passages(state: AgentState, arguments: dict[str, Any]) -> AgentObservation:
    query = str(arguments.get("query", "")).strip()
    if not query:
        query = state.subquestions[-1] if state.subquestions else state.question

    matching_info = extract_matching_info(state.original_context, query)
    matched_passages = parse_context_string(matching_info)
    if not matched_passages:
        matched_passages = state.sample.context_map

    state.candidate_passages.update(matched_passages)
    return AgentObservation(
        summary=f"Retrieved {len(matched_passages)} candidate passage(s) for query: {query}",
        payload={
            "query": query,
            "matched_titles": list(matched_passages),
        },
    )


def expand_entity(state: AgentState, generator: TextGenerator) -> AgentObservation:
    response = generator(build_expand_entity_prompt(state))
    payload = extract_json_payload(response)
    entity = str(payload.get("entity", "")).strip()
    query = str(payload.get("query", entity or state.question)).strip()
    if entity:
        _extend_unique(state.candidate_entities, [entity])

    matching_info = extract_matching_info(state.original_context, query)
    matched_passages = parse_context_string(matching_info)
    if matched_passages:
        state.candidate_passages.update(matched_passages)

    return AgentObservation(
        summary=f"Expanded entity `{entity or 'N/A'}` and added {len(matched_passages)} passage(s).",
        payload={
            "entity": entity,
            "query": query,
            "matched_titles": list(matched_passages),
        },
    )


def extract_evidence(state: AgentState, generator: TextGenerator) -> AgentObservation:
    if not state.candidate_passages:
        return AgentObservation(
            summary="No candidate passages available for evidence extraction.",
            payload={"evidence_titles": []},
        )

    response = generator(build_extract_evidence_prompt(state))
    try:
        payload = extract_json_payload(response)
        evidence_entries = payload.get("evidence", [])
        if not isinstance(evidence_entries, list):
            evidence_entries = []
    except (ValueError, json.JSONDecodeError):
        evidence_entries = []

    if not evidence_entries:
        evidence_entries = [
            {"title": title, "content": content}
            for title, content in list(state.candidate_passages.items())[:2]
        ]

    added_titles: list[str] = []
    for entry in evidence_entries:
        title = str(entry.get("title", "")).strip()
        content = str(entry.get("content", "")).strip()
        if not title or not content:
            continue
        if _has_evidence(state, title, content):
            continue
        state.evidence_buffer.append(EvidenceItem(title=title, content=content))
        added_titles.append(title)

    state.working_memory.append(f"Evidence titles: {', '.join(added_titles) or 'none'}")
    return AgentObservation(
        summary=f"Added {len(added_titles)} evidence item(s).",
        payload={"evidence_titles": added_titles},
    )


def stop_and_answer(state: AgentState, generator: TextGenerator) -> AgentObservation:
    answer = generator(build_final_answer_prompt(state)).strip()
    state.final_answer = answer
    state.draft_answer = answer
    return AgentObservation(
        summary="Generated final answer.",
        payload={"final_answer": answer},
    )


def extract_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Expected JSON content, received an empty response.")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("No JSON object found in response.")

    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def build_budget_fallback_answer(state: AgentState, generator: TextGenerator) -> str:
    answer = generator(build_final_answer_prompt(state)).strip()
    return answer or state.draft_answer


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _has_evidence(state: AgentState, title: str, content: str) -> bool:
    return any(item.title == title and item.content == content for item in state.evidence_buffer)
