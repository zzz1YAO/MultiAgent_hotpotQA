from __future__ import annotations

import json
from typing import Iterable

try:
    from src.workflow_schema import AgentState, QuestionSample
except ImportError:
    from workflow_schema import AgentState, QuestionSample


def build_filter_prompt(sample: QuestionSample) -> str:
    return f"""System: Respond in the following format:
<Info>...</Info>
<Answer>...</Answer>

{sample.formatted_context}
Of the information above, find the original text that is most relevant to the question below.
Question: {sample.question}
Put the selected text inside <Info></Info>.
"""


def build_teacher_filter_prompt(filtered_info: str, sample: QuestionSample) -> str:
    return f"""Context:
{sample.formatted_context}

Question: {sample.question}

Please decide whether the info given below appears in the context above and whether it is strongly related to the question.
Answer with only one label: yes, no, or unknown.

Info: {filtered_info}
"""


def build_answer_prompt(filtered_info: str, question_content: str, context: str = "") -> str:
    return f"""User:
Question: {question_content}
Key information: {filtered_info}
Other information: {context}
Answer:"""


def build_teacher_answer_prompt(question_prompt: str, answer: str) -> str:
    return f"""Please decide whether this question-answer pair is reasonable and whether there is any obvious reasoning error.
If there is no error, answer only 'yes'. If there is an error, answer only 'no'. If you cannot tell, answer only 'unknown'.

Question: {question_prompt}
Answer: {answer}

Judgment:"""


def build_controller_prompt(state: AgentState, allowed_actions: Iterable[str]) -> str:
    summary = summarize_state(state)
    action_schema = {
        "action": "one of the allowed actions",
        "arguments": {"query": "optional string", "entity": "optional string"},
        "reason": "short explanation",
    }
    return f"""You are the controller of a multi-hop QA agent.
Choose exactly one next action from this list:
{json.dumps(list(allowed_actions), ensure_ascii=False)}

Return a single JSON object with this schema:
{json.dumps(action_schema, ensure_ascii=False)}

Current state:
{summary}
"""


def build_decompose_prompt(state: AgentState) -> str:
    return f"""Break the question into at most 3 concise subquestions for multi-hop reasoning.
Return JSON with keys "subquestions" and "entities".

Question: {state.question}
"""


def build_expand_entity_prompt(state: AgentState) -> str:
    evidence_blob = _join_evidence(state)
    return f"""Based on the question and current evidence, identify the next bridge entity to follow.
Return JSON with keys "entity" and "query".

Question: {state.question}
Evidence:
{evidence_blob}
"""


def build_extract_evidence_prompt(state: AgentState) -> str:
    passage_blob = "\n\n".join(
        f"Title: {title}\nContent: {content}"
        for title, content in state.candidate_passages.items()
    )
    return f"""Select the most useful supporting statements for answering the question.
Return JSON with key "evidence", whose value is a list of objects with keys "title" and "content".

Question: {state.question}
Candidate passages:
{passage_blob}
"""


def build_verify_prompt(state: AgentState) -> str:
    evidence_blob = _join_evidence(state)
    verifier_schema = {
        "status": "SUFFICIENT | INSUFFICIENT | CONFLICT | UNKNOWN",
        "reason": "short explanation",
        "missing_items": ["optional missing information"],
        "confidence": 0.0,
    }
    return f"""Judge whether the current evidence is sufficient to answer the question.
Return a single JSON object with this schema:
{json.dumps(verifier_schema, ensure_ascii=False)}

Question: {state.question}
Evidence:
{evidence_blob}
Draft answer: {state.draft_answer or '<empty>'}
"""


def build_final_answer_prompt(state: AgentState) -> str:
    evidence_blob = _join_evidence(state)
    return f"""Answer the question using only the evidence below.
Question: {state.question}

Evidence:
{evidence_blob}

Answer briefly and directly.
"""


def summarize_state(state: AgentState) -> str:
    summary = {
        "question": state.question,
        "subquestions": state.subquestions,
        "candidate_entities": state.candidate_entities,
        "candidate_passage_titles": list(state.candidate_passages),
        "evidence_titles": [item.title for item in state.evidence_buffer],
        "draft_answer": state.draft_answer,
        "verification_status": state.verification_status,
        "verification_reason": state.verification_reason,
        "remaining_steps": state.remaining_steps,
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _join_evidence(state: AgentState) -> str:
    if not state.evidence_buffer:
        return "<no evidence>"

    return "\n\n".join(
        f"Title: {item.title}\nContent: {item.content}"
        for item in state.evidence_buffer
    )
