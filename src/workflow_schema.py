from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContextDocument:
    title: str
    sentences: tuple[str, ...]

    @property
    def content(self) -> str:
        return " ".join(sentence.strip() for sentence in self.sentences).strip()


@dataclass(frozen=True)
class QuestionSample:
    sample_id: str
    question: str
    answer: str
    question_type: str
    level: str
    supporting_facts: tuple[tuple[str, int], ...]
    context_documents: tuple[ContextDocument, ...]

    @property
    def formatted_context(self) -> str:
        context_parts = [
            f"{document.title}:\n {document.content}\n"
            for document in self.context_documents
        ]
        return ". ".join(context_parts)

    @property
    def context_map(self) -> dict[str, str]:
        return {document.title: document.content for document in self.context_documents}

    def to_processed_sample(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "model_input": {
                "context": self.formatted_context,
                "question": self.question,
            },
            "answer": self.answer,
            "type": self.question_type,
            "level": self.level,
            "supporting_facts": [list(item) for item in self.supporting_facts],
        }


@dataclass
class EvidenceItem:
    title: str
    content: str
    source: str = "context"
    query: str = ""
    score: float | None = None


@dataclass(frozen=True)
class AgentAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class AgentObservation:
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierResult:
    status: str
    reason: str = ""
    missing_items: tuple[str, ...] = ()
    confidence: float | None = None


@dataclass
class AgentTraceStep:
    step_index: int
    action: str
    arguments: dict[str, Any]
    reason: str
    observation_summary: str
    verification_status: str = "UNKNOWN"
    verification_reason: str = ""


@dataclass
class AgentState:
    sample: QuestionSample
    question: str
    original_context: str
    subquestions: list[str] = field(default_factory=list)
    candidate_entities: list[str] = field(default_factory=list)
    candidate_passages: dict[str, str] = field(default_factory=dict)
    evidence_buffer: list[EvidenceItem] = field(default_factory=list)
    working_memory: list[str] = field(default_factory=list)
    draft_answer: str = ""
    final_answer: str = ""
    verification_status: str = "UNKNOWN"
    verification_reason: str = ""
    step_budget: int = 6
    trace: list[AgentTraceStep] = field(default_factory=list)

    @classmethod
    def from_sample(cls, sample: QuestionSample, step_budget: int = 6) -> "AgentState":
        return cls(
            sample=sample,
            question=sample.question,
            original_context=sample.formatted_context,
            step_budget=step_budget,
        )

    @property
    def remaining_steps(self) -> int:
        return max(self.step_budget - len(self.trace), 0)


@dataclass(frozen=True)
class AgentResult:
    sample_id: str
    final_answer: str
    status: str
    draft_answer: str
    verification_status: str
    verification_reason: str
    trace: tuple[AgentTraceStep, ...]
    evidence: tuple[EvidenceItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "final_answer": self.final_answer,
            "status": self.status,
            "draft_answer": self.draft_answer,
            "verification_status": self.verification_status,
            "verification_reason": self.verification_reason,
            "trace": [
                {
                    "step_index": step.step_index,
                    "action": step.action,
                    "arguments": step.arguments,
                    "reason": step.reason,
                    "observation_summary": step.observation_summary,
                    "verification_status": step.verification_status,
                    "verification_reason": step.verification_reason,
                }
                for step in self.trace
            ],
            "evidence": [
                {
                    "title": item.title,
                    "content": item.content,
                    "source": item.source,
                    "query": item.query,
                    "score": item.score,
                }
                for item in self.evidence
            ],
        }
