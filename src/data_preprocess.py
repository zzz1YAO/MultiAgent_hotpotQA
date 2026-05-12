from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from src.workflow_schema import ContextDocument, QuestionSample
    from src.agent_prompts import (
        build_answer_prompt,
        build_filter_prompt,
        build_teacher_answer_prompt,
        build_teacher_filter_prompt,
    )
except ImportError:
    from workflow_schema import ContextDocument, QuestionSample
    from agent_prompts import (
        build_answer_prompt,
        build_filter_prompt,
        build_teacher_answer_prompt,
        build_teacher_filter_prompt,
    )


HotpotContext = list[list[Any]]
ProcessedSample = dict[str, Any]


def format_context(context_items: HotpotContext) -> str:
    context_parts: list[str] = []
    for title, sentences in context_items:
        sentence_block = " ".join(sentences)
        context_parts.append(f"{title}:\n {sentence_block}\n")
    return ". ".join(context_parts)


def load_hotpot_samples(file_path: str | Path) -> list[QuestionSample]:
    with open(file_path, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    samples: list[QuestionSample] = []
    for item in raw_data:
        context_documents = tuple(
            ContextDocument(
                title=str(title),
                sentences=tuple(str(sentence) for sentence in sentences),
            )
            for title, sentences in item["context"]
        )
        supporting_facts = tuple(
            (str(title), int(sentence_index))
            for title, sentence_index in item["supporting_facts"]
        )
        samples.append(
            QuestionSample(
                sample_id=str(item["_id"]),
                question=str(item["question"]),
                answer=str(item["answer"]),
                question_type=str(item["type"]),
                level=str(item["level"]),
                supporting_facts=supporting_facts,
                context_documents=context_documents,
            )
        )
    return samples


def preprocess_hotpot_file(file_path: str | Path) -> list[ProcessedSample]:
    """
    Preserve the legacy processed-sample structure used by older scripts.
    """
    return [sample.to_processed_sample() for sample in load_hotpot_samples(file_path)]


def processed_sample_to_question_sample(data_item: ProcessedSample) -> QuestionSample:
    context_documents = tuple(
        ContextDocument(title=title, sentences=(content,))
        for title, content in parse_context_string(data_item["model_input"]["context"]).items()
    )
    supporting_facts = tuple(
        (str(title), int(sentence_index))
        for title, sentence_index in data_item.get("supporting_facts", [])
    )
    return QuestionSample(
        sample_id=str(data_item["id"]),
        question=str(data_item["model_input"]["question"]),
        answer=str(data_item.get("answer", "")),
        question_type=str(data_item.get("type", "")),
        level=str(data_item.get("level", "")),
        supporting_facts=supporting_facts,
        context_documents=context_documents,
    )


def refine_for_filter_input(data_item: ProcessedSample | QuestionSample) -> str:
    sample = _coerce_to_question_sample(data_item)
    return build_filter_prompt(sample)


def refine_for_fliter_input(data_item: ProcessedSample | QuestionSample) -> str:
    return refine_for_filter_input(data_item)


def extract_info_from_response(response_text: str) -> str:
    match = re.search(r"<Info>(.*?)</Info>", response_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def refine_for_teacher_judge_filter(
    filtered_info: str,
    data_item: ProcessedSample | QuestionSample,
) -> str:
    sample = _coerce_to_question_sample(data_item)
    return build_teacher_filter_prompt(filtered_info, sample)


def refine_for_teacher_judge_fliter(
    filtered_info: str,
    data_item: ProcessedSample | QuestionSample,
) -> str:
    return refine_for_teacher_judge_filter(filtered_info, data_item)


def teacher_agent_judge(response: str) -> str:
    lowered = response.lower()
    has_yes = bool(re.search(r"\byes\b", lowered))
    has_no = bool(re.search(r"\bno\b", lowered))
    has_unknown = bool(re.search(r"\bunknown\b", lowered))

    if has_unknown or (has_yes and has_no) or (not has_yes and not has_no):
        return "UNKNOWN"
    if has_yes:
        return "YES"
    return "NO"


def refine_for_answer_agent(
    filtered_info: str,
    question_content: str,
    context: str = "",
) -> str:
    return build_answer_prompt(filtered_info, question_content, context)


def refine_for_teacher_judge_answer(prompt: str, answer: str) -> str:
    return build_teacher_answer_prompt(prompt, answer)


def transform_hotpotqa_to_sft(input_file: str | Path, output_file: str | Path) -> None:
    with open(input_file, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    sft_data: list[dict[str, str]] = []
    for item in raw_data:
        context_parts: list[str] = []
        for title, sentences in item["context"]:
            context_parts.append(f"Title: {title}")
            context_parts.extend(sentences)
        context = "\n".join(context_parts)

        support_sentences: list[str] = []
        for title, sent_idx in item["supporting_facts"]:
            for ctx_title, sentences in item["context"]:
                if ctx_title == title and sent_idx < len(sentences):
                    support_sentences.append(sentences[sent_idx])
                    break

        sft_data.append(
            {
                "input": f"{item['question']}\n{context}",
                "output": "\n".join(support_sentences),
            }
        )

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(sft_data, handle, ensure_ascii=False, indent=2)


def extract_entity_perspective(text: str) -> tuple[str | None, str | None]:
    entity, perspective = _extract_entity_perspective_from_json(text)
    if entity is not None or perspective is not None:
        return entity, perspective
    return _extract_entity_perspective_with_regex(text)


def _extract_entity_perspective_from_json(text: str) -> tuple[str | None, str | None]:
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return None, None

    try:
        data = json.loads(text[start_index : end_index + 1])
    except json.JSONDecodeError:
        return None, None

    return data.get("entity"), data.get("perspective")


def _extract_entity_perspective_with_regex(text: str) -> tuple[str | None, str | None]:
    entity_match = re.search(r'"entity":\s*"(.*?)"', text)
    perspective_match = re.search(r'"perspective":\s*"(.*?)"', text)
    entity = entity_match.group(1) if entity_match else None
    perspective = perspective_match.group(1) if perspective_match else None
    return entity, perspective


def extract_matching_info(
    data_string: str,
    query_string: str,
    fuzzy_threshold: float = 0.8,
) -> str:
    parsed_info = parse_context_string(data_string)
    matches = find_matching_entries(query_string, parsed_info, fuzzy_threshold)
    if not matches:
        return ""
    return "\n. ".join(f"{title}:\n{content}" for title, content in matches.items())


def parse_context_string(data_string: str) -> dict[str, str]:
    parsed_entries: dict[str, str] = {}
    for entry in data_string.split("\n. "):
        cleaned_entry = entry.strip()
        if not cleaned_entry:
            continue

        colon_position = cleaned_entry.find(":\n")
        if colon_position == -1:
            continue

        title = cleaned_entry[:colon_position].strip()
        content = cleaned_entry[colon_position + 2 :].strip()
        if title.startswith(". "):
            title = title[2:]
        parsed_entries[title] = content

    return parsed_entries


def find_matching_entries(
    query_string: str,
    parsed_entries: dict[str, str],
    fuzzy_threshold: float,
) -> dict[str, str]:
    matches: dict[str, str] = {}
    query_lower = query_string.lower()
    query_words = [word for word in re.findall(r"\b\w+\b", query_lower) if len(word) >= 3]

    for title, content in parsed_entries.items():
        title_lower = title.lower()
        if title_lower in query_lower:
            matches[title] = content
            continue

        title_words = [word for word in re.findall(r"\b\w+\b", title_lower) if len(word) >= 3]
        if any(
            SequenceMatcher(None, title_word, query_word).ratio() >= fuzzy_threshold
            for title_word in title_words
            for query_word in query_words
        ):
            matches[title] = content

    return matches


def _coerce_to_question_sample(data_item: ProcessedSample | QuestionSample) -> QuestionSample:
    if isinstance(data_item, QuestionSample):
        return data_item
    return processed_sample_to_question_sample(data_item)
