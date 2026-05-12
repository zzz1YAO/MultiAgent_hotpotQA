from __future__ import annotations

import json
import re

from datasets import Dataset

try:
    from src.text_metrics import token_f1
except ImportError:
    from text_metrics import token_f1


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def load_rlhf_dataset(file_path: str, system_prompt: str = SYSTEM_PROMPT) -> Dataset:
    with open(file_path, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    prompts: list[list[dict[str, str]]] = []
    answers: list[str] = []
    for item in raw_data:
        context = "\n".join(
            f"{title}:\n{'. '.join(sentences)}" for title, sentences in item["context"]
        )
        prompts.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {item['question']}\nContext: {context}"},
            ]
        )
        answers.append(item["answer"])

    return Dataset.from_dict({"prompt": prompts, "answer": answers})


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, response, re.DOTALL) else 0.0 for response in responses]


def calculate_f1(prediction: str, truth: str) -> float:
    return min(token_f1(prediction, truth, remove_articles=False), 1.0)


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    rewards: list[float] = []
    for completion, answer_text in zip(completions, answer):
        response = completion[0]["content"] if isinstance(completion, list) else completion
        extracted_answer = extract_xml_answer(response)
        rewards.append(calculate_f1(extracted_answer, answer_text) * 3)
    return rewards


def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(content) for content in contents]
