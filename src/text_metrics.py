from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str, remove_articles: bool = True) -> str:
    normalized = text.lower()
    normalized = "".join(char for char in normalized if char not in set(string.punctuation))
    if remove_articles:
        normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(normalized.split())


def token_f1(prediction: str, truth: str, remove_articles: bool = True) -> float:
    pred_tokens = normalize_answer(prediction, remove_articles=remove_articles).split()
    truth_tokens = normalize_answer(truth, remove_articles=remove_articles).split()

    if (
        pred_tokens in [["yes"], ["no"], ["noanswer"]]
        or truth_tokens in [["yes"], ["no"], ["noanswer"]]
    ):
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)
