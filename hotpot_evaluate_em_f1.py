from __future__ import annotations

import json
import sys
from pathlib import Path

from src.text_metrics import normalize_answer, token_f1


def f1_score(prediction: str, truth: str) -> float:
    return token_f1(prediction, truth, remove_articles=True)


def evaluate(pred_file: str | Path, gold_file: str | Path) -> dict[str, float]:
    with open(pred_file, "r", encoding="utf-8") as pred_handle:
        preds = json.load(pred_handle)
    with open(gold_file, "r", encoding="utf-8") as gold_handle:
        golds = json.load(gold_handle)

    gold_dict = {item["_id"]: item["answer"] for item in golds}
    total_em = 0.0
    total_f1 = 0.0
    count = 0

    for qid, pred_ans in preds.items():
        if qid not in gold_dict:
            print(f"警告: 预测中存在未知问题ID {qid}")
            continue

        truth_ans = gold_dict[qid]
        total_em += float(normalize_answer(pred_ans) == normalize_answer(truth_ans))
        total_f1 += f1_score(pred_ans, truth_ans)
        count += 1

    if count == 0:
        raise ValueError("No overlapping question IDs found between prediction and gold files.")

    metrics = {
        "count": float(count),
        "em": total_em / count,
        "f1": total_f1 / count,
    }
    print(f"评估结果 (共 {count} 条):")
    print(f"Exact Match (EM): {metrics['em']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    return metrics


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("用法: python hotpot_evaluate_em_f1.py [预测文件.json] [标准答案文件.json]")
        return 1
    evaluate(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
