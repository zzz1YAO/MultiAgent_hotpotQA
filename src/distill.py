from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from src.config import DEFAULT_MODEL_PATHS, DEFAULT_TEST_DATASET_PATH, ROOT_DIR, ensure_directory
    from src.data_preprocess import (
        extract_entity_perspective,
        preprocess_hotpot_file,
    )
    from src.load_model import (
        get_answer_agent_response,
        get_fliter_agent_response,
        get_teacher_agent_response,
        load_answer_agent_model,
        load_fliter_agent_model,
        load_teacher_agent_model,
    )
    from src.logging_utils import get_logger
    from src.reward_for_grpo import SYSTEM_PROMPT
except ImportError:
    from config import DEFAULT_MODEL_PATHS, DEFAULT_TEST_DATASET_PATH, ROOT_DIR, ensure_directory
    from data_preprocess import extract_entity_perspective, preprocess_hotpot_file
    from load_model import (
        get_answer_agent_response,
        get_fliter_agent_response,
        get_teacher_agent_response,
        load_answer_agent_model,
        load_fliter_agent_model,
        load_teacher_agent_model,
    )
    from logging_utils import get_logger
    from reward_for_grpo import SYSTEM_PROMPT


logger = get_logger(__name__)


@dataclass(frozen=True)
class DistillConfig:
    dataset_path: str = str(DEFAULT_TEST_DATASET_PATH)
    output_path: str = str(ROOT_DIR / "distill_prepare_dataset.json")
    experiment_log_path: str = str(ROOT_DIR / "experiment_log.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run multi-agent distillation script")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="filter/answer/teacher agent model path",
    )
    return parser.parse_args()


def validate_model_selection(selection: str) -> None:
    valid_ids = set(DEFAULT_MODEL_PATHS)
    if len(selection) != 3 or any(model_id not in valid_ids for model_id in selection):
        raise ValueError("模型参数必须是3位数字，且每位需存在于 MODEL_PATHS 映射中")


def main() -> None:
    args = parse_args()
    validate_model_selection(args.models)
    config = DistillConfig()
    ensure_directory(Path(config.output_path).parent)

    dataset = preprocess_hotpot_file(config.dataset_path)
    fliter_agent, _, _ = load_fliter_agent_model(DEFAULT_MODEL_PATHS[args.models[0]])
    answer_agent, _, _ = load_answer_agent_model(DEFAULT_MODEL_PATHS[args.models[1]])
    teacher_agent, _, _ = load_teacher_agent_model(DEFAULT_MODEL_PATHS[args.models[2]])

    results: list[str] = []
    for test_sample in dataset:
        question_content = test_sample["model_input"]["question"]
        context = test_sample["model_input"]["context"]

        try:
            prompt_teacher_instruct = f"""
Answer in dict format:
Please identify the entity that the question is asking about and the perspective of the inquiry
Question:{question_content}"""
            teacher_instruct_response = get_teacher_agent_response(teacher_agent, prompt_teacher_instruct)
            entity, _ = extract_entity_perspective(teacher_instruct_response)

            prompt_fliter = f"""
You need to find out the information started with{entity}
Information:{context}
"""
            extracted_context = get_fliter_agent_response(fliter_agent, prompt_fliter)

            prompt_answer = f"""
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

Question:{question_content}
Info:{extracted_context}
"""
            weak_answer = get_teacher_agent_response(answer_agent, prompt_answer)

            prompt_strong_answer = f"""
SYSTEM:{SYSTEM_PROMPT}
User:Question: {question_content}\nContext: {context}
"""
            strong_answer = get_answer_agent_response(answer_agent, prompt_strong_answer)
        except Exception as exc:
            logger.error("Error during distillation sample %s: %s", test_sample["id"], exc)
            continue

        final_string = f"""SYSTEM：I will give you a question and two answers.Please output the answer which you think is better.
    Question:{question_content}
    Context:{context}
Answer1:{weak_answer}
Answer2:{strong_answer}
"""
        results.append(final_string)

    with open(config.output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    logger.info("Saved as %s", config.output_path)

    experiment_data = {
        "run_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model_paths": {
            "filter_agent": DEFAULT_MODEL_PATHS[args.models[0]],
            "answer_agent": DEFAULT_MODEL_PATHS[args.models[1]],
            "teacher_agent": DEFAULT_MODEL_PATHS[args.models[2]],
        },
        "output_path": str(ROOT_DIR / f"results/answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
    }
    with open(config.experiment_log_path, "w", encoding="utf-8") as handle:
        json.dump(experiment_data, handle, indent=2)


if __name__ == "__main__":
    main()
