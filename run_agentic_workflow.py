from __future__ import annotations

import argparse
import json

from src.workflow_runner import WorkflowConfig, WorkflowRunner, WorkflowRuntime
from src.config import DEFAULT_MODEL_PATHS, DEFAULT_TEST_DATASET_PATH
from src.data_preprocess import load_hotpot_samples
from src.load_model import (
    get_answer_agent_response,
    get_filter_agent_response,
    get_teacher_agent_response,
    load_answer_agent_model,
    load_filter_agent_model,
    load_teacher_agent_model,
)
from src.logging_utils import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic single-sample inference on HotpotQA.")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Three model ids: filter/evidence, answer, and teacher(controller+verifier).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_TEST_DATASET_PATH),
        help="HotpotQA dataset path.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default="",
        help="Specific sample id to run. Overrides --sample-index when set.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based sample index when --sample-id is not provided.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum controller steps before forced stop.",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print the full execution trace as JSON.",
    )
    return parser.parse_args()


def validate_model_selection(selection: str) -> None:
    valid_ids = set(DEFAULT_MODEL_PATHS)
    if len(selection) != 3 or any(model_id not in valid_ids for model_id in selection):
        raise ValueError("模型参数必须是3位数字，且每位需存在于 MODEL_PATHS 映射中")


def build_runtime(selection: str) -> WorkflowRuntime:
    filter_model, _, filter_sampling = load_filter_agent_model(DEFAULT_MODEL_PATHS[selection[0]])
    answer_model, _, _ = load_answer_agent_model(DEFAULT_MODEL_PATHS[selection[1]])
    teacher_model, _, _ = load_teacher_agent_model(DEFAULT_MODEL_PATHS[selection[2]])

    return WorkflowRuntime(
        controller=lambda prompt: get_teacher_agent_response(teacher_model, prompt),
        evidence=lambda prompt: get_filter_agent_response(filter_model, prompt, filter_sampling),
        answer=lambda prompt: get_answer_agent_response(answer_model, prompt),
        verifier=lambda prompt: get_teacher_agent_response(teacher_model, prompt),
    )


def resolve_sample(dataset_path: str, sample_id: str, sample_index: int):
    samples = load_hotpot_samples(dataset_path)
    if sample_id:
        for sample in samples:
            if sample.sample_id == sample_id:
                return sample
        raise ValueError(f"Sample id not found: {sample_id}")

    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"Sample index out of range: {sample_index}")
    return samples[sample_index]


def main() -> None:
    args = parse_args()
    validate_model_selection(args.models)

    sample = resolve_sample(args.dataset, args.sample_id, args.sample_index)
    runtime = build_runtime(args.models)
    runner = WorkflowRunner(runtime, WorkflowConfig(max_steps=args.max_steps))
    result = runner.run(sample)

    logger.info("Sample ID: %s", sample.sample_id)
    logger.info("Question: %s", sample.question)
    logger.info("Final answer: %s", result.final_answer)
    logger.info("Status: %s | Verification: %s", result.status, result.verification_status)

    if args.show_trace:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
