from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.workflow_runner import WorkflowConfig, WorkflowRunner, WorkflowRuntime
from src.config import DEFAULT_MODEL_PATHS, DEFAULT_RESULTS_DIR, DEFAULT_TEST_DATASET_PATH, ensure_directory
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
    parser = argparse.ArgumentParser(description="Batch agentic evaluation on HotpotQA.")
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Three model ids: filter/evidence, answer, and teacher(controller+verifier).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_TEST_DATASET_PATH,
        help="HotpotQA dataset path.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum controller steps before forced stop.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional sample limit for quick evaluation. Use 0 for the full dataset.",
    )
    return parser.parse_args()


def validate_model_selection(selection: str) -> None:
    valid_ids = set(DEFAULT_MODEL_PATHS)
    if len(selection) != 3 or any(model_id not in valid_ids for model_id in selection):
        raise ValueError("模型参数必须是3位数字，且每位需存在于 MODEL_PATHS 映射中")


def build_runtime(selection: str) -> WorkflowRuntime:
    filter_model, _, filter_sampling = load_filter_agent_model(DEFAULT_MODEL_PATHS[selection[0]])
    answer_model, _, _ = load_answer_agent_model(DEFAULT_MODEL_PATHS[selection[1]], gpu_memory_utilization=0.8)
    teacher_model, _, _ = load_teacher_agent_model(DEFAULT_MODEL_PATHS[selection[2]])

    return WorkflowRuntime(
        controller=lambda prompt: get_teacher_agent_response(teacher_model, prompt),
        evidence=lambda prompt: get_filter_agent_response(filter_model, prompt, filter_sampling),
        answer=lambda prompt: get_answer_agent_response(answer_model, prompt),
        verifier=lambda prompt: get_teacher_agent_response(teacher_model, prompt),
    )


def build_output_paths() -> tuple[Path, Path]:
    ensure_directory(DEFAULT_RESULTS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_path = DEFAULT_RESULTS_DIR / f"agentic_answers_{timestamp}.json"
    trace_path = DEFAULT_RESULTS_DIR / f"agentic_traces_{timestamp}.json"
    return prediction_path, trace_path


def evaluate_dataset(
    dataset_path: Path,
    runtime: WorkflowRuntime,
    max_steps: int,
    limit: int,
) -> tuple[dict[str, str], dict[str, dict]]:
    samples = load_hotpot_samples(dataset_path)
    if limit > 0:
        samples = samples[:limit]

    runner = WorkflowRunner(runtime, WorkflowConfig(max_steps=max_steps))
    predictions: dict[str, str] = {}
    traces: dict[str, dict] = {}

    for sample in samples:
        try:
            result = runner.run(sample)
        except Exception as exc:
            logger.error("Error during inference for ID %s: %s", sample.sample_id, exc)
            continue

        predictions[sample.sample_id] = result.final_answer
        traces[sample.sample_id] = result.to_dict()

    return predictions, traces


def main() -> None:
    args = parse_args()
    validate_model_selection(args.models)
    runtime = build_runtime(args.models)
    predictions, traces = evaluate_dataset(args.dataset, runtime, args.max_steps, args.limit)
    prediction_path, trace_path = build_output_paths()

    with prediction_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=False)
    with trace_path.open("w", encoding="utf-8") as handle:
        json.dump(traces, handle, indent=2, ensure_ascii=False)

    logger.info("Saved %s predictions to %s", len(predictions), prediction_path)
    logger.info("Saved %s traces to %s", len(traces), trace_path)


if __name__ == "__main__":
    main()
