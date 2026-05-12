from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from src.config import DEFAULT_TEST_DATASET_PATH, OpenAICompatibleConfig
from src.data_preprocess import preprocess_hotpot_file
from src.logging_utils import get_logger
from src.openai_compatible import build_openai_client, chat_completion


logger = get_logger(__name__)


PromptBuilder = Callable[[dict], str]


@dataclass(frozen=True)
class BaselineEvalConfig:
    client_config: OpenAICompatibleConfig
    output_path: Path
    dataset_path: Path = DEFAULT_TEST_DATASET_PATH
    system_prefix: str = "System:Answer questions as briefly as possible"
    save_every: int = 20
    retry_delay_seconds: int = 5
    progress_desc: str = "Processing"


def default_prompt_builder(sample: dict) -> str:
    return str(sample["model_input"])


def evaluate_baseline_model(
    config: BaselineEvalConfig,
    prompt_builder: PromptBuilder = default_prompt_builder,
) -> dict[str, str]:
    client = build_openai_client(config.client_config)
    dataset = preprocess_hotpot_file(str(config.dataset_path))
    results: dict[str, str] = {}

    for sample in tqdm(dataset, desc=config.progress_desc):
        prompt = config.system_prefix + prompt_builder(sample)
        try:
            answer = chat_completion(
                client=client,
                model_name=config.client_config.model_name,
                user_prompt=prompt,
                system_prompt=config.client_config.system_prompt,
                temperature=config.client_config.temperature,
                max_tokens=config.client_config.max_tokens,
            )
        except Exception as exc:
            logger.error("API call failed for %s: %s", sample["id"], exc)
            time.sleep(config.retry_delay_seconds)
            continue

        if answer:
            results[sample["id"]] = answer

        if config.save_every and results and len(results) % config.save_every == 0:
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            with config.output_path.with_name("temp_results.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(results, handle, ensure_ascii=False)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    logger.info("Saved %s predictions to %s", len(results), config.output_path)
    return results
