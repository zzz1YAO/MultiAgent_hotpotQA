from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_TEST_DATASET_PATH = ROOT_DIR / "hotpotqa_test_400.json"
DEFAULT_TRAIN_DATASET_PATH = ROOT_DIR / "hotpot_train_v1.1.json"
DEFAULT_SFT_DATASET_PATH = ROOT_DIR / "Hotpotqa_sft.json"

DEFAULT_MODEL_PATHS: dict[str, str] = {
    "0": "Qwen/Qwen2.5-3B-Instruct",
    "1": "./fliter_llm_sft",
    "2": "./answer_llm_grpo_v1",
    "3": "./model3",
    "4": "./answer_llm_grpo_v2",
    "5": "./answer_llm_7B_v1/checkpoint-4500",
    "6": "./answer_llm_7B_v1/checkpoint-7500",
}


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT_DIR / path


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    model_name: str
    base_url: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    def resolved_api_key(self) -> str:
        api_key = self.api_key or get_env(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key. Set `{self.api_key_env}` before running this script."
            )
        return api_key
