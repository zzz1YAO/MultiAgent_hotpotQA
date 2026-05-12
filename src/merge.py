from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.config import ROOT_DIR, ensure_directory
    from src.logging_utils import get_logger
except ImportError:
    from config import ROOT_DIR, ensure_directory
    from logging_utils import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class MergeConfig:
    checkpoint: int = 7500
    base_model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_root: str = str(ROOT_DIR / "hotpotqa_grpo_model")
    output_root: str = str(ROOT_DIR / "answer_llm_7B_v1")


def merge_and_save_model(checkpoint: int, config: MergeConfig) -> None:
    lora_directory = os.path.join(config.lora_root, f"checkpoint-{checkpoint}")
    merged_model_directory = os.path.join(config.output_root, f"checkpoint-{checkpoint}")
    ensure_directory(Path(merged_model_directory))

    logger.info("Processing checkpoint: %s", checkpoint)
    logger.info("LoRA from: %s", lora_directory)
    logger.info("Will save to: %s", merged_model_directory)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_path)

        added_tokens_path = os.path.join(lora_directory, "added_tokens.json")
        if os.path.exists(added_tokens_path):
            with open(added_tokens_path, "r", encoding="utf-8") as handle:
                added_tokens_data = json.load(handle)
            added_tokens = added_tokens_data.get("additional_special_tokens", [])
            if isinstance(added_tokens, list) and all(isinstance(token, str) for token in added_tokens):
                tokenizer.add_tokens(added_tokens)
                logger.info("Added %s special tokens", len(added_tokens))
            else:
                logger.warning("Invalid token format in added_tokens.json")

        model = PeftModel.from_pretrained(
            model,
            lora_directory,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = model.merge_and_unload()
        model.save_pretrained(merged_model_directory)
        tokenizer.save_pretrained(merged_model_directory)
        logger.info("Successfully merged and saved model to: %s", merged_model_directory)
    except Exception as exc:
        logger.error("Error processing checkpoint %s: %s", checkpoint, exc)


def main() -> None:
    config = MergeConfig()
    merge_and_save_model(config.checkpoint, config)
    logger.info("Merge process completed")


if __name__ == "__main__":
    main()
