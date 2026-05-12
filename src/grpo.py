from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

try:
    from src.config import DEFAULT_TRAIN_DATASET_PATH, ROOT_DIR, ensure_directory
    from src.logging_utils import get_logger
    from src.reward_for_grpo import (
        correctness_reward_func,
        load_rlhf_dataset,
        soft_format_reward_func,
        xmlcount_reward_func,
    )
except ImportError:
    from config import DEFAULT_TRAIN_DATASET_PATH, ROOT_DIR, ensure_directory
    from logging_utils import get_logger
    from reward_for_grpo import (
        correctness_reward_func,
        load_rlhf_dataset,
        soft_format_reward_func,
        xmlcount_reward_func,
    )


logger = get_logger(__name__)


@dataclass(frozen=True)
class GrpoTrainingConfig:
    log_dir: str = str(ROOT_DIR / "train_logs_grpo_hotpotqa")
    model_output_dir: str = str(ROOT_DIR / "hotpotqa_grpo_model")
    dataset_path: str = str(DEFAULT_TRAIN_DATASET_PATH)
    base_model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 4096
    lora_rank: int = 64
    gpu_memory_utilization: float = 0.4


def create_trainer(config: GrpoTrainingConfig) -> tuple[GRPOTrainer, FastLanguageModel, object]:
    ensure_directory(Path(config.log_dir))
    ensure_directory(Path(config.model_output_dir))
    dataset = load_rlhf_dataset(config.dataset_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_path,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=config.lora_rank,
        gpu_memory_utilization=config.gpu_memory_utilization,
        device_map="auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_prompt_length=4096,
        max_completion_length=150,
        num_train_epochs=2,
        save_steps=1500,
        max_grad_norm=0.1,
        report_to="tensorboard",
        output_dir=config.model_output_dir,
        logging_dir=config.log_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            soft_format_reward_func,
            correctness_reward_func,
            xmlcount_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    return trainer, model, tokenizer


def main() -> None:
    config = GrpoTrainingConfig()
    ensure_directory(Path(config.log_dir))
    ensure_directory(Path(config.model_output_dir))
    SummaryWriter(log_dir=config.log_dir)
    logger.info("Current PID: %s", os.getpid())

    trainer, model, tokenizer = create_trainer(config)
    logger.info("Starting RLHF training")
    trainer.train()

    final_model_path = os.path.join(config.model_output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info("Training complete. Model saved to %s", final_model_path)


if __name__ == "__main__":
    main()
