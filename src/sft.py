from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

try:
    from src.config import DEFAULT_SFT_DATASET_PATH, ROOT_DIR, ensure_directory
    from src.logging_utils import get_logger
except ImportError:
    from config import DEFAULT_SFT_DATASET_PATH, ROOT_DIR, ensure_directory
    from logging_utils import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class SftTrainingConfig:
    dataset_path: str = str(DEFAULT_SFT_DATASET_PATH)
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    log_dir: str = str(ROOT_DIR / "train_logs_sft_hotpotqa")
    output_dir: str = str(ROOT_DIR / "hotpotqa_sft_model/outputs")
    lora_output_dir: str = str(ROOT_DIR / "hotpotqa_sft_model/sft_loras")
    max_seq_length: int = 4096
    lora_rank: int = 64
    gpu_memory_utilization: float = 0.4


def load_hotpotqa_data(json_file: str):
    dataset = load_dataset("json", data_files=json_file, split="train")
    logger.info("Loaded %s examples from %s", len(dataset), json_file)

    def convert_to_chat(example):
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "Extract the most relevant statements related to the question from the information below. ",
                },
                {"role": "user", "content": "Question:" + example["input"]},
                {"role": "assistant", "content": "Answer:" + example["output"]},
            ]
        }

    dataset = dataset.map(convert_to_chat, batched=False)
    return dataset.remove_columns(["input", "output"])


def formatting_prompts_func(tokenizer, examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ) + tokenizer.eos_token
        texts.append(text)
    return texts


def main() -> None:
    config = SftTrainingConfig()
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    ensure_directory(Path(config.log_dir))
    ensure_directory(Path(config.output_dir).parent)
    ensure_directory(Path(config.lora_output_dir).parent)
    SummaryWriter(log_dir=config.log_dir)
    logger.info("Current PID: %s", os.getpid())

    dataset = load_hotpotqa_data(config.dataset_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=config.lora_rank,
        gpu_memory_utilization=config.gpu_memory_utilization,
        device_map="cuda",
    )

    tokenizer.chat_template = """
{% for message in messages %}
{% if message.role == 'system' %}<|im_start|>system\n{{ message.content }}<|im_end>\n{% endif %}
{% if message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end>\n{% endif %}
{% if message.role == 'assistant' %}<|im_start|>assistant\n{{ message.content }}<|im_end>\n{% endif %}
{% endfor %}
"""

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

    training_args = SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        logging_steps=10,
        num_train_epochs=2,
        report_to="tensorboard",
        logging_dir=config.log_dir,
        max_seq_length=config.max_seq_length,
        bf16=True,
        save_strategy="steps",
        save_steps=500,
        lr_scheduler_type="cosine",
        warmup_steps=30,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        formatting_func=lambda examples: formatting_prompts_func(tokenizer, examples),
    )
    trainer.train()
    model.save_lora(config.lora_output_dir)


if __name__ == "__main__":
    main()
