from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from trl import GRPOConfig, GRPOTrainer

import json

from reward_for_grpo import *

# ==================== 1. 路径配置 ====================
class Config:
    # 训练日志
    RLHF_LOG_DIR = "./train_logs_grpo_hotpotqa"  
    # 模型输出
    MODEL_OUTPUT_DIR = "./hotpotqa_grpo_model"
    # 数据路径
    DATASET_PATH = "./hotpot_train_v1.1.json"  
    # 初始模型
    BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct" 

# 确保目录存在
os.makedirs(Config.RLHF_LOG_DIR, exist_ok=True)
os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)

# ==================== 2. 初始化 ====================
writer = SummaryWriter(log_dir=Config.RLHF_LOG_DIR)
print(f"Current PID: {os.getpid()}")



dataset = load_rlhf_dataset(Config.DATASET_PATH)



# ==================== 4. 模型加载 ====================
max_seq_length = 4096 
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=Config.BASE_MODEL_PATH,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.4,
    device_map='auto'  # 自动选择设备
)

# ==================== 5. LoRA配置 ====================
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)



# ==================== 7. 训练配置 ====================

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 4096,
    max_completion_length = 150,
    num_train_epochs=2,
    save_steps=1500,
    max_grad_norm = 0.1,
    report_to = "tensorboard", # Can use Weights & Biases
    output_dir=Config.MODEL_OUTPUT_DIR,
    logging_dir=Config.RLHF_LOG_DIR,
)
# ==================== 8. 训练器初始化 ====================
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[soft_format_reward_func,correctness_reward_func,xmlcount_reward_func],  # 使用组合奖励
    args=training_args,
    train_dataset=dataset,
)

# ==================== 9. 训练执行 ====================
print("Starting RLHF training...")
trainer.train()

# ==================== 10. 模型保存 ====================
final_model_path = os.path.join(Config.MODEL_OUTPUT_DIR, "final_model")
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Training complete. Model saved to {final_model_path}")