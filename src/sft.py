# 1. 导入必要模块
from unsloth import FastLanguageModel
import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from torch.utils.tensorboard import SummaryWriter

# 2. 环境设置
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
writer = SummaryWriter(log_dir="./train_logs_sft_hotpotqa")
print(f"Current PID: {os.getpid()}")

# 3. 加载并预处理HotpotQA SFT格式数据
def load_hotpotqa_data(json_file):
    dataset = load_dataset('json', data_files=json_file, split='train')
    print(f"Loaded {len(dataset)} examples from {json_file}")
    
    # 转换为对话格式
    def convert_to_chat(example):
        return {
            "messages": [
                {"role": "system", "content": "Extract the most relevant statements related to the question from the information below. "},
                {"role": "user", "content": "Question:"+example["input"]},
                {"role": "assistant", "content": "Answer:"+example["output"]}
            ]
        }
    
    dataset = dataset.map(convert_to_chat, batched=False)
    return dataset.remove_columns(["input", "output"])

dataset = load_hotpotqa_data("Hotpotqa_sft.json")

# 4. 加载模型和分词器
model_name = "Qwen/Qwen2.5-3B-Instruct"
max_seq_length = 4096  # 根据需求调整
lora_rank = 64

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.4,
    device_map='cuda'
)

# 5. 设置Qwen聊天模板
tokenizer.chat_template = """
{% for message in messages %}
{% if message.role == 'system' %}<|im_start|>system\n{{ message.content }}<|im_end>\n{% endif %}
{% if message.role == 'user' %}<|im_start|>user\n{{ message.content }}<|im_end>\n{% endif %}
{% if message.role == 'assistant' %}<|im_start|>assistant\n{{ message.content }}<|im_end>\n{% endif %}
{% endfor %}
"""

# 6. 配置LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407
)

# 7. 定义格式化函数
def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        ) + tokenizer.eos_token
        texts.append(text)
    return texts

# 8. 配置训练参数
training_args = SFTConfig(
    output_dir="./hotpotqa_sft_model/outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    logging_steps=10,
    num_train_epochs=2,
    report_to="tensorboard",
    logging_dir="./train_logs_sft_hotpotqa",
    max_seq_length=max_seq_length,
    bf16=True,
    save_strategy="steps",
    save_steps=500,
    lr_scheduler_type="cosine",
    warmup_steps=30,
)

# 9. 初始化Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
)

# 10. 开始训练
trainer.train()
model.save_lora("./hotpotqa_sft_model/sft_loras")