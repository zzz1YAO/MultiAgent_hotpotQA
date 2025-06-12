import json
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 配置参数 (根据你的工作区结构调整)
CHECKPOINT = 7500  # 设置为目标checkpoint
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"  # 基础模型路径（来自之前训练的SFT模型）
LORA_ROOT = "../hotpotqa_grpo_model/"  # LoRA检查点根目录
OUTPUT_ROOT = "../answer_llm_7B_v1"  # 合并模型输出目录

def merge_and_save_model(checkpoint):
    # 构建路径
    lora_directory = os.path.join(LORA_ROOT, f"checkpoint-{checkpoint}")
    merged_model_directory = os.path.join(OUTPUT_ROOT, f"checkpoint-{checkpoint}")
    
    # 创建输出目录
    os.makedirs(merged_model_directory, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Processing checkpoint: {checkpoint}")
    print(f"LoRA from: {lora_directory}")
    print(f"Will save to: {merged_model_directory}")
    print(f"{'='*50}")

    try:
        # 加载基础模型和tokenizer (16-bit)
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

        # 加载额外的token（如果有）
        added_tokens_path = os.path.join(lora_directory, "added_tokens.json")
        if os.path.exists(added_tokens_path):
            print("Loading additional tokens...")
            with open(added_tokens_path, 'r') as f:
                added_tokens_data = json.load(f)
                added_tokens = added_tokens_data.get("additional_special_tokens", [])
                if isinstance(added_tokens, list) and all(isinstance(token, str) for token in added_tokens):
                    tokenizer.add_tokens(added_tokens)
                    print(f"Added {len(added_tokens)} special tokens")
                else:
                    print("Warning: Invalid token format in added_tokens.json")

        # 加载LoRA适配器
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(
            model, 
            lora_directory, 
            device_map='auto', 
            torch_dtype=torch.float16
        )

        # 合并权重
        print("Merging weights...")
        model = model.merge_and_unload()

        # 保存合并后的模型
        print("Saving merged model...")
        model.save_pretrained(merged_model_directory)
        tokenizer.save_pretrained(merged_model_directory)

        print(f"\n✅ Successfully merged and saved model to: {merged_model_directory}")

    except Exception as e:
        print(f"\n❌ Error processing checkpoint {checkpoint}: {str(e)}")

if __name__ == "__main__":
    # 单checkpoint模式（3750）
    merge_and_save_model(CHECKPOINT)
    
    print("\nMerge process completed!")