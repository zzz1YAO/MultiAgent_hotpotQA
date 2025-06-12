import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
import json
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import os

import argparse
from datetime import datetime

from src.load_model import *
from src.data_preprocess import *
from src.reward_for_grpo import SYSTEM_PROMPT

from openai import OpenAI
api_key = "sk-itxnszoqzhtlbpqrdtufqywcaosvxzshakrajyqzggbyqjrq"
model_name = "Qwen/Qwen2.5-7B-Instruct"
client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")



#config 
test_dataset_path="./hotpotqa_test_400.json"
output_path = f"./results/answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# 定义模型映射 (0=base, 1=model1, 2=model2, 3=model3)
MODEL_PATHS = {
    '0': "Qwen/Qwen2.5-3B-Instruct",  # 基础模型
    '1': "./fliter_llm_sft",         # 自定义模型1路径
    '2': "./answer_llm_grpo_v1",         # 自定义模型2路径 
    '3': "./model3"  ,
    "4":"./answer_llm_grpo_v2",
    "5":"./answer_llm_7B_v1/checkpoint-4500",
    "6":"./answer_llm_7B_v1/checkpoint-7500"        # 自定义模型3路径
}



# 命令行参数解析
parser = argparse.ArgumentParser(description='run multi-agent script')
parser.add_argument('--models', type=str, required=True,
                    help='filter/answer/teacher agent model path')
args = parser.parse_args()

# 验证输入格式




# 分配模型路径
  # 第一位对应filter
answer_agent_llm_path = MODEL_PATHS[args.models[0]]  # 第二位对应answer
# 第三位对应teacher


#==============LOAD DATASET============================

dataset=preprocess_hotpot_file(test_dataset_path)

#===============LOAD MODEL INTO AGENT====================


answer_agent,_,_=load_answer_agent_model(answer_agent_llm_path,gpu_memory_utilization=0.8)

#================START INFERENCE LOOP=====================
results={}
for test_sample in tqdm(dataset, desc="Processing samples"):

    # necessary info
    question_id = test_sample['id']
    question_content = test_sample['model_input']['question']
    context = test_sample['model_input']['context']


    # --- Filter Agent turn ---
    extracted_context = "" # Initialize
    try:
        # Prompt for Filter Agent, directly using string concatenation
        prompt_fliter = f"""
Please identify the entity that the question is asking about.Then output the entities' name.
Question:{question_content}
"""

        extracted_context = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_fliter}
                ],
                temperature=0.6,
                max_tokens=350
            )
        extracted_context=extracted_context.choices[0].message.content.strip()
        # print(f"ID: {question_id} - Filter Output: {extracted_context[:100]}...") # Optional debug
    except Exception as e:
        print(f"Error during Filter Agent turn for ID {question_id}: {e}")
        extracted_context = f"Error running filter agent: {e}" # Store error

    key_info=extract_matching_info(context,extracted_context)


    prompt_strong_answer=f'''
SYSTEM:{SYSTEM_PROMPT}
User:Question: {question_content}\n
Context:{key_info}\nAssistant:
'''
    strong_answer=get_answer_agent_response(answer_agent,prompt_strong_answer)


 
        
    results[question_id] = strong_answer






with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved as {output_path}")

