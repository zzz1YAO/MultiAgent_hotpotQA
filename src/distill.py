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
#config 
test_dataset_path="./hotpotqa_test_400.json"
output_path = "distill_prepare_dataset.json"

# 定义模型映射 (0=base, 1=model1, 2=model2, 3=model3)
MODEL_PATHS = {
    '0': "Qwen/Qwen2.5-3B-Instruct",  # 基础模型
    '1': "./fliter_llm_sft",         # 自定义模型1路径
    '2': "./answer_llm_grpo_v1",         # 自定义模型2路径 
    '3': "./model3"  ,
    "4":"./answer_llm_grpo_v2"        # 自定义模型3路径
}



# 命令行参数解析
parser = argparse.ArgumentParser(description='run multi-agent script')
parser.add_argument('--models', type=str, required=True,
                    help='filter/answer/teacher agent model path')
args = parser.parse_args()

# 验证输入格式




# 分配模型路径
fliter_agent_llm_path = MODEL_PATHS[args.models[0]]  # 第一位对应filter
answer_agent_llm_path = MODEL_PATHS[args.models[1]]  # 第二位对应answer
teacher_agent_llm_path = MODEL_PATHS[args.models[2]] # 第三位对应teacher


#==============LOAD DATASET============================

dataset=preprocess_hotpot_file(test_dataset_path)

#===============LOAD MODEL INTO AGENT====================

fliter_agent,tokenizer,sample_config=load_fliter_agent_model(fliter_agent_llm_path)
answer_agent,_,_=load_answer_agent_model(answer_agent_llm_path)
teacher_agent,_,_=load_teacher_agent_model(teacher_agent_llm_path)

#================START INFERENCE LOOP=====================
results=[]
for test_sample in tqdm(dataset, desc="Processing samples"):

    # necessary info
    question_id = test_sample['id']
    question_content = test_sample['model_input']['question']
    context = test_sample['model_input']['context']

    # --- Teacher Agent turn (Initial Instruction) ---
    teacher_instruct_response = "" # Initialize
    try:
        prompt_teacher_instruct = f"""
Answer in dict format:
Please identify the entity that the question is asking about and the perspective of the inquiry
Question:{question_content}"""
        
        teacher_instruct_response = get_teacher_agent_response(
            teacher_agent, prompt_teacher_instruct
        )
        # print(f"ID: {question_id} - Teacher Instruct: {teacher_instruct_response[:100]}...") # Optional debug
    except Exception as e:
        print(f"Error during Teacher Agent (instruct) turn for ID {question_id}: {e}")
        teacher_instruct_response = f"Error: {e}" # Store error


    entity,perspective=extract_entity_perspective(teacher_instruct_response)
    # --- Filter Agent turn ---
    extracted_context = "" # Initialize
    try:
        # Prompt for Filter Agent, directly using string concatenation
        prompt_fliter = f"""
You need to find out the information started with{entity}
Information:{context}
"""
        extracted_context = get_fliter_agent_response(
            fliter_agent, prompt_fliter
        )
        # print(f"ID: {question_id} - Filter Output: {extracted_context[:100]}...") # Optional debug
    except Exception as e:
        print(f"Error during Filter Agent turn for ID {question_id}: {e}")
        extracted_context = f"Error running filter agent: {e}" # Store error


    try:
        # Prompt for Answer Agent, directly using string concatenation
        prompt_answer = f"""
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

Question:{question_content}
Info:{extracted_context}
"""
        weak_answer = get_teacher_agent_response(
            answer_agent, prompt_answer
        )
        # print(f"ID: {question_id} - Answer Output: {uncertain_answer[:100]}...") # Optional debug
    except Exception as e:
        print(f"Error during Answer Agent turn for ID {question_id}: {e}")
        uncertain_answer = f"Error generating answer: {e}" # Store error

    prompt_strong_answer=f'''
SYSTEM:{SYSTEM_PROMPT}
User:Question: {question_content}\nContext: {context}
'''
    strong_answer=get_answer_agent_response(answer_agent,prompt_strong_answer)


    # --- Store final result ---
    final_string=f"""SYSTEM：I will give you a question and two answers.Please output the answer which you think is better.
    Question:{question_content}
    Context:{context}
Answer1:{weak_answer}
Answer2:{strong_answer}
"""
    results.append(final_string)




with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved as {output_path}")

# 1. 准备实验数据
experiment_data = {
    "run_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "model_paths": {
        "filter_agent": MODEL_PATHS[args.models[0]],
        "answer_agent": MODEL_PATHS[args.models[1]], 
        "teacher_agent": MODEL_PATHS[args.models[2]]
    },
    "output_path": f"./results/answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
}

# 2. 写入JSON文件
with open("experiment_log.json", "w") as f:
    json.dump(experiment_data, f, indent=2)