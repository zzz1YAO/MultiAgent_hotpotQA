import openai
import json
import os
from tqdm import tqdm
from datetime import datetime
from time import sleep
from openai import OpenAI
def preprocess_hotpot_file(file_path):
    """
    处理HotpotQA数据集的预处理函数
    context结构: [[title, [sentences]], ...] 的二维列表
    supporting_facts结构: [[title, sent_id], ...] 的二维列表

    返回: [{
        'id': str,
        'model_input': {'context': str, 'question': str},
        'answer': str,
        'type': str,
        'level': str,
        'supporting_facts': list  # 实际为[[title, sent_id], ...]
    }, ...]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed = []
    for item in data:
        # 处理context结构（现在是[title, sentences_list]的列表）
        context_parts = []
        for para in item['context']:
            title = para[0]  # 第一个元素是标题
            sentences = " ".join(para[1])  # 第二个元素是句子列表
            context_parts.append(f"{title}:\n {sentences}\n")
        
        context = ". ".join(context_parts)  # 用句号+空格连接不同段落
        
        # 构建新字典
        processed.append({
            'id': item['_id'],
            'model_input': {
                'context': context,
                'question': item['question']
            },
            'answer': item['answer'],
            'type': item['type'],
            'level': item['level'],
            'supporting_facts': item['supporting_facts']
        })
    
    return processed
# 配置OpenAI API
api_key="sk-itxnszoqzhtlbpqrdtufqywcaosvxzshakrajyqzggbyqjrq"  # 从环境变量读取
MODEL_NAME = "Qwen/QwQ-32B-Preview"  # 可替换为 "gpt-4"
client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

#==============LOAD DATASET============================
test_dataset_path="../hotpotqa_test_400.json"
dataset=preprocess_hotpot_file(test_dataset_path)

# 初始化结果字典
results = {}

def call_openai_api(prompt):
    """调用OpenAI API的封装函数"""
    try:
        response =  client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],

        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        sleep(5)  # 失败时暂停
        return None

# 主处理循环
for test_sample in tqdm(dataset, desc="Processing"):
    # 构建prompt（直接使用model_input）
    prompt = str(test_sample["model_input"])
    
    # 调用API
    answer = call_openai_api("System:Answer questions as briefly as possible"+prompt)

    
    # 保存结果
    if answer:
        results[test_sample["id"]] = answer
    
    # 每隔20条保存一次
    if len(results) % 20 == 0:
        with open("temp_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

# 最终保存（带时间戳）
output_path = f"results_qvq_32B_preview.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)

print(f"处理完成！结果保存至: {output_path}")
print(f"共处理 {len(results)} 条数据")