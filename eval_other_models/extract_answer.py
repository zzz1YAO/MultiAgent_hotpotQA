import json
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
import argparse

# 配置API
api_key = "sk-itxnszoqzhtlbpqrdtufqywcaosvxzshakrajyqzggbyqjrq"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

# 命令行参数解析
parser = argparse.ArgumentParser(description='Extract answers using LLM with question context')
parser.add_argument('input_file', type=str, help='Path to input JSON file to extract answers from')
args = parser.parse_args()

# 文件路径处理
input_path = Path(args.input_file)
output_file = input_path.with_name(f"{input_path.stem}_llm_extract{input_path.suffix}")

# 加载待提取的数据
with open(input_path, 'r', encoding='utf-8') as f:
    answer_data = json.load(f)

# 加载HotpotQA测试集获取问题内容
with open('/home/liurt/disk3/Multi_agent/hotpotqa_test_400.json', 'r', encoding='utf-8') as f:
    hotpot_data = json.load(f)

# 创建问题ID到内容的映射
question_dict = {item["_id"]: item["question"] for item in hotpot_data}

# 提取提示模板
EXTRACTION_PROMPT = """Extract the concise answer from the text based on the given question. 
Follow these rules:
1. If you find explicit markers like "Answer:" or <Answer>, extract what follows it
2. If the text directly answers the question, extract that part
3. If no clear answer is found, return the original text unchanged


Text: {text}

Extracted answer:"""

# 处理每个问答对
results = {}
for qid, answer_text in tqdm(answer_data.items(), desc="Extracting answers"):
    try:
        # 获取对应的问题内容
        question_content = question_dict.get(qid, "Unknown question")
        
        # 调用API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise answer extraction assistant."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(
                    question=question_content,
                    text=answer_text
                )}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        results[qid] = response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error processing {qid}: {str(e)}")
        results[qid] = answer_text  # 回退到原始答案

# 保存结果
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Extraction complete! Results saved to {output_file}")