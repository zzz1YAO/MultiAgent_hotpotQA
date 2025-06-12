import json
import re
import argparse
from pathlib import Path

def extract_answer(text):
    """
    按优先级提取答案：
    1. 匹配 <answer>...</answer> 中的内容
    2. 匹配 "answer:" 开头，直到 "You are a" 之间的内容
    3. 匹配 "The better answer is :" 或 "The final answer is :" 及其后面的整段内容
    4. 匹配 <reasoning>: 前的所有内容
    5. 原样返回（如果以上都不匹配）
    """
    # 规则1：匹配 <answer>...</answer> 中的内容
    answer_tag_pattern = r'<answer>([^<]*)</answer>'
    answer_tag_match = re.search(answer_tag_pattern, text, re.IGNORECASE)
    if answer_tag_match:
        return answer_tag_match.group(1).strip()

    # 规则2：匹配 "answer:" 开头，直到 "You are a" 之前的内容（非贪婪）
    answer_colon_pattern = r'answer:\s*(.*?)(?=\s*You are a)'
    answer_colon_match = re.search(answer_colon_pattern, text, re.IGNORECASE | re.DOTALL)
    if answer_colon_match:
        return answer_colon_match.group(1).strip()

    # 规则3：匹配 "The better answer is :" 或 "The final answer is :" 及其后面的整段内容（忽略大小写，匹配到文本末尾）
    better_final_pattern = r'(?i)(?:The better/final answer is\s*:|The final answer is\s*:)\s*(.*)'
    better_final_match = re.search(better_final_pattern, text, re.DOTALL)
    if better_final_match:
        return better_final_match.group(1).strip()

    # 规则4：匹配 <reasoning>: 前的所有内容（非贪婪）
    reasoning_pattern = r'^(.*?)(?=\n<reasoning>:)'
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    if reasoning_match:
        return reasoning_match.group(1).strip()

    # 规则5：原样返回
    return text.strip()

def process_json_file(input_file):
    """处理输入JSON文件并保存结果"""
    input_path = Path(input_file)
    output_file = input_path.with_name(f"{input_path.stem}_extracted{input_path.suffix}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = {k: extract_answer(v) for k, v in data.items()}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"Processing complete! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract answers from JSON file')
    parser.add_argument('input_file', help='Path to input JSON file')
    args = parser.parse_args()

    process_json_file(args.input_file)
