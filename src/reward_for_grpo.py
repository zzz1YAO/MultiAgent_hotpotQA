from datasets import load_dataset
import re
from datasets import Dataset
import json


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()




def load_rlhf_dataset(file_path, system_prompt=SYSTEM_PROMPT):
    """
    加载JSON格式的HotpotQA数据并转换为HF Dataset格式
    
    Args:
        file_path: JSON文件路径
        system_prompt: 系统提示内容
        
    Returns:
        HuggingFace Dataset结构: {
            "prompt": [{"role": "system", "content": ...}, {"role": "user", "content": ...}],
            "answer": str
        }
    """
    def format_context(context):
        """将context格式化为自然文本"""
        return "\n".join(
            f"{title}:\n{'. '.join(sentences)}"
            for title, sentences in context
        )

    # 读取原始JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 构建HF Dataset兼容的数据结构
    hf_data = {
        "prompt": [],
        "answer": []
    }

    for item in raw_data:
        # 格式化上下文
        context_str = format_context(item["context"])
        
        # 构造对话式prompt
        hf_data["prompt"].append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {item['question']}\nContext: {context_str}"}
        ])
        hf_data["answer"].append(item["answer"])

    # 转换为HF Dataset
    return Dataset.from_dict(hf_data)

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]  # 添加 re.DOTALL
    return [0.5 if match else 0.0 for match in matches]

from collections import Counter


def normalize_answer(s):
    """标准化答案文本（小写、去标点、去冠词等）"""
    def remove_punc(text):
        return ''.join(ch for ch in text if ch.isalnum() or ch.isspace())
    def lower(text):
        return text.lower()
    return lower(remove_punc(s))

def calculate_f1(prediction, truth):
    """计算F1分数（优化版）"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    # 处理yes/no特殊情形
    if (pred_tokens in [['yes'], ['no'], ['noanswer']] or 
        truth_tokens in [['yes'], ['no'], ['noanswer']]):
        return float(pred_tokens == truth_tokens)  # 返回1.0或0.0
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return min(f1, 1.0)  # 确保不超过1.0

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """改进版奖励函数：基于F1分数*2"""
    rewards = []
    for prompt, completion, answer_ in zip(prompts, completions, answer):
        # 提取答案（假设completion是消息列表）
        response = completion[0]['content'] if isinstance(completion, list) else completion
        extracted_answer = extract_xml_answer(response)
        
        # 计算F1分数
        f1 = calculate_f1(extracted_answer, answer_)


        
        rewards.append(f1 * 3)  # 最终奖励 = F1 * 2
    
    return rewards

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

