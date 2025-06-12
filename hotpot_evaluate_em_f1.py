import sys
import json
import re
import string
from collections import Counter

def normalize_answer(s):
    """标准化答案文本（移除非关键字符）"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, truth):
    """计算F1分数"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    # 处理yes/no特殊情形
    if (pred_tokens in [['yes'], ['no'], ['noanswer']] or 
        truth_tokens in [['yes'], ['no'], ['noanswer']]):
        return int(pred_tokens == truth_tokens)
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

def evaluate(pred_file, gold_file):
    """执行评估"""
    with open(pred_file) as f:
        preds = json.load(f)  # 格式: {"qid1": "answer1", "qid2": "answer2"}
    with open(gold_file) as f:
        golds = json.load(f)  # 格式: [{"_id": "qid1", "answer": "truth1"}, ...]
    
    # 转换为 {qid: answer} 格式
    gold_dict = {item["_id"]: item["answer"] for item in golds}
    
    total_em = 0
    total_f1 = 0
    count = 0
    
    for qid, pred_ans in preds.items():
        if qid not in gold_dict:
            print(f"警告: 预测中存在未知问题ID {qid}")
            continue
            
        truth_ans = gold_dict[qid]
        # 计算EM
        em = int(normalize_answer(pred_ans) == normalize_answer(truth_ans))
        # 计算F1
        f1 = f1_score(pred_ans, truth_ans)
        
        total_em += em
        total_f1 += f1
        count += 1
    
    print(f"评估结果 (共 {count} 条):")
    print(f"Exact Match (EM): {total_em/count:.4f}")
    print(f"F1 Score: {total_f1/count:.4f}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python hotpot_evaluate_em&f1.py [预测文件.json] [标准答案文件.json]")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])