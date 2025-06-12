import json
import re

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

def refine_for_fliter_input(data_item: dict) -> str:
    """
    将预处理后的数据润色为更自然的LLM输入格式
    
    Args:
        data_item: preprocess_hotpot_file()输出的单个元素
        
    Returns:
        润色后的自然语言提示字符串
    """
    context = data_item['model_input']['context']
    question = data_item['model_input']['question']
    
    return f"""System:Respond in following format:<Info>...</Info><Answer>...</Answer>
{context}
Of the information above, find out the original text that is of most concern to the following question:
{question}
Put them into <Info></Info>
"""

def extract_info_from_response(response_text: str) -> str:
    """
    从模型响应文本中提取<Info>标签内的内容
    
    Args:
        response_text: 包含<Info>标签的文本字符串
        
    Returns:
        <Info>标签内的内容字符串（不包含标签本身）
        如果未找到则返回空字符串
    """
    pattern = r'<Info>(.*?)</Info>'
    match = re.search(pattern, response_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def refine_for_teacher_judge_fliter(filtered_info: str, data_item: dict) -> str:
    """
    生成评估信息相关性的提示
    
    Args:
        filtered_info: 从模型响应中提取的信息片段
        data_item: 原始数据项(包含context和question)
        
    Returns:
        格式化后的评估提示字符串
    """
    context = data_item['model_input']['context']
    question = data_item['model_input']['question']
    
    return f"""Context:
{context}

Question: {question}

Please decide whether the info given below is from the content above, and whether it is strongly related to the question. If both conditions are met, answer 'yes', otherwise 'no'.

Info: {filtered_info}
"""

def teacher_agent_judge(response: str) -> str:
    """
    根据输入字符串判断返回YES或NO
    
    Args:
        response: 待判断的字符串
        
    Returns:
        "YES" 如果字符串中包含'yes'(不区分大小写)
        "NO" 如果字符串中包含'no'(不区分大小写)
        默认返回"NO"
    """
    response_lower = response.lower()
    if 'yes' in response_lower:
        return "YES"
    elif 'no' in response_lower:
        return "NO"
    return "YES"  # 默认情况

def refine_for_answer_agent(filtered_info: str, question_content: str,context) -> str:
    """
    生成信息相关性评估的标准提示
    
    Args:
        filtered_info: 从内容中提取的关键信息片段
        question_content: 需要评估的问题
        
    Returns:
        格式化后的评估提示字符串
    """
    return f"""\nUser:
    Question: {question_content}
    Key infomation:{filtered_info}
    Other Info:{context}
Answer:"""



def refine_for_teacher_judge_answer(prompt: str, answer: str) -> str:
    """
    Generate an evaluation prompt for judging the reasonableness of a QA pair
    
    Args:
        prompt: The question being evaluated
        answer: The answer to be judged
        
    Returns:
        Formatted evaluation prompt string
    """
    return f"""Please decide whether this Question-Answer Pair is reasonable, and whether there is any obvious reasoning error.
If there is no error (reasonable), please output 'yes', otherwise please output 'no'.

Question: {prompt}
Answer: {answer}

Judgment:"""

def transform_hotpotqa_to_sft(input_file, output_file):
    """
    将HotpotQA原始数据转换为SFT格式
    
    参数:
        input_file: 原始HotpotQA JSON文件路径
        output_file: 输出JSON文件路径
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换数据
    sft_data = []
    for item in data:
        # 提取问题
        question = item['question']
        
        # 构建上下文字符串
        context_parts = []
        for title, sentences in item['context']:
            context_parts.append(f"Title: {title}")
            context_parts.extend(sentences)
        context = "\n".join(context_parts)
        
        # 提取支持事实的完整句子
        support_sentences = []
        for fact in item['supporting_facts']:
            title, sent_idx = fact
            # 在上下文中找到对应的标题和句子
            for ctx_title, sentences in item['context']:
                if ctx_title == title and sent_idx < len(sentences):
                    support_sentences.append(sentences[sent_idx])
                    break
        
        # 构建输入输出对
        sft_item = {
            "input": f"{question}\n{context}",
            "output": "\n".join(support_sentences)
        }
        sft_data.append(sft_item)
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

import json
import re # Import the regular expression module
import difflib

def extract_entity_perspective(text):
    """
    Extracts 'entity' and 'perspective' from a string that may contain
    JSON-like content, even if it has extra surrounding text or minor
    formatting issues.

    Prioritizes parsing a valid JSON object if one can be isolated.
    Falls back to regular expressions if JSON parsing fails.

    Args:
        text: The input string from the Teacher Agent.

    Returns:
        A tuple containing (entity, perspective). Returns (None, None)
        if neither method successfully extracts both values.
    """
    entity = None
    perspective = None

    # --- Method 1: Attempt to isolate and parse JSON object ---
    try:
        # Find the first '{' and the last '}'
        start_index = text.find('{')
        end_index = text.rfind('}') # Use rfind to find the last occurrence

        if start_index != -1 and end_index != -1 and start_index < end_index:
            # Extract the substring between the first { and the last }
            json_substring = text[start_index : end_index + 1]

            # Attempt to parse the substring as JSON
            data = json.loads(json_substring)

            # Extract values if parsing was successful
            entity = data.get("entity")
            perspective = data.get("perspective")

            # If both are found via JSON, we're done
            if entity is not None and perspective is not None:
                # print("Extraction Method: JSON Parsing") # Optional debug
                return entity, perspective

    except json.JSONDecodeError:
        # JSON parsing failed, continue to fallback method
        # print("JSON parsing failed, attempting regex fallback.") # Optional debug
        pass # Don't return, proceed to regex
    except Exception as e:
        # Handle any other unexpected errors during JSON parsing attempt
        print(f"An unexpected error occurred during JSON parsing attempt: {e}")
        pass # Don't return, proceed to regex


    # --- Method 2: Fallback using Regular Expressions ---
    # This is less strict and can find patterns even if the overall structure is broken.
    # Pattern to find '"key": "value"' (non-greedy match for the value)
    entity_pattern = r'"entity":\s*"(.*?)"'
    perspective_pattern = r'"perspective":\s*"(.*?)"'

    try:
        # Search for the entity pattern
        entity_match = re.search(entity_pattern, text)
        if entity_match:
            entity = entity_match.group(1) # group(1) captures the content inside the quotes

        # Search for the perspective pattern
        perspective_match = re.search(perspective_pattern, text)
        if perspective_match:
            perspective = perspective_match.group(1) # group(1) captures the content inside the quotes

        # print("Extraction Method: Regex Fallback") # Optional debug
        return entity, perspective

    except Exception as e:
        print(f"An unexpected error occurred during regex fallback: {e}")
        return None, None # Return None, None on any error in regex

    # If neither method worked well enough (e.g., regex didn't find both), we reach here
    # print("Warning: Could not fully extract entity and perspective using either method.") # Optional debug
    # The current values (potentially None) will be returned
    return entity, perspective


import re
from difflib import SequenceMatcher

def extract_matching_info(data_string, query_string, fuzzy_threshold=0.8):
    """
    从总信息字符串中提取匹配查询字符串的信息
    
    Args:
        data_string (str): 包含所有信息的长字符串（格式：title:\n content\n. title:\n content...）
        query_string (str): 查询字符串（例如："我想要知道Melanie Oudin和Ted Schroeder"）
        fuzzy_threshold (float): 模糊匹配阈值，默认0.8
    
    Returns:
        str: 提取到的匹配信息字符串，如果没有匹配则返回空字符串
    """
    
    def parse_data_string(data_str):
        """解析数据字符串，提取title和内容"""
        players_info = {}
        entries = data_str.split('\n. ')
        
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
                
            colon_pos = entry.find(':\n')
            if colon_pos != -1:
                title = entry[:colon_pos].strip()
                content = entry[colon_pos + 2:].strip()
                
                # 清理title开头可能的点号
                if title.startswith('. '):
                    title = title[2:]
                
                players_info[title] = content
        
        return players_info
    
    def find_matches(query_str, info_dict, threshold):
        """查找匹配的信息"""
        matched_info = {}
        query_lower = query_str.lower()
        
        for title, content in info_dict.items():
            title_lower = title.lower()
            
            # 精确匹配
            if title_lower in query_lower:
                matched_info[title] = content
                continue
            
            # 模糊匹配
            query_words = re.findall(r'\b\w+\b', query_lower)
            title_words = re.findall(r'\b\w+\b', title_lower)
            
            for title_word in title_words:
                if len(title_word) < 3:  # 跳过过短的词
                    continue
                for query_word in query_words:
                    if len(query_word) < 3:  # 跳过过短的词
                        continue
                    similarity = SequenceMatcher(None, title_word, query_word).ratio()
                    if similarity >= threshold:
                        matched_info[title] = content
                        break
                if title in matched_info:
                    break
        
        return matched_info
    
    # 解析数据
    parsed_info = parse_data_string(data_string)
    
    # 查找匹配
    matches = find_matches(query_string, parsed_info, fuzzy_threshold)
    
    # 构建结果字符串
    if not matches:
        return ""
    
    result_parts = []
    for title, content in matches.items():
        result_parts.append(f"{title}:\n{content}")
    
    return "\n. ".join(result_parts)