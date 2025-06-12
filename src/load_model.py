from vllm import LLM, SamplingParams
from unsloth import FastLanguageModel
import torch

def load_fliter_agent_model(
    model_name: str,
    max_seq_length: int = 3072,
    load_in_4bit: bool = True,
    lora_rank: int = 64,
    gpu_memory_utilization: float = 0.4
):
    """
    加载并配置信息筛选模型
    
    Args:
        model_name: 模型路径/名称
        max_seq_length: 最大序列长度
        load_in_4bit: 是否启用4bit量化
        lora_rank: LoRA适配器秩
        gpu_memory_utilization: GPU显存利用率
    
    Returns:
        (model, tokenizer, sampling_params)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        device_map='auto'
    )
    
    sampling_params = SamplingParams(
        temperature=0.6,
        repetition_penalty=1.2,
        max_tokens=300,
        skip_special_tokens=True,
        stop=["User:","<Answer>"]
    )
    
    return model, tokenizer, sampling_params

def get_fliter_agent_response(
    model,
    prompt: str,
    sampling_params=None
) -> str:
    """
    执行单样本推理并返回原始模型输出文本
    
    Args:
        model: 已加载的模型
        prompt: 输入文本
        sampling_params: 生成参数（可选）
    
    Returns:
        模型生成的原始文本 (str)
    """
    # 默认生成参数
    if sampling_params is None:
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=1024,
            skip_special_tokens=True,
            stop=["User:"]
        )
    
    # 执行推理
    outputs = model.fast_generate([prompt], sampling_params)
    return outputs[0].outputs[0].text
def load_answer_agent_model(
    model_name: str,
    max_seq_length: int = 3072,
    load_in_4bit: bool = True,
    lora_rank: int = 64,
    gpu_memory_utilization: float = 0.4
):
    """
    加载并配置信息筛选模型
    
    Args:
        model_name: 模型路径/名称
        max_seq_length: 最大序列长度
        load_in_4bit: 是否启用4bit量化
        lora_rank: LoRA适配器秩
        gpu_memory_utilization: GPU显存利用率
    
    Returns:
        (model, tokenizer, sampling_params)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        device_map='auto'
    )
    
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=1024,
        skip_special_tokens=True,
        stop=["User:"]
    )
    
    return model, tokenizer, sampling_params

def get_answer_agent_response(
    model,
    prompt: str,
    sampling_params=None
) -> str:
    """
    执行单样本推理并返回原始模型输出文本
    
    Args:
        model: 已加载的模型
        prompt: 输入文本
        sampling_params: 生成参数（可选）
    
    Returns:
        模型生成的原始文本 (str)
    """
    # 默认生成参数
    if sampling_params is None:
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=256,
            skip_special_tokens=True,
            stop=["User:"]
        )
    
    # 执行推理
    outputs = model.fast_generate([prompt], sampling_params)
    return outputs[0].outputs[0].text
def load_teacher_agent_model(
    model_name: str,
    max_seq_length: int = 3072,
    load_in_4bit: bool = True,
    lora_rank: int = 64,
    gpu_memory_utilization: float = 0.4
):
    """
    加载并配置信息筛选模型
    
    Args:
        model_name: 模型路径/名称
        max_seq_length: 最大序列长度
        load_in_4bit: 是否启用4bit量化
        lora_rank: LoRA适配器秩
        gpu_memory_utilization: GPU显存利用率
    
    Returns:
        (model, tokenizer, sampling_params)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        device_map='auto'
    )
    
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=1024,
        skip_special_tokens=True,
        stop=["User:"]
    )
    
    return model, tokenizer, sampling_params

def get_teacher_agent_response(
    model,
    prompt: str,
    sampling_params=None
) -> str:
    """
    执行单样本推理并返回原始模型输出文本
    
    Args:
        model: 已加载的模型
        prompt: 输入文本
        sampling_params: 生成参数（可选）
    
    Returns:
        模型生成的原始文本 (str)
    """
    # 默认生成参数
    if sampling_params is None:
        sampling_params = SamplingParams(
            temperature=0.6,
            max_tokens=1024,
            skip_special_tokens=True,
            stop=["User:"]
        )
    
    # 执行推理
    outputs = model.fast_generate([prompt], sampling_params)
    return outputs[0].outputs[0].text