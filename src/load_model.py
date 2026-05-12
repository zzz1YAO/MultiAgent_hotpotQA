from __future__ import annotations

from typing import Any

from unsloth import FastLanguageModel
from vllm import SamplingParams


DEFAULT_LOAD_KWARGS = {
    "max_seq_length": 3072,
    "load_in_4bit": True,
    "lora_rank": 64,
    "gpu_memory_utilization": 0.4,
}

FILTER_SAMPLING_DEFAULTS = {
    "temperature": 0.6,
    "repetition_penalty": 1.2,
    "max_tokens": 300,
    "skip_special_tokens": True,
    "stop": ["User:", "<Answer>"],
}
ANSWER_SAMPLING_DEFAULTS = {
    "temperature": 0.6,
    "max_tokens": 1024,
    "skip_special_tokens": True,
    "stop": ["User:"],
}
ANSWER_RESPONSE_DEFAULTS = {
    "temperature": 0.6,
    "max_tokens": 256,
    "skip_special_tokens": True,
    "stop": ["User:"],
}
TEACHER_SAMPLING_DEFAULTS = {
    "temperature": 0.6,
    "max_tokens": 1024,
    "skip_special_tokens": True,
    "stop": ["User:"],
}


def build_sampling_params(**overrides: Any) -> SamplingParams:
    return SamplingParams(**overrides)


def load_agent_model(
    model_name: str,
    sampling_defaults: dict[str, Any],
    max_seq_length: int = DEFAULT_LOAD_KWARGS["max_seq_length"],
    load_in_4bit: bool = DEFAULT_LOAD_KWARGS["load_in_4bit"],
    lora_rank: int = DEFAULT_LOAD_KWARGS["lora_rank"],
    gpu_memory_utilization: float = DEFAULT_LOAD_KWARGS["gpu_memory_utilization"],
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        device_map="auto",
    )
    return model, tokenizer, build_sampling_params(**sampling_defaults)


def generate_response(
    model: Any,
    prompt: str,
    sampling_params: SamplingParams | None = None,
    default_sampling: dict[str, Any] | None = None,
) -> str:
    sampling = sampling_params or build_sampling_params(**(default_sampling or ANSWER_RESPONSE_DEFAULTS))
    outputs = model.fast_generate([prompt], sampling)
    return outputs[0].outputs[0].text


def load_fliter_agent_model(
    model_name: str,
    max_seq_length: int = DEFAULT_LOAD_KWARGS["max_seq_length"],
    load_in_4bit: bool = DEFAULT_LOAD_KWARGS["load_in_4bit"],
    lora_rank: int = DEFAULT_LOAD_KWARGS["lora_rank"],
    gpu_memory_utilization: float = DEFAULT_LOAD_KWARGS["gpu_memory_utilization"],
):
    return load_agent_model(
        model_name=model_name,
        sampling_defaults=FILTER_SAMPLING_DEFAULTS,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def get_fliter_agent_response(model: Any, prompt: str, sampling_params: SamplingParams | None = None) -> str:
    return generate_response(
        model,
        prompt,
        sampling_params=sampling_params,
        default_sampling=FILTER_SAMPLING_DEFAULTS,
    )


def load_filter_agent_model(
    model_name: str,
    max_seq_length: int = DEFAULT_LOAD_KWARGS["max_seq_length"],
    load_in_4bit: bool = DEFAULT_LOAD_KWARGS["load_in_4bit"],
    lora_rank: int = DEFAULT_LOAD_KWARGS["lora_rank"],
    gpu_memory_utilization: float = DEFAULT_LOAD_KWARGS["gpu_memory_utilization"],
):
    return load_fliter_agent_model(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def get_filter_agent_response(model: Any, prompt: str, sampling_params: SamplingParams | None = None) -> str:
    return get_fliter_agent_response(model, prompt, sampling_params=sampling_params)


def load_answer_agent_model(
    model_name: str,
    max_seq_length: int = DEFAULT_LOAD_KWARGS["max_seq_length"],
    load_in_4bit: bool = DEFAULT_LOAD_KWARGS["load_in_4bit"],
    lora_rank: int = DEFAULT_LOAD_KWARGS["lora_rank"],
    gpu_memory_utilization: float = DEFAULT_LOAD_KWARGS["gpu_memory_utilization"],
):
    return load_agent_model(
        model_name=model_name,
        sampling_defaults=ANSWER_SAMPLING_DEFAULTS,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def get_answer_agent_response(model: Any, prompt: str, sampling_params: SamplingParams | None = None) -> str:
    return generate_response(
        model,
        prompt,
        sampling_params=sampling_params,
        default_sampling=ANSWER_RESPONSE_DEFAULTS,
    )


def load_teacher_agent_model(
    model_name: str,
    max_seq_length: int = DEFAULT_LOAD_KWARGS["max_seq_length"],
    load_in_4bit: bool = DEFAULT_LOAD_KWARGS["load_in_4bit"],
    lora_rank: int = DEFAULT_LOAD_KWARGS["lora_rank"],
    gpu_memory_utilization: float = DEFAULT_LOAD_KWARGS["gpu_memory_utilization"],
):
    return load_agent_model(
        model_name=model_name,
        sampling_defaults=TEACHER_SAMPLING_DEFAULTS,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def get_teacher_agent_response(model: Any, prompt: str, sampling_params: SamplingParams | None = None) -> str:
    return generate_response(
        model,
        prompt,
        sampling_params=sampling_params,
        default_sampling=TEACHER_SAMPLING_DEFAULTS,
    )
