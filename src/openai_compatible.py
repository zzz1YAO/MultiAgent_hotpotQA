from __future__ import annotations

from openai import OpenAI

try:
    from src.config import OpenAICompatibleConfig
except ImportError:
    from config import OpenAICompatibleConfig


def build_openai_client(config: OpenAICompatibleConfig) -> OpenAI:
    return OpenAI(api_key=config.resolved_api_key(), base_url=config.base_url)


def chat_completion(
    client: OpenAI,
    model_name: str,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    request_kwargs: dict[str, object] = {
        "model": model_name,
        "messages": messages,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content
    return content.strip() if content else ""
