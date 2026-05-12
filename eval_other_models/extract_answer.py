from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_TEST_DATASET_PATH, OpenAICompatibleConfig
from src.logging_utils import get_logger
from src.openai_compatible import build_openai_client, chat_completion


logger = get_logger(__name__)

EXTRACTION_PROMPT = """Extract the concise answer from the text based on the given question.
Follow these rules:
1. If you find explicit markers like "Answer:" or <Answer>, extract what follows it
2. If the text directly answers the question, extract that part
3. If no clear answer is found, return the original text unchanged

Question: {question}
Text: {text}

Extracted answer:"""

CLIENT_CONFIG = OpenAICompatibleConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    base_url="https://api.siliconflow.cn/v1",
    api_key_env="SILICONFLOW_API_KEY",
    system_prompt="You are a precise answer extraction assistant.",
    temperature=0.1,
    max_tokens=300,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract answers using LLM with question context")
    parser.add_argument("input_file", type=Path, help="Path to input JSON file to extract answers from")
    parser.add_argument(
        "--question-file",
        type=Path,
        default=DEFAULT_TEST_DATASET_PATH,
        help="Path to HotpotQA question file",
    )
    return parser.parse_args()


def load_question_mapping(question_file: Path) -> dict[str, str]:
    with question_file.open("r", encoding="utf-8") as handle:
        hotpot_data = json.load(handle)
    return {item["_id"]: item["question"] for item in hotpot_data}


def main() -> None:
    args = parse_args()
    output_file = args.input_file.with_name(f"{args.input_file.stem}_llm_extract{args.input_file.suffix}")

    with args.input_file.open("r", encoding="utf-8") as handle:
        answer_data = json.load(handle)

    question_dict = load_question_mapping(args.question_file)
    client = build_openai_client(CLIENT_CONFIG)
    results: dict[str, str] = {}

    for qid, answer_text in tqdm(answer_data.items(), desc="Extracting answers"):
        question_content = question_dict.get(qid, "Unknown question")
        try:
            results[qid] = chat_completion(
                client=client,
                model_name=CLIENT_CONFIG.model_name,
                user_prompt=EXTRACTION_PROMPT.format(question=question_content, text=answer_text),
                system_prompt=CLIENT_CONFIG.system_prompt,
                temperature=CLIENT_CONFIG.temperature,
                max_tokens=CLIENT_CONFIG.max_tokens,
            )
        except Exception as exc:
            logger.error("Error processing %s: %s", qid, exc)
            results[qid] = answer_text

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    logger.info("Extraction complete! Results saved to %s", output_file)


if __name__ == "__main__":
    main()
