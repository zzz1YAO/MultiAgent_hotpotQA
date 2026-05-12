from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_answer(text: str) -> str:
    answer_tag_match = re.search(r"<answer>([^<]*)</answer>", text, re.IGNORECASE)
    if answer_tag_match:
        return answer_tag_match.group(1).strip()

    answer_colon_match = re.search(
        r"answer:\s*(.*?)(?=\s*You are a)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if answer_colon_match:
        return answer_colon_match.group(1).strip()

    better_final_match = re.search(
        r"(?i)(?:The better/final answer is\s*:|The final answer is\s*:)\s*(.*)",
        text,
        re.DOTALL,
    )
    if better_final_match:
        return better_final_match.group(1).strip()

    reasoning_match = re.search(r"^(.*?)(?=\n<reasoning>:)", text, re.DOTALL)
    if reasoning_match:
        return reasoning_match.group(1).strip()

    return text.strip()


def process_json_file(input_file: str | Path) -> Path:
    input_path = Path(input_file)
    output_file = input_path.with_name(f"{input_path.stem}_extracted{input_path.suffix}")

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    processed_data = {key: extract_answer(value) for key, value in data.items()}

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(processed_data, handle, ensure_ascii=False, indent=2)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract answers from JSON file")
    parser.add_argument("input_file", help="Path to input JSON file")
    args = parser.parse_args()

    output_file = process_json_file(args.input_file)
    print(f"Processing complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
