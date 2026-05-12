from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_eval import BaselineEvalConfig, evaluate_baseline_model
from src.config import DEFAULT_RESULTS_DIR, OpenAICompatibleConfig


def main() -> None:
    config = BaselineEvalConfig(
        client_config=OpenAICompatibleConfig(
            model_name="Qwen/QwQ-32B-Preview",
            base_url="https://api.siliconflow.cn/v1",
            api_key_env="SILICONFLOW_API_KEY",
        ),
        output_path=DEFAULT_RESULTS_DIR / "results_qvq_32B_preview.json",
    )
    evaluate_baseline_model(config)


if __name__ == "__main__":
    main()
