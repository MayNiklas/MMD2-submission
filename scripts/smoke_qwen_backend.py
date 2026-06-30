"""Run a short Qwen backend smoke test for the BRIGHTER decoder pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.brighter_emotion_pipeline_qwen import RunConfig
from src.brighter_emotion_pipeline_qwen import create_smoke_train_internal_dev_split
from src.brighter_emotion_pipeline_qwen import create_trainer
from src.brighter_emotion_pipeline_qwen import ensure_directories
from src.brighter_emotion_pipeline_qwen import evaluate_dataset
from src.brighter_emotion_pipeline_qwen import load_model
from src.brighter_emotion_pipeline_qwen import load_tokenizer
from src.brighter_emotion_pipeline_qwen import set_random_seed
from src.brighter_emotion_pipeline_qwen import tokenize_dataset


def build_smoke_config() -> RunConfig:
    """Build the short-runtime smoke-test configuration."""

    config: RunConfig = RunConfig(
        output_dir=PROJECT_ROOT / "outputs",
        reports_dir=PROJECT_ROOT / "reports",
        run_suffix="smoke",
        max_length=128,
        max_new_tokens=16,
        max_steps=2,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        lime_num_samples=4,
    )
    return config


def main() -> None:
    """Train Qwen for two steps and run a tiny generated-label evaluation."""

    config: RunConfig = build_smoke_config()
    set_random_seed(seed=config.seed)
    ensure_directories(config=config)
    tokenizer: Any = load_tokenizer(config=config)
    splits: Any = create_smoke_train_internal_dev_split(config=config)
    model: Any = load_model(config=config)
    tokenized_train_dataset: Any = tokenize_dataset(dataset=splits["train"], tokenizer=tokenizer, config=config)
    tokenized_internal_dev_dataset: Any = tokenize_dataset(
        dataset=splits["internal_dev"],
        tokenizer=tokenizer,
        config=config,
    )
    trainer: Any = create_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_train_dataset=tokenized_train_dataset,
        tokenized_internal_dev_dataset=tokenized_internal_dev_dataset,
        config=config,
    )
    train_result: Any = trainer.train()
    trainer.save_model(str(config.model_dir))
    tokenizer.save_pretrained(config.model_dir)
    metrics_frame: pd.DataFrame
    metrics_frame, _, _ = evaluate_dataset(
        model=trainer.model,
        tokenizer=tokenizer,
        dataset=splits["internal_dev"],
        split="smoke_internal_dev",
        config=config,
    )
    print(train_result)
    print(metrics_frame[["language_name", "macro_f1", "micro_f1", "samples_f1"]])
    print(f"Saved smoke model to {config.model_dir}")


if __name__ == "__main__":
    main()
