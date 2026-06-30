"""Regenerate Qwen LIME outputs using the saved fine-tuned model and existing predictions CSV."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pin to GPU 0; override with CUDA_VISIBLE_DEVICES before launching.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.brighter_emotion_pipeline_qwen import (
    RunConfig,
    explain_lime_examples,
    select_lime_examples,
)


def run_qwen_lime() -> None:
    print("=== Qwen LIME ===")
    config = RunConfig(
        output_dir=PROJECT_ROOT / "outputs",
        reports_dir=PROJECT_ROOT / "reports",
    )
    print(f"model_dir : {config.model_dir}")
    print(f"artifact_prefix: {config.artifact_prefix}")

    predictions_frame = pd.read_csv(config.tables_dir / f"{config.artifact_prefix}dev_predictions.csv")
    print(f"Loaded {len(predictions_frame)} prediction rows")

    examples = select_lime_examples(predictions_frame=predictions_frame, max_examples=6)
    print(f"Selected {len(examples)} LIME examples")
    print(examples[["language", "text"]].to_string())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {config.model_dir} onto {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(config.model_dir),
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    paths = explain_lime_examples(
        model=model,
        tokenizer=tokenizer,
        examples_frame=examples,
        config=config,
    )
    print(f"Wrote {len(paths)} LIME files")
    for p in paths:
        print(f"  {p}")
        png = p.with_suffix(".png")
        if png.exists():
            print(f"  {png}")


if __name__ == "__main__":
    run_qwen_lime()
