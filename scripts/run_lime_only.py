"""Regenerate LIME outputs using saved fine-tuned models and existing predictions CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.brighter_emotion_pipeline import (
    LABEL_COLUMNS,
    RunConfig,
    explain_lime_examples,
    select_lime_examples,
)


def run_mbert_lime() -> None:
    print("=== mBERT LIME ===")
    config = RunConfig(
        output_dir=PROJECT_ROOT / "outputs",
        reports_dir=PROJECT_ROOT / "reports",
    )

    predictions_frame = pd.read_csv(config.tables_dir / "dev_predictions.csv")
    print(f"Loaded {len(predictions_frame)} prediction rows")

    examples = select_lime_examples(predictions_frame=predictions_frame, max_examples=6)
    print(f"Selected {len(examples)} LIME examples")
    print(examples[["language", "text"]].to_string())

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_dir,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
    )
    model.eval()
    print(f"Loaded fine-tuned mBERT from {config.model_dir}")

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
    run_mbert_lime()
