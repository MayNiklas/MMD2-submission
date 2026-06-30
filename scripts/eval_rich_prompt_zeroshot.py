"""Controlled A/B of the rich system prompt on the OFFICIAL dev split, incl. zero-shot languages.

The multi-seed sweep only scores the trained languages on internal-dev, so it could not see whether
the recall-oriented ``RICH_SYSTEM_PROMPT`` helps zero-shot transfer (deu/ptbr/yor)  -  the one place a
constant prompt is a language's only lever. This script closes that sub-question: it trains two
models that are identical except for the prompt (default vs ``--rich-prompt``), at the locked config
(lr=1e-5, 3 epochs, cosine+warmup), with the same seed, and evaluates BOTH on the whole official dev
split for every EVAL language. The only moving part is the prompt, so the per-language delta is clean.

Single run (a few minutes per model on one GPU)::

    conda activate mmd2-emotion
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_rich_prompt_zeroshot.py \
        2>&1 | tee outputs/rich_prompt_zeroshot_$(date +%Y%m%d_%H%M%S).log

Outputs:
- reports/tables/qwen3.5-0.8B_rich_prompt_zeroshot_dev.csv   -  per (prompt, language) official-dev F1
and a default-vs-rich comparison table printed to stdout, zero-shot languages highlighted.
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch

from src.brighter_emotion_pipeline_qwen import EVAL_LANGUAGES
from src.brighter_emotion_pipeline_qwen import SEED
from src.brighter_emotion_pipeline_qwen import ZERO_SHOT_LANGUAGES
from src.brighter_emotion_pipeline_qwen import RunConfig
from src.brighter_emotion_pipeline_qwen import create_train_internal_dev_split
from src.brighter_emotion_pipeline_qwen import create_trainer
from src.brighter_emotion_pipeline_qwen import ensure_directories
from src.brighter_emotion_pipeline_qwen import load_combined_eval_split
from src.brighter_emotion_pipeline_qwen import load_model
from src.brighter_emotion_pipeline_qwen import load_tokenizer
from src.brighter_emotion_pipeline_qwen import macro_f1_by_language
from src.brighter_emotion_pipeline_qwen import set_random_seed
from src.brighter_emotion_pipeline_qwen import tokenize_dataset


def train_and_eval(rich_prompt: bool, seed: int, eval_dataset, tokenizer) -> pd.DataFrame:
    """Train one model (locked config, given prompt) and score the official dev split per language."""

    variant: str = "rich" if rich_prompt else "default"
    config: RunConfig = RunConfig(
        output_dir=PROJECT_ROOT / "outputs",
        reports_dir=PROJECT_ROOT / "reports",
        run_suffix=f"rich_prompt_zeroshot_{variant}",
        seed=seed,
        learning_rate=1e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        select_best_model=False,
        rich_system_prompt=rich_prompt,
    )
    print(f"\n=== training variant={variant} (rich_system_prompt={rich_prompt}), seed={seed} ===", flush=True)
    set_random_seed(seed=config.seed)
    splits = create_train_internal_dev_split(config=config)
    model = load_model(config=config)
    tokenized_train = tokenize_dataset(dataset=splits["train"], tokenizer=tokenizer, config=config)
    tokenized_internal_dev = tokenize_dataset(dataset=splits["internal_dev"], tokenizer=tokenizer, config=config)
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_train_dataset=tokenized_train,
        tokenized_internal_dev_dataset=tokenized_internal_dev,
        config=config,
    )
    trainer.train()

    print(f"=== scoring variant={variant} on official dev ({len(eval_dataset)} examples) ===", flush=True)
    frame: pd.DataFrame = macro_f1_by_language(
        model=model, tokenizer=tokenizer, dataset=eval_dataset, config=config
    )
    frame.insert(0, "prompt", variant)
    frame["training_status"] = [
        "zero-shot" if language in ZERO_SHOT_LANGUAGES else "trained" for language in frame["language"]
    ]

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED, help="Shared RNG seed for both variants.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    arguments = parse_arguments()
    base_config = RunConfig(output_dir=PROJECT_ROOT / "outputs", reports_dir=PROJECT_ROOT / "reports")
    ensure_directories(config=base_config)
    tokenizer = load_tokenizer(config=base_config)

    # Load the official dev split once; both variants score the exact same examples.
    eval_dataset = load_combined_eval_split(languages=EVAL_LANGUAGES, split="dev", config=base_config)

    frames = [
        train_and_eval(rich_prompt=False, seed=arguments.seed, eval_dataset=eval_dataset, tokenizer=tokenizer),
        train_and_eval(rich_prompt=True, seed=arguments.seed, eval_dataset=eval_dataset, tokenizer=tokenizer),
    ]
    results = pd.concat(frames, ignore_index=True)
    out_path = base_config.tables_dir / "qwen3.5-0.8B_rich_prompt_zeroshot_dev.csv"
    results.to_csv(out_path, index=False)

    # Build a default-vs-rich comparison on macro-F1.
    pivot = results.pivot(index=["language", "language_name", "training_status"], columns="prompt", values="macro_f1")
    pivot = pivot.reset_index()
    pivot["delta"] = pivot["rich"] - pivot["default"]
    pivot = pivot.sort_values(["training_status", "language"])

    print("\n=== Official-dev macro-F1: default vs rich prompt ===")
    print(f"{'lang':>5} | {'status':>9} | {'default':>8} | {'rich':>8} | {'delta':>8}")
    for _, row in pivot.iterrows():
        print(
            f"{row['language']:>5} | {row['training_status']:>9} | "
            f"{row['default']:.4f} | {row['rich']:.4f} | {row['delta']:+.4f}"
        )
    for status in ("zero-shot", "trained"):
        sub = pivot[pivot["training_status"] == status]
        print(
            f"  mean {status:>9}: default {sub['default'].mean():.4f}  "
            f"rich {sub['rich'].mean():.4f}  delta {sub['rich'].mean() - sub['default'].mean():+.4f}"
        )
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
