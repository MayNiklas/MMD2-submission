"""Train a single Qwen model for the scaling study and save weights to disk.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_scaling_save.py --model Qwen/Qwen3.5-2B
    CUDA_VISIBLE_DEVICES=1 python scripts/train_scaling_save.py --model Qwen/Qwen3.5-4B
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.brighter_emotion_pipeline_qwen import (  # noqa: E402
    EVAL_LANGUAGES,
    SCALING_SIZE_SETTINGS,
    RunConfig,
    _finetune_unit,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="e.g. Qwen/Qwen3.5-2B")
args = parser.parse_args()

model_name: str = args.model
batch, accumulation, checkpointing = SCALING_SIZE_SETTINGS[model_name]

base_config = RunConfig(
    output_dir=PROJECT_ROOT / "outputs",
    reports_dir=PROJECT_ROOT / "reports",
    num_train_epochs=3,
    per_device_train_batch_size=batch,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=accumulation,
    gradient_checkpointing=checkpointing,
    optim="paged_adamw_8bit",
)

print(f"Training {model_name} | batch={batch} accum={accumulation} checkpointing={checkpointing}", flush=True)

_finetune_unit(
    model_name=model_name,
    eval_languages=EVAL_LANGUAGES,
    base_config=base_config,
    seed=base_config.seed,
    epochs=3,
    learning_rate=1e-5,
    run_suffix=f"scaling_ft_{model_name.rsplit('/', maxsplit=1)[-1]}",
    save_model=True,
)

print(f"\nDone — {model_name} saved.", flush=True)
