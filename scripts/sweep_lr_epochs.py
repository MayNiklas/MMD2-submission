"""Sweep learning rate and epochs for the BRIGHTER Qwen decoder, tuning on internal-dev.

Each learning rate trains one fresh model for ``--max-epochs`` while a callback records internal-dev
macro-F1 after every epoch, so the epochs axis comes for free and the grid is ``len(lrs)`` trainings
rather than ``len(lrs) x len(epochs)``. The official dev split is never touched here  -  it stays
reserved for final reporting in the notebook.

Built for unattended overnight runs in tmux::

    tmux new -s sweep
    conda activate mmd2-emotion
    python scripts/sweep_lr_epochs.py --mode full --max-epochs 4 \
        --lrs 1e-5 2e-5 3e-5 2>&1 | tee outputs/sweep_$(date +%Y%m%d_%H%M%S).log
    # detach: Ctrl-b d ; reattach: tmux attach -t sweep

A quick directional sweep on the subsampled split (minutes, not hours)::

    python scripts/sweep_lr_epochs.py --mode lite --max-epochs 4 --lrs 1e-5 2e-5 3e-5 5e-5

Outputs (namespaced by ``--run-suffix``, default ``sweep`` / ``sweep_lite``):
- reports/tables/<prefix>lr_epoch_sweep.csv   -  tidy per-lr/epoch/language macro-F1
- reports/figures/<prefix>lr_epoch_sweep.png  -  mean macro-F1 vs epoch, one line per lr
and a best-setting summary printed to stdout. Sweep runs skip checkpointing (select_best_model=False),
so they do not write models or collide with the main run's artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.brighter_emotion_pipeline_qwen import SEED
from src.brighter_emotion_pipeline_qwen import RunConfig
from src.brighter_emotion_pipeline_qwen import plot_lr_epoch_sweep
from src.brighter_emotion_pipeline_qwen import run_lr_epoch_sweep


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Sweep learning rate / epochs for the Qwen emotion decoder (tunes on internal-dev).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[1e-5, 2e-5, 3e-5],
        help="Learning rates to try (one training run each).",
    )
    parser.add_argument("--max-epochs", type=int, default=4, help="Epochs per run; the curve covers 1..max.")
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="RNG seed for the run. Vary it across runs to measure the noise band / get error bars.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (L2 regularization) applied to every run in the sweep.",
    )
    parser.add_argument(
        "--rich-prompt",
        action="store_true",
        help="Use the recall-oriented system prompt with explicit rare-emotion cues (train + eval).",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Oversample rare-emotion training rows toward the target rate (train split only).",
    )
    parser.add_argument(
        "--oversample-target-rate",
        type=float,
        default=0.15,
        help="Target positive rate that rare emotions are oversampled toward (with --oversample).",
    )
    parser.add_argument(
        "--oversample-max-factor",
        type=int,
        default=5,
        help="Cap on how many times a rare-emotion row may be duplicated (with --oversample).",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "lite"),
        default="full",
        help="full: official train split (slow, reliable). lite: 80-example/language subsample (fast).",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=0,
        help="Internal-dev examples scored generatively per epoch. 0 (default) scores the whole "
        "internal-dev split; a positive value caps to a seeded sample to trade accuracy for speed.",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Artifact namespace. Defaults to 'sweep' (full) or 'sweep_lite' (lite).",
    )
    return parser.parse_args(argv)


def build_base_config(arguments: argparse.Namespace) -> tuple[RunConfig, bool]:
    """Build the base RunConfig for the sweep and whether to use the subsampled split.

    Learning rate and epoch count are overridden per run by the sweep, so they are left at defaults
    here. Throughput settings mirror the notebook's tuned ``full`` / ``lite`` configurations.
    """

    if arguments.mode == "lite":
        run_suffix: str = arguments.run_suffix or "sweep_lite"
        base_config: RunConfig = RunConfig(
            output_dir=PROJECT_ROOT / "outputs",
            reports_dir=PROJECT_ROOT / "reports",
            run_suffix=run_suffix,
            seed=arguments.seed,
            max_length=256,
            max_new_tokens=24,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            gradient_checkpointing=False,
            weight_decay=arguments.weight_decay,
            rich_system_prompt=arguments.rich_prompt,
            oversample_rare_emotions=arguments.oversample,
            oversample_target_rate=arguments.oversample_target_rate,
            oversample_max_factor=arguments.oversample_max_factor,
            smoke_train_examples_per_language=80,
            smoke_internal_dev_examples_per_language=20,
        )
        return base_config, True

    run_suffix = arguments.run_suffix or "sweep"
    base_config = RunConfig(
        output_dir=PROJECT_ROOT / "outputs",
        reports_dir=PROJECT_ROOT / "reports",
        run_suffix=run_suffix,
        seed=arguments.seed,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        weight_decay=arguments.weight_decay,
        rich_system_prompt=arguments.rich_prompt,
        oversample_rare_emotions=arguments.oversample,
        oversample_target_rate=arguments.oversample_target_rate,
        oversample_max_factor=arguments.oversample_max_factor,
    )
    return base_config, False


def report_results(results_frame: pd.DataFrame) -> None:
    """Print the mean-macro-F1 grid and the best (learning_rate, epoch) setting."""

    aggregated_frame: pd.DataFrame = results_frame.groupby(
        ["learning_rate", "epoch"], as_index=False
    )["macro_f1"].mean()
    grid: pd.DataFrame = aggregated_frame.pivot(index="epoch", columns="learning_rate", values="macro_f1")
    print("\nMean internal-dev macro-F1 (rows = epoch, cols = learning rate):")
    print(grid.to_string(float_format=lambda value: f"{value:.4f}"))
    best_row: pd.Series = aggregated_frame.sort_values("macro_f1", ascending=False).iloc[0]
    print(
        f"\nBest setting: learning_rate={best_row['learning_rate']:g}, "
        f"epoch={int(best_row['epoch'])}, mean macro-F1={best_row['macro_f1']:.4f}"
    )


def main() -> None:
    """CLI entry point."""

    arguments: argparse.Namespace = parse_arguments()
    base_config, use_subsampled_split = build_base_config(arguments=arguments)
    print(
        f"Sweep mode={arguments.mode} | lrs={[f'{lr:g}' for lr in arguments.lrs]} | "
        f"max_epochs={arguments.max_epochs} | eval_sample_size={arguments.eval_sample_size} | "
        f"artifact_prefix={base_config.artifact_prefix}",
        flush=True,
    )
    results_frame: pd.DataFrame = run_lr_epoch_sweep(
        base_config=base_config,
        learning_rates=arguments.lrs,
        max_epochs=arguments.max_epochs,
        use_subsampled_split=use_subsampled_split,
        eval_sample_size=arguments.eval_sample_size,
    )
    figure_path: Path = plot_lr_epoch_sweep(results_frame=results_frame, config=base_config)
    report_results(results_frame=results_frame)
    print(
        f"\nWrote {base_config.tables_dir / f'{base_config.artifact_prefix}lr_epoch_sweep.csv'}"
        f"\nWrote {figure_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
