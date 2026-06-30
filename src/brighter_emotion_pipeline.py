"""Minimal BRIGHTER multilingual emotion classification pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, Trainer, TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

SEED: int = 42069
MODEL_NAME: str = "google-bert/bert-base-multilingual-cased"
DATASET_NAME: str = "brighter-dataset/BRIGHTER-emotion-categories"
TRAIN_LANGUAGES: tuple[str, ...] = ("eng", "swa", "ukr")
EVAL_LANGUAGES: tuple[str, ...] = ("eng", "swa", "ukr", "deu", "ptbr", "yor")
ZERO_SHOT_LANGUAGES: tuple[str, ...] = ("deu", "ptbr", "yor")
LABEL_COLUMNS: tuple[str, ...] = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
LANGUAGE_NAMES: dict[str, str] = {
    "deu": "German",
    "eng": "English",
    "ptbr": "Brazilian Portuguese",
    "swa": "Swahili",
    "ukr": "Ukrainian",
    "yor": "Yoruba",
}


@dataclass(frozen=True)
class RunConfig:
    """Configuration for the MVP experiment."""

    seed: int = SEED
    model_name: str = MODEL_NAME
    dataset_name: str = DATASET_NAME
    output_dir: Path = Path("outputs")
    reports_dir: Path = Path("reports")
    max_length: int = 128
    internal_dev_size: float = 0.15
    threshold: float = 0.5
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    weight_decay: float = 0.01

    @property
    def model_dir(self) -> Path:

        return self.output_dir / "model"

    @property
    def logging_dir(self) -> Path:

        return self.output_dir / "logs"

    @property
    def figures_dir(self) -> Path:

        return self.reports_dir / "figures"

    @property
    def tables_dir(self) -> Path:

        return self.reports_dir / "tables"

    @property
    def lime_dir(self) -> Path:

        return self.reports_dir / "lime"


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible MVP runs."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def ensure_directories(config: RunConfig) -> None:
    """Create required output directories."""

    directories: tuple[Path, ...] = (
        config.output_dir,
        config.model_dir,
        config.logging_dir,
        config.figures_dir,
        config.tables_dir,
        config.lime_dir,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def language_display_name(language: str) -> str:

    return LANGUAGE_NAMES[language]


def load_language_split(language: str, split: str, config: RunConfig) -> Dataset:
    """Load one BRIGHTER language split and attach the language code."""

    dataset: Dataset = load_dataset(config.dataset_name, language, split=split)
    language_values: list[str] = [language] * len(dataset)
    dataset = dataset.add_column("language", language_values)
    return dataset


def load_combined_split(languages: Iterable[str], split: str, config: RunConfig) -> Dataset:
    """Load and concatenate one split for several languages."""

    datasets: list[Dataset] = [
        load_language_split(language=language, split=split, config=config) for language in languages
    ]
    combined_dataset: Dataset = concatenate_datasets(datasets)
    return combined_dataset


def create_train_internal_dev_split(config: RunConfig) -> DatasetDict:
    """Split official train data into train and internal dev splits."""

    official_train_dataset: Dataset = load_combined_split(languages=TRAIN_LANGUAGES, split="train", config=config)
    split_dataset: DatasetDict = official_train_dataset.train_test_split(
        test_size=config.internal_dev_size,
        seed=config.seed,
        shuffle=True,
    )
    renamed_dataset: DatasetDict = DatasetDict(
        {
            "train": split_dataset["train"],
            "internal_dev": split_dataset["test"],
        }
    )
    return renamed_dataset


def build_label_matrix(batch: dict[str, list[Any]]) -> list[list[float]]:
    """Build a multi-label target matrix from BRIGHTER emotion lists."""

    labels: list[list[float]] = []
    row_count: int = len(batch["text"])
    for row_index in range(row_count):
        row_emotions: set[str] = set(batch["emotions"][row_index])
        row_labels: list[float] = [1.0 if label in row_emotions else 0.0 for label in LABEL_COLUMNS]
        labels.append(row_labels)
    return labels


def compute_language_label_positive_weights(dataset: Dataset) -> dict[str, dict[str, float]]:
    """Compute inverse-prevalence positive weights per language and label."""

    dataframe: pd.DataFrame = dataset.to_pandas()
    language_label_positive_weights: dict[str, dict[str, float]] = {}
    for language, language_frame in dataframe.groupby("language"):
        language_weights: dict[str, float] = {}
        row_count: int = len(language_frame)
        for label in LABEL_COLUMNS:
            positive_count: int = sum(label in emotions for emotions in language_frame["emotions"])
            negative_count: int = row_count - positive_count
            positive_weight: float = 1.0 if positive_count == 0 else float(negative_count / positive_count)
            language_weights[label] = positive_weight
        language_label_positive_weights[str(language)] = language_weights
    return language_label_positive_weights


def build_loss_weight_matrix(
    batch: dict[str, list[Any]],
    labels: list[list[float]],
    language_label_positive_weights: dict[str, dict[str, float]],
) -> list[list[float]]:
    """Build language-aware positive-label loss weights."""

    loss_weights: list[list[float]] = []
    for row_index, row_labels in enumerate(labels):
        language: str = str(batch["language"][row_index])
        row_weights: list[float] = []
        for label, label_value in zip(LABEL_COLUMNS, row_labels, strict=True):
            row_weight: float = language_label_positive_weights[language][label] if label_value == 1.0 else 1.0
            row_weights.append(row_weight)
        loss_weights.append(row_weights)
    return loss_weights


def tokenize_batch(
    batch: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    language_label_positive_weights: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Tokenize a BRIGHTER batch and attach multi-label targets."""

    tokenized_batch: dict[str, Any] = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    labels: list[list[float]] = build_label_matrix(batch=batch)
    tokenized_batch["labels"] = labels
    tokenized_batch["loss_weights"] = build_loss_weight_matrix(
        batch=batch,
        labels=labels,
        language_label_positive_weights=language_label_positive_weights,
    )
    return tokenized_batch


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    config: RunConfig,
) -> Dataset:
    """Tokenize a dataset for Hugging Face Trainer."""

    language_label_positive_weights: dict[str, dict[str, float]] = compute_language_label_positive_weights(
        dataset=dataset
    )
    tokenized_dataset: Dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": config.max_length,
            "language_label_positive_weights": language_label_positive_weights,
        },
    )
    return tokenized_dataset


def load_tokenizer(config: RunConfig) -> PreTrainedTokenizerBase:
    """Load the configured tokenizer."""

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model_name)
    return tokenizer


def load_model(config: RunConfig) -> PreTrainedModel:
    """Load the configured multilingual transformer for multi-label classification."""

    id2label: dict[int, str] = {label_index: label for label_index, label in enumerate(LABEL_COLUMNS)}
    label2id: dict[str, int] = {label: label_index for label_index, label in enumerate(LABEL_COLUMNS)}
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(LABEL_COLUMNS),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )
    return model


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Apply sigmoid to logits."""

    probabilities: np.ndarray = 1.0 / (1.0 + np.exp(-logits))
    return probabilities


def metric_dict_from_arrays(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute macro, micro, sample, and per-label F1 metrics."""

    predictions: np.ndarray = (probabilities >= threshold).astype(int)
    metrics: dict[str, float] = {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(labels, predictions, average="samples", zero_division=0)),
    }
    per_label_f1: np.ndarray = f1_score(labels, predictions, average=None, zero_division=0)
    for label, label_f1 in zip(LABEL_COLUMNS, per_label_f1, strict=True):
        metrics[f"f1_{label}"] = float(label_f1)
    return metrics


def compute_trainer_metrics(eval_prediction: EvalPrediction, threshold: float = 0.5) -> dict[str, float]:
    """Compute metrics for Hugging Face Trainer."""

    logits: np.ndarray = eval_prediction.predictions
    labels: np.ndarray = eval_prediction.label_ids
    probabilities: np.ndarray = sigmoid(logits=logits)
    metrics: dict[str, float] = metric_dict_from_arrays(
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
    )
    return metrics


class ImbalanceAwareTrainer(Trainer):
    """Trainer with language-aware weighted binary cross-entropy."""

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute weighted multi-label BCE loss."""

        labels: torch.Tensor = inputs.pop("labels")
        loss_weights: torch.Tensor = inputs.pop("loss_weights")
        outputs: Any = model(**inputs)
        logits: torch.Tensor = outputs.logits
        loss_matrix: torch.Tensor = F.binary_cross_entropy_with_logits(
            logits,
            labels.to(dtype=logits.dtype),
            reduction="none",
        )
        weighted_loss_matrix: torch.Tensor = loss_matrix * loss_weights.to(device=logits.device, dtype=logits.dtype)
        loss: torch.Tensor = weighted_loss_matrix.mean()
        if return_outputs:
            return loss, outputs
        return loss


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokenized_train_dataset: Dataset,
    tokenized_internal_dev_dataset: Dataset,
    config: RunConfig,
) -> Trainer:
    """Create a Hugging Face Trainer for the MVP experiment."""

    training_arguments: TrainingArguments = TrainingArguments(
        output_dir=str(config.output_dir),
        logging_dir=str(config.logging_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer: Trainer = ImbalanceAwareTrainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_internal_dev_dataset,
        processing_class=tokenizer,
        compute_metrics=lambda eval_prediction: compute_trainer_metrics(
            eval_prediction=eval_prediction,
            threshold=config.threshold,
        ),
    )
    return trainer


def describe_training_data(config: RunConfig) -> pd.DataFrame:
    """Summarize official train data for the training languages."""

    rows: list[dict[str, float | int | str]] = []
    for language in TRAIN_LANGUAGES:
        dataset: Dataset = load_language_split(language=language, split="train", config=config)
        dataframe: pd.DataFrame = dataset.to_pandas()
        row: dict[str, float | int | str] = {
            "language": language,
            "language_name": language_display_name(language=language),
            "rows": int(len(dataframe)),
        }
        for label in LABEL_COLUMNS:
            positive_count: int = sum(label in emotions for emotions in dataframe["emotions"])
            row[f"{label}_positive_rate"] = float(positive_count / len(dataframe))
        rows.append(row)
    overview: pd.DataFrame = pd.DataFrame(rows)
    overview.to_csv(config.tables_dir / "training_data_overview.csv", index=False)
    return overview


def build_prediction_frame(
    dataset: Dataset,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Build a per-example prediction table."""

    dataframe: pd.DataFrame = dataset.to_pandas()
    predictions: np.ndarray = (probabilities >= threshold).astype(int)
    for label_index, label in enumerate(LABEL_COLUMNS):
        true_values: list[int] = [int(label in emotions) for emotions in dataframe["emotions"]]
        dataframe[f"true_{label}"] = true_values
        dataframe[f"pred_{label}"] = predictions[:, label_index].astype(int)
        dataframe[f"prob_{label}"] = probabilities[:, label_index].astype(float)
    true_columns: list[str] = [f"true_{label}" for label in LABEL_COLUMNS]
    pred_columns: list[str] = [f"pred_{label}" for label in LABEL_COLUMNS]
    dataframe["has_error"] = (dataframe[true_columns].to_numpy() != dataframe[pred_columns].to_numpy()).any(axis=1)
    return dataframe


def evaluate_one_language(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    language: str,
    split: str,
    config: RunConfig,
) -> tuple[dict[str, float | str], pd.DataFrame, pd.DataFrame]:
    """Evaluate one language split and return metrics, per-label metrics, and predictions."""

    dataset: Dataset = load_language_split(language=language, split=split, config=config)
    tokenized_dataset: Dataset = tokenize_dataset(dataset=dataset, tokenizer=tokenizer, config=config)
    prediction_output: Any = trainer.predict(test_dataset=tokenized_dataset)
    probabilities: np.ndarray = sigmoid(logits=prediction_output.predictions)
    labels: np.ndarray = np.asarray(prediction_output.label_ids)
    metrics: dict[str, float] = metric_dict_from_arrays(
        labels=labels,
        probabilities=probabilities,
        threshold=config.threshold,
    )
    metric_row: dict[str, float | str] = {
        "language": language,
        "language_name": language_display_name(language=language),
        "split": split,
        "training_status": "zero-shot" if language in ZERO_SHOT_LANGUAGES else "trained",
        **metrics,
    }
    per_label_rows: list[dict[str, float | str]] = [
        {
            "language": language,
            "language_name": language_display_name(language=language),
            "split": split,
            "training_status": "zero-shot" if language in ZERO_SHOT_LANGUAGES else "trained",
            "label": label,
            "f1": metrics[f"f1_{label}"],
        }
        for label in LABEL_COLUMNS
    ]
    per_label_frame: pd.DataFrame = pd.DataFrame(per_label_rows)
    prediction_frame: pd.DataFrame = build_prediction_frame(
        dataset=dataset,
        probabilities=probabilities,
        threshold=config.threshold,
    )
    prediction_frame["language_name"] = language_display_name(language=language)
    prediction_frame["split"] = split
    prediction_frame["training_status"] = "zero-shot" if language in ZERO_SHOT_LANGUAGES else "trained"
    return metric_row, per_label_frame, prediction_frame


def evaluate_languages(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    languages: Iterable[str],
    split: str,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate several languages and save metric tables."""

    metric_rows: list[dict[str, float | str]] = []
    per_label_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for language in languages:
        metric_row, per_label_frame, prediction_frame = evaluate_one_language(
            trainer=trainer,
            tokenizer=tokenizer,
            language=language,
            split=split,
            config=config,
        )
        metric_rows.append(metric_row)
        per_label_frames.append(per_label_frame)
        prediction_frames.append(prediction_frame)
    metrics_frame: pd.DataFrame = pd.DataFrame(metric_rows)
    per_label_metrics_frame: pd.DataFrame = pd.concat(per_label_frames, ignore_index=True)
    predictions_frame: pd.DataFrame = pd.concat(prediction_frames, ignore_index=True)
    metrics_frame.to_csv(config.tables_dir / f"{split}_language_metrics.csv", index=False)
    per_label_metrics_frame.to_csv(config.tables_dir / f"{split}_per_label_metrics.csv", index=False)
    predictions_frame.to_csv(config.tables_dir / f"{split}_predictions.csv", index=False)
    return metrics_frame, per_label_metrics_frame, predictions_frame


def plot_macro_f1_by_language(metrics_frame: pd.DataFrame, config: RunConfig) -> Path:
    """Plot macro-F1 by language."""

    output_path: Path = config.figures_dir / "macro_f1_by_language.png"
    ordered_frame: pd.DataFrame = metrics_frame.sort_values("macro_f1", ascending=False)
    plt.figure(figsize=(10, 5))
    axis: plt.Axes = sns.barplot(
        data=ordered_frame,
        x="language_name",
        y="macro_f1",
        hue="training_status",
    )
    axis.set_xlabel("Language")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Final Dev Macro-F1 By Language")
    axis.set_ylim(0.0, 1.0)
    axis.legend(title="Training Status")
    axis.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_per_label_f1_by_language(per_label_metrics_frame: pd.DataFrame, config: RunConfig) -> Path:
    """Plot per-label F1 by language."""

    output_path: Path = config.figures_dir / "per_label_f1_by_language.png"
    plt.figure(figsize=(12, 6))
    axis: plt.Axes = sns.barplot(
        data=per_label_metrics_frame,
        x="label",
        y="f1",
        hue="language_name",
    )
    axis.set_xlabel("Emotion Label")
    axis.set_ylabel("F1")
    axis.set_title("Final Dev Per-Label F1 By Language")
    axis.set_ylim(0.0, 1.0)
    axis.legend(title="Language", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def select_lime_examples(predictions_frame: pd.DataFrame, max_examples: int = 6) -> pd.DataFrame:
    """Select a small set of errors for LIME analysis."""

    error_frame: pd.DataFrame = predictions_frame[predictions_frame["has_error"]].copy()
    selected_frame: pd.DataFrame = (
        error_frame.sort_values(["training_status", "language", "id"]).groupby("language").head(1).head(max_examples)
    )
    return selected_frame


def build_lime_classifier(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: RunConfig,
) -> Callable[[list[str]], np.ndarray]:
    """Build a LIME classifier function from the trained transformer."""

    device: torch.device = next(model.parameters()).device
    model.eval()

    def classify_texts(texts: list[str]) -> np.ndarray:
        batch_size: int = 32
        all_probs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch: list[str] = texts[start : start + batch_size]
            encoded: BatchEncoding = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            encoded_on_device: dict[str, torch.Tensor] = {
                key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)
            }
            with torch.no_grad():
                outputs: Any = model(**encoded_on_device)
            all_probs.append(torch.sigmoid(outputs.logits).detach().cpu().numpy())
        return np.concatenate(all_probs, axis=0)

    return classify_texts


def _save_lime_png(
    explanation: Any,
    label_index: int,
    text: str,
    language: str,
    true_labels: list[str],
    predicted_labels: list[str],
    output_path: Path,
) -> None:
    label_name: str = LABEL_COLUMNS[label_index]
    word_weights: list[tuple[str, float]] = sorted(
        explanation.as_list(label=label_index), key=lambda x: x[1]
    )
    if not word_weights:
        return
    words: list[str] = [w for w, _ in word_weights]
    weights: list[float] = [float(v) for _, v in word_weights]
    colors: list[str] = ["#e76f51" if v < 0 else "#2a9d8f" for v in weights]
    fig_height: float = max(3.0, len(words) * 0.45 + 1.8)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(range(len(words)), weights, color=colors)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"LIME weight for '{label_name}'")
    true_str: str = ", ".join(true_labels) if true_labels else "none"
    pred_str: str = ", ".join(predicted_labels) if predicted_labels else "none"
    short_text: str = text[:70] + "..." if len(text) > 70 else text
    ax.set_title(f"[{language}] {short_text}\nTrue: {true_str}  |  Predicted: {pred_str}", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _write_lime_summary(summary_rows: list[dict], output_path: Path, model_slug: str) -> None:
    lines: list[str] = [
        f"# LIME Summary: {model_slug}",
        "",
        "| # | Lang | True | Predicted | Explained | Top positive | Top negative | Text |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['index']} | {row['language']} | {row['true_labels']} "
            f"| {row['predicted_labels']} | {row['explained_label']} "
            f"| {row['top_positive']} | {row['top_negative']} | {row['text']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def explain_lime_examples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples_frame: pd.DataFrame,
    config: RunConfig,
    num_features: int = 10,
) -> list[Path]:
    """Save LIME explanations for selected examples. Also writes PNGs and a summary markdown."""

    classifier: Callable[[list[str]], np.ndarray] = build_lime_classifier(
        model=model, tokenizer=tokenizer, config=config
    )
    explainer: LimeTextExplainer = LimeTextExplainer(class_names=list(LABEL_COLUMNS))
    output_paths: list[Path] = []
    summary_rows: list[dict] = []
    for row_index, row in examples_frame.reset_index(drop=True).iterrows():
        text: str = str(row["text"])
        language: str = str(row["language"])
        probabilities: np.ndarray = classifier([text])
        # If model predicted nothing confidently, explain the first true label instead.
        if float(probabilities[0].max()) >= 0.5:
            label_index: int = int(probabilities[0].argmax())
        else:
            true_indices: list[int] = [
                i for i, lbl in enumerate(LABEL_COLUMNS) if row.get(f"true_{lbl}", 0) == 1
            ]
            label_index = true_indices[0] if true_indices else int(probabilities[0].argmax())

        true_labels: list[str] = [lbl for lbl in LABEL_COLUMNS if row.get(f"true_{lbl}", 0) == 1]
        predicted_labels: list[str] = [
            lbl for i, lbl in enumerate(LABEL_COLUMNS) if float(probabilities[0][i]) >= 0.5
        ]

        explanation: Any = explainer.explain_instance(
            text,
            classifier,
            labels=[label_index],
            num_features=num_features,
        )

        stem: str = f"lime_{row_index:02d}_{language}_{LABEL_COLUMNS[label_index]}"
        html_path: Path = config.lime_dir / f"{stem}.html"
        explanation.save_to_file(str(html_path))
        output_paths.append(html_path)

        _save_lime_png(
            explanation=explanation,
            label_index=label_index,
            text=text,
            language=language,
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            output_path=config.lime_dir / f"{stem}.png",
        )

        word_weights: list[tuple[str, float]] = sorted(
            explanation.as_list(label=label_index), key=lambda x: abs(x[1]), reverse=True
        )
        top_pos: list[str] = [f"{w} (+{v:.3f})" for w, v in word_weights if v > 0][:3]
        top_neg: list[str] = [f"{w} ({v:.3f})" for w, v in word_weights if v < 0][:3]
        summary_rows.append(
            {
                "index": row_index,
                "language": language,
                "true_labels": ", ".join(true_labels) or "none",
                "predicted_labels": ", ".join(predicted_labels) or "none",
                "explained_label": LABEL_COLUMNS[label_index],
                "top_positive": "; ".join(top_pos),
                "top_negative": "; ".join(top_neg),
                "text": text[:120].replace("|", "/"),
            }
        )

    if summary_rows:
        _write_lime_summary(
            summary_rows=summary_rows,
            output_path=config.lime_dir / "lime_summary.md",
            model_slug=MODEL_NAME.split("/")[-1],
        )
    return output_paths


def write_markdown_summary(
    metrics_frame: pd.DataFrame,
    per_label_metrics_frame: pd.DataFrame,
    lime_paths: list[Path],
    config: RunConfig,
) -> Path:
    """Write concise result bullets after final evaluation."""

    output_path: Path = config.reports_dir / "results_summary.md"
    trained_frame: pd.DataFrame = metrics_frame[metrics_frame["training_status"] == "trained"]
    zero_shot_frame: pd.DataFrame = metrics_frame[metrics_frame["training_status"] == "zero-shot"]
    hardest_labels_frame: pd.DataFrame = per_label_metrics_frame.groupby("label", as_index=False)["f1"].mean()
    hardest_labels_frame = hardest_labels_frame.sort_values("f1", ascending=True)
    best_row: pd.Series = metrics_frame.sort_values("macro_f1", ascending=False).iloc[0]
    worst_row: pd.Series = metrics_frame.sort_values("macro_f1", ascending=True).iloc[0]
    best_language: str = str(best_row["language_name"])
    best_macro_f1: float = float(best_row["macro_f1"])
    worst_language: str = str(worst_row["language_name"])
    worst_macro_f1: float = float(worst_row["macro_f1"])
    trained_macro_f1: float = float(trained_frame["macro_f1"].mean())
    zero_shot_macro_f1: float = float(zero_shot_frame["macro_f1"].mean())
    hardest_label: str = str(hardest_labels_frame.iloc[0]["label"])
    hardest_label_f1: float = float(hardest_labels_frame.iloc[0]["f1"])
    lines: list[str] = [
        "# Results Summary",
        "",
        f"- Best final-dev language: {best_language} with macro-F1 {best_macro_f1:.3f}.",
        f"- Hardest final-dev language: {worst_language} with macro-F1 {worst_macro_f1:.3f}.",
        f"- Mean trained-language macro-F1: {trained_macro_f1:.3f}.",
        f"- Mean zero-shot macro-F1: {zero_shot_macro_f1:.3f}.",
        f"- Hardest average label: {hardest_label} with F1 {hardest_label_f1:.3f}.",
        f"- LIME explanations written: {len(lime_paths)}.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
