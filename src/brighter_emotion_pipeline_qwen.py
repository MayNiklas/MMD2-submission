"""BRIGHTER multilingual emotion classification pipeline for decoder SFT."""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import os

# Force the process to use the first CUDA GPU by default when running on an
# NVIDIA machine. Users can override by setting `CUDA_VISIBLE_DEVICES` in the
# environment before launching Python (e.g. `CUDA_VISIBLE_DEVICES=1 python ...`).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from lime.lime_text import LimeTextExplainer
from packaging import version
from sklearn.metrics import f1_score
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

SEED: int = 42069
MODEL_NAME: str = "Qwen/Qwen3.5-0.8B"
DATASET_NAME: str = "brighter-dataset/BRIGHTER-emotion-categories"
TRAIN_LANGUAGES: tuple[str, ...] = ("eng", "swa", "ukr")
EVAL_LANGUAGES: tuple[str, ...] = ("eng", "swa", "ukr", "deu", "ptbr", "yor")
ZERO_SHOT_LANGUAGES: tuple[str, ...] = ("deu", "ptbr", "yor")
LABEL_COLUMNS: tuple[str, ...] = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
LANGUAGE_NAMES: dict[str, str] = {
    "chn": "Chinese",
    "deu": "German",
    "eng": "English",
    "ptbr": "Brazilian Portuguese",
    "swa": "Swahili",
    "ukr": "Ukrainian",
    "yor": "Yoruba",
}
SYSTEM_PROMPT: str = (
    "You are an expert BRIGHTER emotion annotation model. "
    "Classify the text with any of these emotions: anger, disgust, fear, joy, sadness, surprise. "
    "Return only valid JSON with this schema: {\"emotions\": [\"label\"]}. "
    "Use an empty list when none of the listed emotions are present."
)
# Prompt variant with per-emotion cues for rare labels (fear, disgust). Tested but within noise;
# kept for reference. Applied at both train and inference time.
RICH_SYSTEM_PROMPT: str = (
    "You are an expert BRIGHTER emotion annotation model. "
    "Classify the text with any of these emotions: anger, disgust, fear, joy, sadness, surprise. "
    "A text may express several of these emotions at once, or none. Apply every label whose emotion "
    "is present, including when it is only implied or subtle  -  do not stop at the single most obvious "
    "feeling. Cues for the emotions that are easy to miss:\n"
    "- fear: worry, anxiety, dread, threat, or being scared, even when only hinted at.\n"
    "- disgust: revulsion, contempt, distaste, or finding something repellent or morally offensive.\n"
    "- anger: irritation, frustration, outrage, or blame.\n"
    "- surprise: shock, astonishment, or reacting to the unexpected.\n"
    "Do not default to joy or sadness when a subtler emotion fits better. "
    "Return only valid JSON with this schema: {\"emotions\": [\"label\"]}. "
    "Use an empty list when none of the listed emotions are present."
)
USER_PROMPT_TEMPLATE: str = "Language: {language_name}\nText: {text}\nEmotions JSON:"


def system_prompt_for(config: "RunConfig") -> str:
    """Return the system prompt for a run (richer rare-emotion guidance when enabled)."""

    return RICH_SYSTEM_PROMPT if config.rich_system_prompt else SYSTEM_PROMPT


@dataclass(frozen=True)
class RunConfig:
    """Configuration for the decoder SFT experiment."""

    seed: int = SEED
    model_name: str = MODEL_NAME
    dataset_name: str = DATASET_NAME
    run_suffix: str = ""
    output_dir: Path = Path("outputs")
    reports_dir: Path = Path("reports")
    max_length: int = 384
    max_new_tokens: int = 48
    internal_dev_size: float = 0.15
    max_steps: int = -1
    num_train_epochs: int = 2
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    weight_decay: float = 0.01
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True
    select_best_model: bool = True
    rich_system_prompt: bool = False
    oversample_rare_emotions: bool = False
    oversample_target_rate: float = 0.15
    oversample_max_factor: int = 5
    lime_num_samples: int = 500
    max_eval_examples_per_language: int = 0
    smoke_train_examples_per_language: int = 2
    smoke_internal_dev_examples_per_language: int = 1
    isolate_recompute: bool = True

    @property
    def model_slug(self) -> str:
        model_name_tail: str = self.model_name.split("/")[-1]
        slug: str = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name_tail)
        if slug.startswith("Qwen"):
            slug = f"q{slug[1:]}"
        return slug

    @property
    def artifact_prefix(self) -> str:
        return f"{self.artifact_stem}_"

    @property
    def artifact_stem(self) -> str:
        clean_suffix: str = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.run_suffix).strip("_")
        artifact_stem: str = f"{self.model_slug}_{clean_suffix}" if clean_suffix else self.model_slug
        return artifact_stem

    @property
    def model_dir(self) -> Path:
        return self.output_dir / f"{self.artifact_stem}_model"

    @property
    def logging_dir(self) -> Path:
        return self.output_dir / f"{self.artifact_stem}_logs"

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
    """Set random seeds for reproducible runs."""

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
    """Return a human-readable language name."""

    return LANGUAGE_NAMES[language]


def load_language_split(language: str, split: str, config: RunConfig) -> Dataset:
    """Load one BRIGHTER language split and attach the language code."""

    dataset: Dataset = load_dataset(config.dataset_name, language, split=split)
    language_values: list[str] = [language] * len(dataset)
    dataset = dataset.add_column("language", language_values)
    return dataset


def load_language_split_sample(language: str, split: str, row_count: int, config: RunConfig) -> Dataset:
    """Seeded random sample from one BRIGHTER split. Shuffle before slicing to keep rare emotions in sample."""

    dataset: Dataset = load_language_split(language=language, split=split, config=config)
    shuffled_dataset: Dataset = dataset.shuffle(seed=config.seed)
    sample_size: int = min(row_count, len(shuffled_dataset))
    sampled_dataset: Dataset = shuffled_dataset.select(range(sample_size))
    return sampled_dataset


def load_combined_split(languages: Iterable[str], split: str, config: RunConfig) -> Dataset:
    """Load and concatenate one split for several languages."""

    datasets: list[Dataset] = [
        load_language_split(language=language, split=split, config=config) for language in languages
    ]
    combined_dataset: Dataset = concatenate_datasets(datasets)
    return combined_dataset


def load_combined_eval_split(languages: Iterable[str], split: str, config: RunConfig) -> Dataset:
    """Load one split across several languages, capped per language when max_eval_examples_per_language > 0."""

    cap: int = config.max_eval_examples_per_language
    if cap <= 0:
        return load_combined_split(languages=languages, split=split, config=config)
    datasets: list[Dataset] = [
        load_language_split_sample(language=language, split=split, row_count=cap, config=config)
        for language in languages
    ]
    combined_dataset: Dataset = concatenate_datasets(datasets)
    return combined_dataset


def build_oversampled_train_dataset(train_dataset: Dataset, config: RunConfig) -> Dataset:
    """Duplicate rare-emotion rows to push their positive rate toward oversample_target_rate.

    For each (language, label) pair below the target, matching rows are repeated by
    round(target / actual), capped at oversample_max_factor. Internal-dev is never touched.
    """

    dataframe: pd.DataFrame = train_dataset.to_pandas()
    positive_rate: dict[tuple[str, str], float] = {}
    for language, language_frame in dataframe.groupby("language", sort=False):
        row_count: int = len(language_frame)
        for label in LABEL_COLUMNS:
            positive_count: int = int(sum(label in emotions for emotions in language_frame["emotions"]))
            positive_rate[(str(language), label)] = positive_count / row_count if row_count else 0.0

    repeated_positions: list[int] = []
    for position, (_index, row) in enumerate(dataframe.iterrows()):
        language = str(row["language"])
        replication: int = 1
        for label in LABEL_COLUMNS:
            if label not in row["emotions"]:
                continue
            rate: float = positive_rate[(language, label)]
            if 0.0 < rate < config.oversample_target_rate:
                factor: int = min(config.oversample_max_factor, max(1, round(config.oversample_target_rate / rate)))
                replication = max(replication, factor)
        repeated_positions.extend([position] * replication)

    oversampled_frame: pd.DataFrame = dataframe.iloc[repeated_positions].reset_index(drop=True)
    oversampled_dataset: Dataset = Dataset.from_pandas(oversampled_frame, preserve_index=False)
    return oversampled_dataset.shuffle(seed=config.seed)


def create_train_internal_dev_split(config: RunConfig) -> DatasetDict:
    """Split official train into train/internal-dev. Optionally oversamples rare emotions in train only."""

    official_train_dataset: Dataset = load_combined_split(languages=TRAIN_LANGUAGES, split="train", config=config)
    split_dataset: DatasetDict = official_train_dataset.train_test_split(
        test_size=config.internal_dev_size,
        seed=config.seed,
        shuffle=True,
    )
    train_dataset: Dataset = split_dataset["train"]
    if config.oversample_rare_emotions:
        train_dataset = build_oversampled_train_dataset(train_dataset=train_dataset, config=config)
    renamed_dataset: DatasetDict = DatasetDict(
        {
            "train": train_dataset,
            "internal_dev": split_dataset["test"],
        }
    )
    return renamed_dataset


def create_smoke_train_internal_dev_split(config: RunConfig) -> DatasetDict:
    """Create a tiny train/internal-dev split for backend smoke tests."""

    train_datasets: list[Dataset] = []
    internal_dev_datasets: list[Dataset] = []
    row_count: int = config.smoke_train_examples_per_language + config.smoke_internal_dev_examples_per_language
    for language in TRAIN_LANGUAGES:
        language_dataset: Dataset = load_language_split_sample(
            language=language,
            split="train",
            row_count=row_count,
            config=config,
        )
        train_end_index: int = min(config.smoke_train_examples_per_language, len(language_dataset))
        internal_dev_end_index: int = min(
            train_end_index + config.smoke_internal_dev_examples_per_language,
            len(language_dataset),
        )
        train_datasets.append(language_dataset.select(range(train_end_index)))
        internal_dev_datasets.append(language_dataset.select(range(train_end_index, internal_dev_end_index)))
    smoke_splits: DatasetDict = DatasetDict(
        {
            "train": concatenate_datasets(train_datasets),
            "internal_dev": concatenate_datasets(internal_dev_datasets),
        }
    )
    return smoke_splits


def build_label_matrix(batch: dict[str, list[Any]]) -> list[list[int]]:
    """Build a multi-label target matrix from BRIGHTER emotion lists."""

    labels: list[list[int]] = []
    row_count: int = len(batch["text"])
    for row_index in range(row_count):
        row_emotions: set[str] = set(batch["emotions"][row_index])
        row_labels: list[int] = [1 if label in row_emotions else 0 for label in LABEL_COLUMNS]
        labels.append(row_labels)
    return labels


def prepare_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Prepare a tokenizer for decoder fine-tuning."""

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_messages(text: str, language: str, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Build chat messages for one BRIGHTER example."""

    language_name: str = language_display_name(language=language)
    user_prompt: str = USER_PROMPT_TEMPLATE.format(language_name=language_name, text=text)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def render_prompt(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    language: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Render the decoder prompt for one example."""

    messages: list[dict[str, str]] = build_messages(text=text, language=language, system_prompt=system_prompt)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        # Qwen3.5 4B+ opens a <think> block by default and never emits the JSON.
        # enable_thinking=False suppresses this; 0.8B ignores the kwarg.
        try:
            prompt: str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        language_name: str = language_display_name(language=language)
        user_prompt: str = USER_PROMPT_TEMPLATE.format(language_name=language_name, text=text)
        prompt = f"{system_prompt}\n\n{user_prompt}\n"
    return prompt


def format_emotion_response(emotions: Sequence[str]) -> str:
    """Format one target response as strict JSON."""

    emotion_set: set[str] = set(emotions)
    ordered_emotions: list[str] = [label for label in LABEL_COLUMNS if label in emotion_set]
    response: str = json.dumps({"emotions": ordered_emotions}, ensure_ascii=False)
    return response


def tokenize_example(
    text: str,
    language: str,
    emotions: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, list[int]]:
    """Tokenize one prompt and mask the prompt tokens in the labels."""

    prompt: str = render_prompt(tokenizer=tokenizer, text=text, language=language, system_prompt=system_prompt)
    response: str = format_emotion_response(emotions=emotions)
    eos_token: str = tokenizer.eos_token or ""
    full_text: str = f"{prompt}{response}{eos_token}"
    prompt_encoding: BatchEncoding = tokenizer(prompt, truncation=True, max_length=max_length)
    full_encoding: BatchEncoding = tokenizer(full_text, truncation=True, max_length=max_length)
    input_ids: list[int] = list(full_encoding["input_ids"])
    attention_mask: list[int] = list(full_encoding["attention_mask"])
    prompt_length: int = min(len(prompt_encoding["input_ids"]), len(input_ids))
    labels: list[int] = [-100] * prompt_length + input_ids[prompt_length:]
    if all(label == -100 for label in labels) and input_ids:
        labels[-1] = input_ids[-1]
    tokenized: dict[str, list[int]] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return tokenized


def tokenize_batch(
    batch: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, list[list[int]]]:
    """Tokenize a BRIGHTER batch for decoder supervised fine-tuning."""

    tokenized_rows: list[dict[str, list[int]]] = [
        tokenize_example(
            text=str(text),
            language=str(language),
            emotions=emotions,
            tokenizer=tokenizer,
            max_length=max_length,
            system_prompt=system_prompt,
        )
        for text, language, emotions in zip(batch["text"], batch["language"], batch["emotions"], strict=True)
    ]
    tokenized_batch: dict[str, list[list[int]]] = {
        "input_ids": [row["input_ids"] for row in tokenized_rows],
        "attention_mask": [row["attention_mask"] for row in tokenized_rows],
        "labels": [row["labels"] for row in tokenized_rows],
    }
    return tokenized_batch


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, config: RunConfig) -> Dataset:
    """Tokenize a dataset for decoder supervised fine-tuning."""

    tokenized_dataset: Dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": config.max_length,
            "system_prompt": system_prompt_for(config),
        },
    )
    return tokenized_dataset


def patch_transformers_runtime() -> None:
    """Patch optional Transformers runtime probes that can break text-only training environments."""

    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return
    torch_version: str = str(torch.__version__).split("+")[0]

    def is_actual_torch_greater_or_equal(library_version: str, accept_dev: bool = False) -> bool:
        """Compare against the imported torch runtime instead of stale package metadata."""

        parsed_torch_version: version.Version = version.parse(torch_version)
        if accept_dev:
            parsed_torch_version = version.parse(parsed_torch_version.base_version)
        return parsed_torch_version >= version.parse(library_version)

    if getattr(import_utils, "_torch_version", torch_version) != torch_version:
        import_utils._torch_version = torch_version
    import_utils.is_torch_greater_or_equal = is_actual_torch_greater_or_equal
    transformers_utils.is_torch_greater_or_equal = is_actual_torch_greater_or_equal

    def report_torchvision_unavailable(*args: Any, **kwargs: Any) -> bool:
        """Prevents a crash from a broken torch/torchvision build on this text-only task."""

        return False

    import_utils.is_torchvision_available = report_torchvision_unavailable
    transformers_utils.is_torchvision_available = report_torchvision_unavailable
    if hasattr(import_utils, "_torchvision_available"):
        import_utils._torchvision_available = False
    image_utils_module = sys.modules.get("transformers.image_utils")
    if image_utils_module is not None:
        image_utils_module.is_torchvision_available = report_torchvision_unavailable


def load_tokenizer(config: RunConfig) -> PreTrainedTokenizerBase:
    """Load the configured text tokenizer."""

    patch_transformers_runtime()
    from transformers import AutoTokenizer

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer = prepare_tokenizer(tokenizer=tokenizer)
    return tokenizer


def load_model(config: RunConfig) -> Any:
    """Load the configured Qwen decoder model for supervised fine-tuning."""

    patch_transformers_runtime()
    import transformers

    model_class: Any = getattr(transformers, "AutoModelForCausalLM", None)
    if model_class is None:
        raise ImportError(
            "Qwen3.5 requires a recent Transformers build with AutoModelForCausalLM support. "
            "Install the latest Transformers version recommended on the Qwen model card."
        )
    # MPS requires eager attention: sdpa crashes on grouped-query attention (mismatched head counts).
    # On CUDA, sdpa is correct and faster (FlashAttention / memory-efficient kernels).
    is_mps_backend: bool = torch.backends.mps.is_available()
    attn_implementation: str = "eager" if is_mps_backend else "sdpa"
    model: Any = model_class.from_pretrained(
        config.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model


@dataclass
class CausalLMDataCollator:
    """Pad causal LM batches while preserving masked labels."""

    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate variable-length tokenized examples."""

        max_length: int = max(len(feature["input_ids"]) for feature in features)
        pad_token_id: int = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for feature in features:
            input_ids: list[int] = list(feature["input_ids"])
            attention_mask: list[int] = list(feature["attention_mask"])
            labels: list[int] = list(feature["labels"])
            pad_length: int = max_length - len(input_ids)
            batch_input_ids.append(input_ids + [pad_token_id] * pad_length)
            batch_attention_mask.append(attention_mask + [0] * pad_length)
            batch_labels.append(labels + [self.label_pad_token_id] * pad_length)
        batch: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
        return batch


def create_trainer(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    tokenized_train_dataset: Dataset,
    tokenized_internal_dev_dataset: Dataset,
    config: RunConfig,
) -> Any:
    """Create a Hugging Face Trainer for decoder supervised fine-tuning."""

    patch_transformers_runtime()
    from transformers import Trainer, TrainingArguments

    text_tokenizer: PreTrainedTokenizerBase = prepare_tokenizer(tokenizer=tokenizer)
    is_smoke_run: bool = config.max_steps > 0
    # Smoke runs skip eval and checkpointing entirely. Real runs save one checkpoint per epoch
    # and pick the best by eval_loss. Sweep runs set select_best_model=False to track macro-F1
    # through a callback instead and skip checkpointing.
    select_best_model: bool = config.select_best_model and not is_smoke_run
    training_arguments: TrainingArguments = TrainingArguments(
        output_dir=str(config.output_dir),
        logging_dir=str(config.logging_dir),
        eval_strategy="no" if is_smoke_run else "epoch",
        save_strategy="epoch" if select_best_model else "no",
        save_total_limit=1,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        weight_decay=config.weight_decay,
        optim=config.optim,
        load_best_model_at_end=select_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
        dataloader_num_workers=0,
        logging_steps=1 if is_smoke_run else 10,
        remove_unused_columns=False,
    )
    trainer: Trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_internal_dev_dataset,
        processing_class=text_tokenizer,
        data_collator=CausalLMDataCollator(tokenizer=text_tokenizer),
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
    overview.to_csv(config.tables_dir / f"{config.artifact_prefix}training_data_overview.csv", index=False)
    return overview


def parse_emotion_response(response_text: str) -> list[str]:
    """Parse generated JSON and return valid emotion labels."""

    cleaned_response: str = response_text.strip()
    match: re.Match[str] | None = re.search(r"\{.*?\}", cleaned_response, flags=re.DOTALL)
    json_text: str = match.group(0) if match else cleaned_response
    try:
        parsed_response: Any = json.loads(json_text)
    except json.JSONDecodeError:
        parsed_response = {}
    raw_emotions: Any = parsed_response.get("emotions", []) if isinstance(parsed_response, dict) else []
    if isinstance(raw_emotions, str):
        raw_emotions = [raw_emotions]
    emotions: list[str] = []
    if isinstance(raw_emotions, list):
        for raw_emotion in raw_emotions:
            emotion: str = str(raw_emotion).strip().lower()
            if emotion in LABEL_COLUMNS and emotion not in emotions:
                emotions.append(emotion)
    return emotions


def labels_from_emotions(emotions: Sequence[str]) -> list[int]:
    """Convert emotion labels into a fixed-order binary vector."""

    emotion_set: set[str] = set(emotions)
    labels: list[int] = [1 if label in emotion_set else 0 for label in LABEL_COLUMNS]
    return labels


def metric_dict_from_arrays(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Compute macro, micro, sample, and per-label F1 metrics."""

    metrics: dict[str, float] = {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(labels, predictions, average="samples", zero_division=0)),
    }
    per_label_f1: np.ndarray = f1_score(labels, predictions, average=None, zero_division=0)
    for label, label_f1 in zip(LABEL_COLUMNS, per_label_f1, strict=True):
        metrics[f"f1_{label}"] = float(label_f1)
    return metrics


def generate_responses(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    languages: Sequence[str],
    config: RunConfig,
) -> list[str]:
    """Generate decoder responses for a batch of texts."""

    system_prompt: str = system_prompt_for(config)
    prompts: list[str] = [
        render_prompt(tokenizer=tokenizer, text=text, language=language, system_prompt=system_prompt)
        for text, language in zip(texts, languages, strict=True)
    ]
    device: torch.device = next(model.parameters()).device
    encoded: BatchEncoding = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    encoded_on_device: dict[str, torch.Tensor] = {
        key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)
    }
    model.eval()
    with torch.no_grad():
        generated_ids: torch.Tensor = model.generate(
            **encoded_on_device,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_width: int = int(encoded_on_device["input_ids"].shape[1])
    response_ids: torch.Tensor = generated_ids[:, prompt_width:]
    responses: list[str] = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    return responses


def predict_dataset(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    config: RunConfig,
) -> tuple[np.ndarray, list[str], list[list[str]]]:
    """Generate and parse predictions for a BRIGHTER dataset."""

    dataframe: pd.DataFrame = dataset.to_pandas()
    batch_size: int = max(1, config.per_device_eval_batch_size)
    generated_responses: list[str] = []
    predicted_emotions: list[list[str]] = []
    for start_index in range(0, len(dataframe), batch_size):
        end_index: int = min(start_index + batch_size, len(dataframe))
        batch_frame: pd.DataFrame = dataframe.iloc[start_index:end_index]
        batch_texts: list[str] = [str(text) for text in batch_frame["text"].tolist()]
        batch_languages: list[str] = [str(language) for language in batch_frame["language"].tolist()]
        batch_responses: list[str] = generate_responses(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            languages=batch_languages,
            config=config,
        )
        generated_responses.extend(batch_responses)
        predicted_emotions.extend(parse_emotion_response(response_text=response) for response in batch_responses)
    predictions: np.ndarray = np.asarray(
        [labels_from_emotions(emotions=emotions) for emotions in predicted_emotions], dtype=int
    )
    return predictions, generated_responses, predicted_emotions


def build_prediction_frame(
    dataset: Dataset,
    predictions: np.ndarray,
    generated_responses: Sequence[str],
    predicted_emotions: Sequence[Sequence[str]],
) -> pd.DataFrame:
    """Build a per-example prediction table."""

    dataframe: pd.DataFrame = dataset.to_pandas()
    for label_index, label in enumerate(LABEL_COLUMNS):
        true_values: list[int] = [int(label in emotions) for emotions in dataframe["emotions"]]
        dataframe[f"true_{label}"] = true_values
        dataframe[f"pred_{label}"] = predictions[:, label_index].astype(int)
    true_columns: list[str] = [f"true_{label}" for label in LABEL_COLUMNS]
    pred_columns: list[str] = [f"pred_{label}" for label in LABEL_COLUMNS]
    dataframe["predicted_emotions"] = [
        json.dumps(list(emotions), ensure_ascii=False) for emotions in predicted_emotions
    ]
    dataframe["generated_response"] = list(generated_responses)
    dataframe["has_error"] = (dataframe[true_columns].to_numpy() != dataframe[pred_columns].to_numpy()).any(axis=1)
    return dataframe


def evaluate_dataset(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    split: str,
    config: RunConfig,
    training_status_by_language: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a prepared dataset and save metric tables."""

    dataframe: pd.DataFrame = dataset.to_pandas()
    metric_rows: list[dict[str, float | str]] = []
    per_label_rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    status_lookup: dict[str, str] = training_status_by_language or {}
    for language, language_frame in dataframe.groupby("language", sort=False):
        language_dataset: Dataset = Dataset.from_pandas(language_frame.reset_index(drop=True), preserve_index=False)
        predictions, generated_responses, predicted_emotions = predict_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=language_dataset,
            config=config,
        )
        labels: np.ndarray = np.asarray(
            [labels_from_emotions(emotions=emotions) for emotions in language_frame["emotions"]], dtype=int
        )
        metrics: dict[str, float] = metric_dict_from_arrays(labels=labels, predictions=predictions)
        training_status: str = status_lookup.get(str(language), "trained")
        metric_rows.append(
            {
                "language": str(language),
                "language_name": language_display_name(language=str(language)),
                "split": split,
                "training_status": training_status,
                **metrics,
            }
        )
        for label in LABEL_COLUMNS:
            per_label_rows.append(
                {
                    "language": str(language),
                    "language_name": language_display_name(language=str(language)),
                    "split": split,
                    "training_status": training_status,
                    "label": label,
                    "f1": metrics[f"f1_{label}"],
                }
            )
        prediction_frame: pd.DataFrame = build_prediction_frame(
            dataset=language_dataset,
            predictions=predictions,
            generated_responses=generated_responses,
            predicted_emotions=predicted_emotions,
        )
        prediction_frame["language_name"] = language_display_name(language=str(language))
        prediction_frame["split"] = split
        prediction_frame["training_status"] = training_status
        prediction_frames.append(prediction_frame)
    metrics_frame: pd.DataFrame = pd.DataFrame(metric_rows)
    per_label_metrics_frame: pd.DataFrame = pd.DataFrame(per_label_rows)
    predictions_frame: pd.DataFrame = pd.concat(prediction_frames, ignore_index=True)
    metrics_frame.to_csv(config.tables_dir / f"{config.artifact_prefix}{split}_language_metrics.csv", index=False)
    per_label_metrics_frame.to_csv(
        config.tables_dir / f"{config.artifact_prefix}{split}_per_label_metrics.csv", index=False
    )
    predictions_frame.to_csv(config.tables_dir / f"{config.artifact_prefix}{split}_predictions.csv", index=False)
    return metrics_frame, per_label_metrics_frame, predictions_frame


def macro_f1_by_language(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    config: RunConfig,
) -> pd.DataFrame:
    """Score macro-F1 per language without writing any artifacts. Used by the training callback and scaling studies."""

    dataframe: pd.DataFrame = dataset.to_pandas()
    rows: list[dict[str, float | str]] = []
    for language, language_frame in dataframe.groupby("language", sort=False):
        language_dataset: Dataset = Dataset.from_pandas(language_frame.reset_index(drop=True), preserve_index=False)
        predictions, _generated, _emotions = predict_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=language_dataset,
            config=config,
        )
        labels: np.ndarray = np.asarray(
            [labels_from_emotions(emotions=emotions) for emotions in language_frame["emotions"]], dtype=int
        )
        metrics: dict[str, float] = metric_dict_from_arrays(labels=labels, predictions=predictions)
        row: dict[str, float | str] = {
            "language": str(language),
            "language_name": language_display_name(language=str(language)),
            "macro_f1": metrics["macro_f1"],
        }
        for label in LABEL_COLUMNS:
            row[f"f1_{label}"] = metrics[f"f1_{label}"]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_languages(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    languages: Iterable[str],
    split: str,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate several language splits and save metric tables."""

    dataset: Dataset = load_combined_eval_split(languages=languages, split=split, config=config)
    training_status_by_language: dict[str, str] = {
        language: "zero-shot" if language in ZERO_SHOT_LANGUAGES else "trained" for language in languages
    }
    return evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        split=split,
        config=config,
        training_status_by_language=training_status_by_language,
    )


def evaluate_base_model_languages(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    languages: Iterable[str],
    split: str,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the untrained base model on the same dev sample as the fine-tuned run. Results written under {split}_base prefix."""

    dataset: Dataset = load_combined_eval_split(languages=languages, split=split, config=config)
    training_status_by_language: dict[str, str] = {language: "base" for language in languages}
    return evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        split=f"{split}_base",
        config=config,
        training_status_by_language=training_status_by_language,
    )


def compare_base_and_finetuned(
    finetuned_metrics_frame: pd.DataFrame,
    base_metrics_frame: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    """Per-language base vs fine-tuned macro-F1 table. macro_f1_delta = fine-tuned - base."""

    finetuned_columns: pd.DataFrame = finetuned_metrics_frame[
        ["language", "language_name", "training_status", "macro_f1"]
    ]
    base_columns: pd.DataFrame = base_metrics_frame[["language", "macro_f1"]]
    comparison_frame: pd.DataFrame = finetuned_columns.merge(
        base_columns, on="language", suffixes=("_finetuned", "_base")
    )
    comparison_frame["macro_f1_delta"] = comparison_frame["macro_f1_finetuned"] - comparison_frame["macro_f1_base"]
    comparison_frame.to_csv(config.tables_dir / f"{config.artifact_prefix}base_vs_finetuned_macro_f1.csv", index=False)
    return comparison_frame


def plot_macro_f1_by_language(metrics_frame: pd.DataFrame, config: RunConfig) -> Path:
    """Plot macro-F1 by language."""

    output_path: Path = config.figures_dir / f"{config.artifact_prefix}macro_f1_by_language.png"
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
    axis.set_title(f"Final Dev Macro-F1 By Language: {config.model_slug}")
    axis.set_ylim(0.0, 1.0)
    axis.legend(title="Training Status")
    axis.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_per_label_f1_by_language(per_label_metrics_frame: pd.DataFrame, config: RunConfig) -> Path:
    """Plot per-label F1 by language."""

    output_path: Path = config.figures_dir / f"{config.artifact_prefix}per_label_f1_by_language.png"
    plt.figure(figsize=(12, 6))
    axis: plt.Axes = sns.barplot(
        data=per_label_metrics_frame,
        x="label",
        y="f1",
        hue="language_name",
    )
    axis.set_xlabel("Emotion Label")
    axis.set_ylabel("F1")
    axis.set_title(f"Final Dev Per-Label F1 By Language: {config.model_slug}")
    axis.set_ylim(0.0, 1.0)
    axis.legend(title="Language", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def compare_base_and_finetuned_per_label(
    finetuned_per_label_frame: pd.DataFrame,
    base_per_label_frame: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    """Per-language, per-label base vs fine-tuned F1 table. f1_delta = fine-tuned - base."""

    finetuned_columns: pd.DataFrame = finetuned_per_label_frame[
        ["language", "language_name", "training_status", "label", "f1"]
    ]
    base_columns: pd.DataFrame = base_per_label_frame[["language", "label", "f1"]]
    comparison_frame: pd.DataFrame = finetuned_columns.merge(
        base_columns, on=["language", "label"], suffixes=("_finetuned", "_base")
    )
    comparison_frame["f1_delta"] = comparison_frame["f1_finetuned"] - comparison_frame["f1_base"]
    comparison_frame.to_csv(
        config.tables_dir / f"{config.artifact_prefix}base_vs_finetuned_per_label_f1.csv", index=False
    )
    return comparison_frame


def plot_macro_f1_improvement_by_language(comparison_frame: pd.DataFrame, config: RunConfig) -> Path:
    """Diverging bar chart of per-language macro-F1 delta. Green bars = improved, red = regressed."""

    output_path: Path = config.figures_dir / f"{config.artifact_prefix}macro_f1_improvement_by_language.png"
    ordered_frame: pd.DataFrame = comparison_frame.sort_values("macro_f1_delta", ascending=False)
    bar_colors: list[str] = [
        "#2a9d8f" if delta >= 0.0 else "#e76f51" for delta in ordered_frame["macro_f1_delta"]
    ]
    plt.figure(figsize=(10, 5))
    plt.bar(ordered_frame["language_name"], ordered_frame["macro_f1_delta"], color=bar_colors)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Language")
    plt.ylabel("Macro-F1 improvement (fine-tuned − base)")
    plt.title(f"Fine-tuning Macro-F1 Improvement By Language: {config.model_slug}")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_per_label_f1_improvement(per_label_comparison_frame: pd.DataFrame, config: RunConfig) -> Path:
    """Per-label F1 delta (fine-tuned - base) grouped by language."""

    output_path: Path = config.figures_dir / f"{config.artifact_prefix}per_label_f1_improvement.png"
    plt.figure(figsize=(12, 6))
    axis: plt.Axes = sns.barplot(
        data=per_label_comparison_frame,
        x="label",
        y="f1_delta",
        hue="language_name",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Emotion Label")
    axis.set_ylabel("F1 improvement (fine-tuned − base)")
    axis.set_title(f"Fine-tuning Per-Label F1 Improvement: {config.model_slug}")
    axis.legend(title="Language", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def build_macro_f1_history_callback(
    tokenizer: PreTrainedTokenizerBase,
    eval_dataset: Dataset,
    config: RunConfig,
) -> tuple[Any, list[pd.DataFrame]]:
    """TrainerCallback that scores generative macro-F1 per epoch. Returns the callback and a list that collects one frame per epoch."""

    patch_transformers_runtime()
    from transformers import TrainerCallback

    history_frames: list[pd.DataFrame] = []

    class MacroF1HistoryCallback(TrainerCallback):
        """Score and log generative macro-F1 at each epoch boundary."""

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            """Evaluate the in-progress model generatively and append the result."""

            model: Any = kwargs["model"]
            epoch: int = int(round(float(state.epoch))) if state.epoch is not None else len(history_frames) + 1
            frame: pd.DataFrame = macro_f1_by_language(
                model=model, tokenizer=tokenizer, dataset=eval_dataset, config=config
            )
            frame.insert(0, "epoch", epoch)
            frame.insert(0, "run_suffix", config.run_suffix)
            history_frames.append(frame)
            mean_macro_f1: float = float(frame["macro_f1"].mean())
            print(
                f"[macro-f1] {config.run_suffix or config.model_slug} epoch {epoch}: "
                f"mean internal-dev macro-F1 {mean_macro_f1:.4f}",
                flush=True,
            )
            # generate() flipped the model to eval(); hand it back to the trainer in train mode.
            model.train()
            return control

    return MacroF1HistoryCallback(), history_frames


def run_lr_epoch_sweep(
    base_config: RunConfig,
    learning_rates: Sequence[float],
    max_epochs: int,
    use_subsampled_split: bool = False,
    eval_sample_size: int = 0,
) -> pd.DataFrame:
    """Sweep learning rates while reading the epoch axis for free via a per-epoch macro-F1 callback.

    Each LR trains one model for max_epochs; the callback records internal-dev macro-F1 after each
    epoch, giving the full 1..max curve from a single run per LR. Official dev is never touched.
    """

    ensure_directories(config=base_config)
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(config=base_config)
    splits: DatasetDict = (
        create_smoke_train_internal_dev_split(config=base_config)
        if use_subsampled_split
        else create_train_internal_dev_split(config=base_config)
    )
    internal_dev_dataset: Dataset = splits["internal_dev"]
    if eval_sample_size and eval_sample_size > 0:
        shuffled_internal_dev: Dataset = internal_dev_dataset.shuffle(seed=base_config.seed)
        eval_dataset: Dataset = shuffled_internal_dev.select(range(min(eval_sample_size, len(shuffled_internal_dev))))
    else:
        eval_dataset = internal_dev_dataset
    print(f"Scoring macro-F1 on {len(eval_dataset)} internal-dev examples each epoch.", flush=True)

    sweep_stem: str = base_config.run_suffix or "sweep"
    all_frames: list[pd.DataFrame] = []
    for learning_rate in learning_rates:
        run_config: RunConfig = replace(
            base_config,
            learning_rate=learning_rate,
            num_train_epochs=max_epochs,
            select_best_model=False,
            run_suffix=f"{sweep_stem}_lr{learning_rate:g}",
        )
        print(f"\n=== sweep run: lr={learning_rate:g}, epochs={max_epochs} ===", flush=True)
        set_random_seed(seed=run_config.seed)
        model: Any = load_model(config=run_config)
        tokenized_train_dataset: Dataset = tokenize_dataset(
            dataset=splits["train"], tokenizer=tokenizer, config=run_config
        )
        tokenized_internal_dev_dataset: Dataset = tokenize_dataset(
            dataset=splits["internal_dev"], tokenizer=tokenizer, config=run_config
        )
        callback, history_frames = build_macro_f1_history_callback(
            tokenizer=tokenizer, eval_dataset=eval_dataset, config=run_config
        )
        trainer: Any = create_trainer(
            model=model,
            tokenizer=tokenizer,
            tokenized_train_dataset=tokenized_train_dataset,
            tokenized_internal_dev_dataset=tokenized_internal_dev_dataset,
            config=run_config,
        )
        trainer.add_callback(callback)
        trainer.train()
        for frame in history_frames:
            frame["learning_rate"] = learning_rate
        all_frames.extend(history_frames)
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_frame: pd.DataFrame = pd.concat(all_frames, ignore_index=True)
    ordered_columns: list[str] = ["learning_rate", "run_suffix", "epoch", "language", "language_name", "macro_f1"]
    ordered_columns += [f"f1_{label}" for label in LABEL_COLUMNS]
    results_frame = results_frame[ordered_columns]
    results_frame.to_csv(base_config.tables_dir / f"{base_config.artifact_prefix}lr_epoch_sweep.csv", index=False)
    return results_frame


def plot_lr_epoch_sweep(results_frame: pd.DataFrame, config: RunConfig) -> Path:
    """Plot mean internal-dev macro-F1 versus epoch, one line per learning rate."""

    output_path: Path = config.figures_dir / f"{config.artifact_prefix}lr_epoch_sweep.png"
    aggregated_frame: pd.DataFrame = results_frame.groupby(
        ["learning_rate", "epoch"], as_index=False
    )["macro_f1"].mean()
    aggregated_frame["learning_rate_label"] = aggregated_frame["learning_rate"].map(lambda value: f"{value:g}")
    plt.figure(figsize=(9, 5))
    axis: plt.Axes = sns.lineplot(
        data=aggregated_frame,
        x="epoch",
        y="macro_f1",
        hue="learning_rate_label",
        marker="o",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean internal-dev macro-F1 (across languages)")
    axis.set_title(f"Learning-rate / epoch sweep: {config.model_slug}")
    axis.set_xticks(sorted(aggregated_frame["epoch"].unique()))
    axis.legend(title="Learning rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_language_learning_curves(
    results_frame: pd.DataFrame,
    config: RunConfig,
    learning_rate: float | None = None,
) -> Path:
    """Per-language internal-dev macro-F1 vs epoch. Pass learning_rate to filter to one rate."""

    frame: pd.DataFrame = results_frame
    filename_suffix: str = ""
    title_suffix: str = ""
    if learning_rate is not None:
        frame = frame[frame["learning_rate"] == learning_rate]
        filename_suffix = f"_lr{learning_rate:g}"
        title_suffix = f" (lr={learning_rate:g})"
    aggregated_frame: pd.DataFrame = frame.groupby(
        ["language_name", "epoch"], as_index=False
    )["macro_f1"].mean()
    output_path: Path = config.figures_dir / f"{config.artifact_prefix}language_learning_curves{filename_suffix}.png"
    plt.figure(figsize=(9, 5))
    axis: plt.Axes = sns.lineplot(
        data=aggregated_frame,
        x="epoch",
        y="macro_f1",
        hue="language_name",
        marker="o",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Internal-dev macro-F1")
    axis.set_title(f"Per-language training impact{title_suffix}: {config.model_slug}")
    axis.set_xticks(sorted(aggregated_frame["epoch"].unique()))
    axis.set_ylim(0.0, 1.0)
    axis.legend(title="Language")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_emotion_learning_curves(
    results_frame: pd.DataFrame,
    config: RunConfig,
    learning_rate: float | None = None,
) -> Path:
    """Per-emotion F1 vs epoch, one panel per language. Pass learning_rate to filter to one rate."""

    frame: pd.DataFrame = results_frame
    filename_suffix: str = ""
    title_suffix: str = ""
    if learning_rate is not None:
        frame = frame[frame["learning_rate"] == learning_rate]
        filename_suffix = f"_lr{learning_rate:g}"
        title_suffix = f" (lr={learning_rate:g})"
    label_columns: list[str] = [f"f1_{label}" for label in LABEL_COLUMNS]
    long_frame: pd.DataFrame = frame.melt(
        id_vars=["language_name", "epoch"],
        value_vars=label_columns,
        var_name="emotion",
        value_name="f1",
    )
    long_frame["emotion"] = long_frame["emotion"].str[len("f1_") :]
    aggregated_frame: pd.DataFrame = long_frame.groupby(
        ["language_name", "emotion", "epoch"], as_index=False
    )["f1"].mean()
    languages: list[str] = sorted(aggregated_frame["language_name"].unique())
    epochs_sorted: list[int] = sorted(aggregated_frame["epoch"].unique())
    figure, axes = plt.subplots(
        1, len(languages), figsize=(5.5 * len(languages) + 1.5, 5.5), sharey=True, squeeze=False
    )
    for axis, language in zip(axes[0], languages, strict=True):
        panel_frame: pd.DataFrame = aggregated_frame[aggregated_frame["language_name"] == language]
        sns.lineplot(data=panel_frame, x="epoch", y="f1", hue="emotion", marker="o", ax=axis)
        axis.set_title(language)
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs_sorted)
        axis.set_ylim(0.0, 1.0)
        # One shared legend for the whole figure (emotions are identical across panels).
        legend_handles, legend_labels = axis.get_legend_handles_labels()
        axis.get_legend().remove()
    axes[0][0].set_ylabel("Internal-dev F1")
    figure.legend(legend_handles, legend_labels, title="Emotion", loc="center right")
    figure.suptitle(f"Per-emotion training impact{title_suffix}: {config.model_slug}")
    output_path: Path = config.figures_dir / f"{config.artifact_prefix}emotion_learning_curves{filename_suffix}.png"
    figure.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def select_lime_examples(predictions_frame: pd.DataFrame, max_examples: int = 6) -> pd.DataFrame:
    """Select a small set of errors for LIME analysis."""

    error_frame: pd.DataFrame = predictions_frame[predictions_frame["has_error"]].copy()
    selected_frame: pd.DataFrame = (
        error_frame.sort_values(["training_status", "language", "id"]).groupby("language").head(1).head(max_examples)
    )
    return selected_frame


def build_lime_classifier(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    config: RunConfig,
    language: str,
) -> Callable[[list[str]], np.ndarray]:
    """Build a LIME classifier function from generative predictions."""

    def classify_texts(texts: list[str]) -> np.ndarray:
        """Classify perturbed texts for LIME."""

        batch_languages: list[str] = [language] * len(texts)
        responses: list[str] = generate_responses(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            languages=batch_languages,
            config=config,
        )
        predictions: np.ndarray = np.asarray(
            [labels_from_emotions(emotions=parse_emotion_response(response_text=response)) for response in responses],
            dtype=float,
        )
        probabilities: np.ndarray = np.where(predictions > 0.0, 0.95, 0.05)
        return probabilities

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
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    examples_frame: pd.DataFrame,
    config: RunConfig,
    num_features: int = 10,
) -> list[Path]:
    """Save LIME explanations for selected examples. Also writes PNGs and a summary markdown."""

    output_paths: list[Path] = []
    if examples_frame.empty:
        return output_paths
    warnings.warn(
        "LIME explanations use generated labels converted to proxy probabilities. "
        "Interpret them as qualitative error probes, not calibrated probabilities.",
        stacklevel=2,
    )
    summary_rows: list[dict] = []
    for row_index, row in examples_frame.reset_index(drop=True).iterrows():
        text: str = str(row["text"])
        language: str = str(row["language"])
        classifier: Callable[[list[str]], np.ndarray] = build_lime_classifier(
            model=model,
            tokenizer=tokenizer,
            config=config,
            language=language,
        )
        prediction_vector: np.ndarray = classifier([text])[0]
        # If model predicted nothing (all proxy probs are 0.05), explain the first true label
        # instead of silently defaulting to index 0 via argmax.
        if prediction_vector.max() > 0.5:
            label_index: int = int(prediction_vector.argmax())
        else:
            true_indices: list[int] = [
                i for i, lbl in enumerate(LABEL_COLUMNS) if row.get(f"true_{lbl}", 0) == 1
            ]
            label_index = true_indices[0] if true_indices else 0

        true_labels: list[str] = [lbl for lbl in LABEL_COLUMNS if row.get(f"true_{lbl}", 0) == 1]
        predicted_labels: list[str] = [lbl for lbl in LABEL_COLUMNS if row.get(f"pred_{lbl}", 0) == 1]

        explainer: LimeTextExplainer = LimeTextExplainer(class_names=list(LABEL_COLUMNS))
        explanation: Any = explainer.explain_instance(
            text,
            classifier,
            labels=[label_index],
            num_features=num_features,
            num_samples=config.lime_num_samples,
        )

        stem: str = f"{config.artifact_prefix}lime_{row_index:02d}_{language}_{LABEL_COLUMNS[label_index]}"
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
            output_path=config.lime_dir / f"{config.artifact_prefix}lime_summary.md",
            model_slug=config.model_slug,
        )
    return output_paths


def find_zero_support_training_labels(config: RunConfig) -> list[tuple[str, str]]:
    """Return (language_name, label) pairs absent from the training overview.

    These emotions have no training examples, so they cannot be learned and their F1 is not
    meaningful (e.g. BRIGHTER English contains no ``disgust`` examples). Reads the overview written
    by ``describe_training_data``; returns an empty list if that table is missing.
    """

    overview_path: Path = config.tables_dir / f"{config.artifact_prefix}training_data_overview.csv"
    if not overview_path.exists():
        return []
    overview_frame: pd.DataFrame = pd.read_csv(overview_path)
    gaps: list[tuple[str, str]] = []
    for _, row in overview_frame.iterrows():
        for label in LABEL_COLUMNS:
            rate_column: str = f"{label}_positive_rate"
            if rate_column in overview_frame.columns and float(row[rate_column]) == 0.0:
                gaps.append((str(row["language_name"]), label))
    return gaps


def write_markdown_summary(
    metrics_frame: pd.DataFrame,
    per_label_metrics_frame: pd.DataFrame,
    lime_paths: list[Path],
    config: RunConfig,
) -> Path:
    """Write concise result bullets after final evaluation."""

    output_path: Path = config.reports_dir / f"{config.artifact_prefix}results_summary.md"
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
    trained_macro_f1: float = float(trained_frame["macro_f1"].mean()) if not trained_frame.empty else float("nan")
    zero_shot_macro_f1: float = float(zero_shot_frame["macro_f1"].mean()) if not zero_shot_frame.empty else float("nan")
    hardest_label: str = str(hardest_labels_frame.iloc[0]["label"])
    hardest_label_f1: float = float(hardest_labels_frame.iloc[0]["f1"])
    lines: list[str] = [
        f"# Results Summary: {config.model_slug}",
        "",
        f"- Model: {config.model_name}.",
        f"- Best final-dev language: {best_language} with macro-F1 {best_macro_f1:.3f}.",
        f"- Hardest final-dev language: {worst_language} with macro-F1 {worst_macro_f1:.3f}.",
        f"- Mean trained-language macro-F1: {trained_macro_f1:.3f}.",
        f"- Mean zero-shot macro-F1: {zero_shot_macro_f1:.3f}.",
        f"- Hardest average label: {hardest_label} with F1 {hardest_label_f1:.3f}.",
        f"- LIME explanations written: {len(lime_paths)}.",
    ]
    data_gaps: list[tuple[str, str]] = find_zero_support_training_labels(config=config)
    if data_gaps:
        gap_descriptions: list[str] = [f"{language_name} has no {label}" for language_name, label in data_gaps]
        lines.append(f"- Known data gaps (no training examples, F1 not meaningful): {'; '.join(gap_descriptions)}.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Study helpers (cache-aware): each study_* function loads a cached CSV when
# available, otherwise recomputes on GPU and saves it. Returns None with a
# warning when no GPU and no cache exist.
# ---------------------------------------------------------------------------

# All 28 BRIGHTER languages for the full cross-lingual heatmap.
ALL_BRIGHTER_LANGUAGES: tuple[str, ...] = (
    "afr",
    "arq",
    "ary",
    "chn",
    "deu",
    "eng",
    "esp",
    "hau",
    "hin",
    "ibo",
    "ind",
    "jav",
    "kin",
    "mar",
    "pcm",
    "ptbr",
    "ptmz",
    "ron",
    "rus",
    "sun",
    "swa",
    "swe",
    "tat",
    "ukr",
    "vmw",
    "xho",
    "yor",
    "zul",
)
# Display names for every BRIGHTER language beyond the core EVAL set.
ALL_LANGUAGE_NAMES: dict[str, str] = {
    "afr": "Afrikaans",
    "arq": "Algerian Arabic",
    "ary": "Moroccan Arabic",
    "chn": "Chinese",
    "esp": "Spanish",
    "hau": "Hausa",
    "hin": "Hindi",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "jav": "Javanese",
    "kin": "Kinyarwanda",
    "mar": "Marathi",
    "pcm": "Nigerian Pidgin",
    "ptmz": "Mozambique Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "sun": "Sundanese",
    "swe": "Swedish",
    "tat": "Tatar",
    "vmw": "Makhuwa",
    "xho": "Xhosa",
    "zul": "Zulu",
}
# Default size ladders. Full fine-tune is memory-bound (8-bit AdamW caps us at ~4B on 24 GB);
# base inference has no optimizer state so it sweeps further (up to 9B).
SCALING_BASE_MODELS: tuple[str, ...] = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B")
SCALING_FINETUNE_MODELS: tuple[str, ...] = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B")
CHINESE_BASE_MODELS: tuple[str, ...] = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B")
CHINESE_FINETUNE_MODELS: tuple[str, ...] = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B")
MULTISEED_SEEDS: tuple[int, ...] = (7, 13, 29, 41, 53)
# Per-size (train batch, accumulation, gradient_checkpointing) so the effective batch stays 16
# while fitting a 24 GB GPU under 8-bit AdamW. Unlisted models fall back to the conservative 4B row.
SCALING_SIZE_SETTINGS: dict[str, tuple[int, int, bool]] = {
    "Qwen/Qwen3.5-0.8B": (4, 4, False),
    "Qwen/Qwen3.5-2B": (2, 8, True),
    "Qwen/Qwen3.5-4B": (1, 16, True),
}


def cuda_is_available() -> bool:
    """Return whether a CUDA GPU is usable for the heavy studies."""

    return torch.cuda.is_available()


def register_all_language_names() -> None:
    """Add the full BRIGHTER language-name table so display lookups never KeyError."""

    for code, name in ALL_LANGUAGE_NAMES.items():
        LANGUAGE_NAMES.setdefault(code, name)


def show_figure(path: Path) -> None:
    """Display a saved PNG inline in a notebook. Outside IPython, prints the path."""

    try:
        from IPython.display import Image, display
    except ImportError:  # pragma: no cover - only happens outside IPython
        print(f"Figure saved to {path}")
        return
    display(Image(filename=str(path)))


def _load_or_compute_study(
    csv_path: Path,
    compute: Callable[[], pd.DataFrame],
    *,
    recompute: bool,
    description: str,
) -> pd.DataFrame | None:
    """Load csv_path if it exists and recompute is False; otherwise call compute(), save, and return."""

    if csv_path.exists() and not recompute:
        print(f"[{description}] loading cached result: {csv_path}", flush=True)
        return pd.read_csv(csv_path)
    if cuda_is_available():
        print(f"[{description}] no cache  -  computing on GPU (slow path) ...", flush=True)
    else:
        print(f"[{description}] no cache and no CUDA GPU  -  computing on CPU (very slow) ...", flush=True)
    frame: pd.DataFrame = compute()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    print(f"[{description}] wrote {csv_path}", flush=True)
    return frame


def _release_model(*objects: Any) -> None:
    """Free GPU memory held by models/trainers between sizes."""

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _repo_root() -> Path:
    """Return the repository root (the parent of the ``src`` package directory)."""

    return Path(__file__).resolve().parent.parent


def _model_token(model_name: str) -> str:
    """Return a filesystem-safe token for a model name (used in per-unit temp filenames)."""

    return re.sub(r"[^A-Za-z0-9]+", "_", model_name.split("/")[-1])


def _run_unit_in_subprocess(
    unit: str,
    out_csv: Path,
    base_config: RunConfig,
    *,
    seed: int,
    extra_args: Sequence[str],
) -> pd.DataFrame:
    """Run one model-load unit in a fresh subprocess so VRAM is fully released when the child exits."""

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    command: list[str] = [
        sys.executable,
        "-m",
        "src.brighter_emotion_pipeline_qwen",
        "--output-dir",
        str(base_config.output_dir),
        "--reports-dir",
        str(base_config.reports_dir),
        "--seed",
        str(seed),
        "--out",
        str(out_csv),
        unit,
        *extra_args,
    ]
    child_env: dict[str, str] = dict(os.environ)
    child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"  [isolated {unit}] {' '.join(extra_args)}", flush=True)
    subprocess.run(command, cwd=str(_repo_root()), env=child_env, check=True)
    return pd.read_csv(out_csv)


def _base_eval_unit(model_name: str, languages: Sequence[str], base_config: RunConfig) -> pd.DataFrame:
    """Score one untrained model per language, isolated in a child process when configured."""

    if not base_config.isolate_recompute:
        return _evaluate_base_on_languages(model_name=model_name, languages=languages, base_config=base_config)
    unit_csv: Path = base_config.tables_dir / f"_unit_base_eval_{_model_token(model_name)}.csv"
    try:
        return _run_unit_in_subprocess(
            "base-eval",
            unit_csv,
            base_config,
            seed=base_config.seed,
            extra_args=["--model", model_name, "--languages", ",".join(languages)],
        )
    finally:
        unit_csv.unlink(missing_ok=True)


def _finetune_unit(
    model_name: str,
    eval_languages: Sequence[str],
    base_config: RunConfig,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    run_suffix: str,
    save_model: bool = False,
) -> pd.DataFrame:
    """Full-fine-tune one model and score it, isolated in a child process when configured."""

    if not base_config.isolate_recompute:
        return _finetune_and_evaluate(
            model_name=model_name,
            eval_languages=eval_languages,
            base_config=base_config,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            run_suffix=run_suffix,
            save_model=save_model,
        )
    unit_csv: Path = base_config.tables_dir / f"_unit_finetune_{_model_token(model_name)}.csv"
    extra_args: list[str] = [
        "--model",
        model_name,
        "--eval-languages",
        ",".join(eval_languages),
        "--epochs",
        str(epochs),
        "--lr",
        f"{learning_rate:g}",
        "--run-suffix",
        run_suffix,
    ]
    if save_model:
        extra_args.append("--save-model")
    try:
        return _run_unit_in_subprocess("finetune", unit_csv, base_config, seed=seed, extra_args=extra_args)
    finally:
        unit_csv.unlink(missing_ok=True)


def _seed_sweep_unit(
    seed: int,
    base_config: RunConfig,
    *,
    max_epochs: int,
    learning_rate: float,
    run_suffix: str,
) -> pd.DataFrame:
    """Run one seed's lr/epoch sweep history, isolated in a child process when configured."""

    if not base_config.isolate_recompute:
        seed_config: RunConfig = replace(base_config, seed=seed, run_suffix=run_suffix)
        history: pd.DataFrame = run_lr_epoch_sweep(
            base_config=seed_config, learning_rates=[learning_rate], max_epochs=max_epochs
        )
        history["seed"] = seed
        return history
    unit_csv: Path = base_config.tables_dir / f"_unit_seed_sweep_s{seed}.csv"
    try:
        return _run_unit_in_subprocess(
            "seed-sweep",
            unit_csv,
            base_config,
            seed=seed,
            extra_args=["--max-epochs", str(max_epochs), "--lr", f"{learning_rate:g}", "--run-suffix", run_suffix],
        )
    finally:
        unit_csv.unlink(missing_ok=True)


def _evaluate_base_on_languages(
    model_name: str,
    languages: Sequence[str],
    base_config: RunConfig,
) -> pd.DataFrame:
    """Load one untrained model and score the official dev split per language (GPU path)."""

    config: RunConfig = replace(base_config, model_name=model_name, gradient_checkpointing=False)
    print(f"  base eval: {model_name}", flush=True)
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(config=config)
    model: Any = load_model(config=config)
    if torch.cuda.is_available():
        model.to("cuda")
    param_billions: float = sum(parameter.numel() for parameter in model.parameters()) / 1e9
    eval_dataset: Dataset = load_combined_eval_split(languages=languages, split="dev", config=config)
    frame: pd.DataFrame = macro_f1_by_language(model=model, tokenizer=tokenizer, dataset=eval_dataset, config=config)
    frame.insert(0, "model", model_name)
    frame.insert(1, "params_billion", round(param_billions, 3))
    frame["training_status"] = [
        "zero-shot" if language in ZERO_SHOT_LANGUAGES else "trained-lang" for language in frame["language"]
    ]
    _release_model(model, tokenizer)
    return frame


def _finetune_and_evaluate(
    model_name: str,
    eval_languages: Sequence[str],
    base_config: RunConfig,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    run_suffix: str,
    save_model: bool = False,
) -> pd.DataFrame:
    """Full-fine-tune one model on the locked recipe and score ``eval_languages`` on official dev."""

    batch, accumulation, checkpointing = SCALING_SIZE_SETTINGS.get(model_name, (1, 16, True))
    config: RunConfig = replace(
        base_config,
        model_name=model_name,
        run_suffix=run_suffix,
        seed=seed,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=accumulation,
        gradient_checkpointing=checkpointing,
        select_best_model=False,
        optim="paged_adamw_8bit",
    )
    print(
        f"  fine-tune {model_name} | eff batch {batch * accumulation} | {epochs} ep | lr={learning_rate:g}", flush=True
    )
    set_random_seed(seed=config.seed)
    tokenizer: PreTrainedTokenizerBase = load_tokenizer(config=config)
    splits: DatasetDict = create_train_internal_dev_split(config=config)
    model: Any = load_model(config=config)
    param_billions: float = sum(parameter.numel() for parameter in model.parameters()) / 1e9
    tokenized_train: Dataset = tokenize_dataset(dataset=splits["train"], tokenizer=tokenizer, config=config)
    tokenized_internal_dev: Dataset = tokenize_dataset(
        dataset=splits["internal_dev"], tokenizer=tokenizer, config=config
    )
    trainer: Any = create_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_train_dataset=tokenized_train,
        tokenized_internal_dev_dataset=tokenized_internal_dev,
        config=config,
    )
    trainer.train()
    if save_model:
        model.save_pretrained(config.model_dir)
        tokenizer.save_pretrained(config.model_dir)
    eval_dataset: Dataset = load_combined_eval_split(languages=eval_languages, split="dev", config=config)
    frame: pd.DataFrame = macro_f1_by_language(model=model, tokenizer=tokenizer, dataset=eval_dataset, config=config)
    frame.insert(0, "model", model_name)
    frame.insert(1, "params_billion", round(param_billions, 3))
    frame["training_status"] = [
        "zero-shot" if language in ZERO_SHOT_LANGUAGES else "trained-lang" for language in frame["language"]
    ]
    _release_model(trainer, model, tokenizer)
    return frame


def study_scaling_base(
    base_config: RunConfig,
    models: Sequence[str] = SCALING_BASE_MODELS,
    *,
    recompute: bool = False,
) -> pd.DataFrame | None:
    """Base-model zero-shot macro-F1 across sizes on the six EVAL languages (cache-aware)."""

    csv_path: Path = base_config.tables_dir / "qwen3.5_scaling_base_dev.csv"

    def compute() -> pd.DataFrame:
        frames: list[pd.DataFrame] = [
            _base_eval_unit(model_name=name, languages=EVAL_LANGUAGES, base_config=base_config) for name in models
        ]
        return pd.concat(frames, ignore_index=True)

    return _load_or_compute_study(csv_path, compute, recompute=recompute, description="scaling-base")


def study_scaling_finetune(
    base_config: RunConfig,
    models: Sequence[str] = SCALING_FINETUNE_MODELS,
    *,
    epochs: int = 3,
    learning_rate: float = 1e-5,
    recompute: bool = False,
) -> pd.DataFrame | None:
    """Full-fine-tuned macro-F1 across sizes on the six EVAL languages (cache-aware, slow)."""

    csv_path: Path = base_config.tables_dir / "qwen3.5_scaling_finetuned_dev.csv"

    def compute() -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for name in models:
            frames.append(
                _finetune_unit(
                    model_name=name,
                    eval_languages=EVAL_LANGUAGES,
                    base_config=base_config,
                    seed=base_config.seed,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    run_suffix=f"scaling_ft_{name.split('/')[-1]}",
                )
            )
            # Persist after every size so a later crash never loses an already-trained result.
            pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
        return pd.concat(frames, ignore_index=True)

    return _load_or_compute_study(csv_path, compute, recompute=recompute, description="scaling-finetune")


def study_all_languages_base(
    base_config: RunConfig,
    models: Sequence[str] = SCALING_BASE_MODELS,
    *,
    recompute: bool = False,
) -> pd.DataFrame | None:
    """Base-model zero-shot macro-F1 across sizes on all 28 BRIGHTER languages (cache-aware, slow)."""

    register_all_language_names()
    csv_path: Path = base_config.tables_dir / "qwen3.5_all_languages_base_dev.csv"

    def compute() -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for name in models:
            frames.append(_base_eval_unit(model_name=name, languages=ALL_BRIGHTER_LANGUAGES, base_config=base_config))
            pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
        return pd.concat(frames, ignore_index=True)

    return _load_or_compute_study(csv_path, compute, recompute=recompute, description="all-languages-base")


def study_chinese(
    base_config: RunConfig,
    base_models: Sequence[str] = CHINESE_BASE_MODELS,
    finetune_models: Sequence[str] = CHINESE_FINETUNE_MODELS,
    *,
    epochs: int = 3,
    learning_rate: float = 1e-5,
    recompute: bool = False,
) -> dict[str, pd.DataFrame | None]:
    """Chinese base vs fine-tuned macro-F1 across sizes (cache-aware). Returns {"base": frame, "finetuned": frame}."""

    register_all_language_names()
    base_csv: Path = base_config.tables_dir / "qwen3.5_chinese_base_dev.csv"
    ft_csv: Path = base_config.tables_dir / "qwen3.5_chinese_finetuned_dev.csv"

    def compute_base() -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for name in base_models:
            frame: pd.DataFrame = _base_eval_unit(model_name=name, languages=("chn",), base_config=base_config)
            frame["training_status"] = "zero-shot"
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def compute_finetune() -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for name in finetune_models:
            frame: pd.DataFrame = _finetune_unit(
                model_name=name,
                eval_languages=("chn",),
                base_config=base_config,
                seed=base_config.seed,
                epochs=epochs,
                learning_rate=learning_rate,
                run_suffix=f"chinese_ft_{name.split('/')[-1]}",
                save_model=True,
            )
            frame["training_status"] = "zero-shot (FT on eng+swa+ukr)"
            frames.append(frame)
            pd.concat(frames, ignore_index=True).to_csv(ft_csv, index=False)
        return pd.concat(frames, ignore_index=True)

    base_frame: pd.DataFrame | None = _load_or_compute_study(
        base_csv, compute_base, recompute=recompute, description="chinese-base"
    )
    ft_frame: pd.DataFrame | None = _load_or_compute_study(
        ft_csv, compute_finetune, recompute=recompute, description="chinese-finetune"
    )
    return {"base": base_frame, "finetuned": ft_frame}


def aggregate_multiseed_bands(per_seed_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed sweep history into mean+std bands per language and epoch."""

    per_seed_frame = per_seed_frame[per_seed_frame["language"].isin(TRAIN_LANGUAGES)].copy()
    # trained_mean: average across the trained languages within each (seed, epoch), then band over seeds.
    trained_per_seed: pd.DataFrame = per_seed_frame.groupby(["seed", "epoch"], as_index=False)["macro_f1"].mean()
    trained_bands: pd.DataFrame = (
        trained_per_seed.groupby("epoch")
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_seeds=("macro_f1", "count"),
        )
        .reset_index()
    )
    trained_bands.insert(0, "scope", "trained_mean")

    language_bands: pd.DataFrame = (
        per_seed_frame.groupby(["language", "epoch"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_seeds=("macro_f1", "count"),
        )
        .reset_index()
        .rename(columns={"language": "scope"})
    )
    return pd.concat([trained_bands, language_bands], ignore_index=True)


def study_multiseed(
    base_config: RunConfig,
    seeds: Sequence[int] = MULTISEED_SEEDS,
    *,
    max_epochs: int = 4,
    learning_rate: float = 1e-5,
    recompute: bool = False,
) -> pd.DataFrame | None:
    """Train the 0.8B recipe once per seed, aggregate per-epoch macro-F1 into mean+std bands (cache-aware)."""

    bands_csv: Path = base_config.tables_dir / f"{base_config.artifact_prefix}multiseed_macro_f1_bands.csv"

    def compute() -> pd.DataFrame:
        per_seed_frames: list[pd.DataFrame] = [
            _seed_sweep_unit(
                seed=seed,
                base_config=base_config,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                run_suffix=f"multiseed_s{seed}",
            )
            for seed in seeds
        ]
        per_seed_frame: pd.DataFrame = pd.concat(per_seed_frames, ignore_index=True)
        return aggregate_multiseed_bands(per_seed_frame=per_seed_frame)

    return _load_or_compute_study(bands_csv, compute, recompute=recompute, description="multiseed")


def plot_scaling_curve(
    base_frame: pd.DataFrame,
    finetuned_frame: pd.DataFrame | None,
    config: RunConfig,
) -> Path:
    """Mean macro-F1 vs model size: base vs fine-tuned, split by training status."""

    def summarize(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
        grouped: pd.DataFrame = frame.groupby(["params_billion", "training_status"], as_index=False)["macro_f1"].mean()
        grouped["regime"] = regime
        return grouped

    parts: list[pd.DataFrame] = [summarize(base_frame, "base")]
    if finetuned_frame is not None:
        parts.append(summarize(finetuned_frame, "fine-tuned"))
    plot_frame: pd.DataFrame = pd.concat(parts, ignore_index=True)
    plot_frame["series"] = plot_frame["regime"] + " / " + plot_frame["training_status"]

    output_path: Path = config.figures_dir / "qwen3.5_scaling_curve.png"
    plt.figure(figsize=(9, 5.5))
    axis: plt.Axes = sns.lineplot(
        data=plot_frame, x="params_billion", y="macro_f1", hue="series", style="series", markers=True, dashes=True
    )
    axis.set_xlabel("Model size (billions of parameters)")
    axis.set_ylabel("Mean macro-F1 (official dev)")
    axis.set_title("Size scaling: base vs fine-tuned, trained vs zero-shot")
    axis.set_ylim(0.0, 0.8)
    axis.legend(title="Regime / language group", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_language_size_heatmap(all_language_frame: pd.DataFrame, config: RunConfig) -> Path:
    """28-language x 4-size heatmap of base zero-shot macro-F1. Languages sorted by 9B score."""

    register_all_language_names()
    pivot: pd.DataFrame = all_language_frame.pivot_table(
        index="language_name", columns="params_billion", values="macro_f1"
    )
    largest: float = max(pivot.columns)
    pivot = pivot.sort_values(by=largest, ascending=False)
    output_path: Path = config.figures_dir / "qwen3.5_language_size_heatmap.png"
    plt.figure(figsize=(8, 11))
    axis: plt.Axes = sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=0.9, cbar_kws={"label": "macro-F1"}
    )
    axis.set_xlabel("Model size (billion parameters)")
    axis.set_ylabel("Language")
    axis.set_title("Base zero-shot macro-F1: 28 languages x 4 sizes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_multiseed_bands(bands_frame: pd.DataFrame, config: RunConfig) -> Path:
    """Plot mean +/- std macro-F1 versus epoch for the trained languages and their mean (noise floor)."""

    output_path: Path = config.figures_dir / f"{config.artifact_prefix}multiseed_macro_f1_bands.png"
    scopes: list[str] = ["trained_mean", *sorted(s for s in bands_frame["scope"].unique() if s != "trained_mean")]
    plt.figure(figsize=(9, 5.5))
    axis: plt.Axes = plt.gca()
    for scope in scopes:
        scope_frame: pd.DataFrame = bands_frame[bands_frame["scope"] == scope].sort_values("epoch")
        emphasis: float = 2.4 if scope == "trained_mean" else 1.2
        line = axis.plot(
            scope_frame["epoch"], scope_frame["macro_f1_mean"], marker="o", linewidth=emphasis, label=scope
        )[0]
        axis.fill_between(
            scope_frame["epoch"],
            scope_frame["macro_f1_mean"] - scope_frame["macro_f1_std"],
            scope_frame["macro_f1_mean"] + scope_frame["macro_f1_std"],
            color=line.get_color(),
            alpha=0.15,
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Internal-dev macro-F1 (mean +/- std)")
    n_seeds: int = int(bands_frame["n_seeds"].max())
    axis.set_title(f"Multi-seed noise floor (n={n_seeds} seeds): {config.model_slug}")
    axis.set_xticks(sorted(bands_frame["epoch"].unique()))
    axis.set_ylim(0.0, 0.8)
    axis.legend(title="Scope")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_chinese_base_vs_finetuned(
    base_frame: pd.DataFrame,
    finetuned_frame: pd.DataFrame | None,
    config: RunConfig,
) -> Path:
    """Chinese macro-F1 vs model size, base vs fine-tuned."""

    base_points: pd.DataFrame = base_frame[base_frame["language"] == "chn"][["params_billion", "macro_f1"]].copy()
    base_points["regime"] = "base (zero-shot)"
    parts: list[pd.DataFrame] = [base_points]
    if finetuned_frame is not None:
        ft_points: pd.DataFrame = finetuned_frame[finetuned_frame["language"] == "chn"][
            ["params_billion", "macro_f1"]
        ].copy()
        ft_points["regime"] = "fine-tuned (on eng+swa+ukr)"
        parts.append(ft_points)
    plot_frame: pd.DataFrame = pd.concat(parts, ignore_index=True)
    output_path: Path = config.figures_dir / "qwen3.5_chinese_base_vs_finetuned.png"
    plt.figure(figsize=(8, 5))
    axis: plt.Axes = sns.lineplot(
        data=plot_frame, x="params_billion", y="macro_f1", hue="regime", style="regime", markers=True, dashes=False
    )
    axis.set_xlabel("Model size (billion parameters)")
    axis.set_ylabel("Chinese macro-F1 (official dev)")
    axis.set_title("Chinese transfer: base vs fine-tuned across sizes")
    axis.set_ylim(0.0, 0.7)
    axis.legend(title="Regime")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# --------------------------------------------------------------------------------------------------
# CLI entry point for isolated subprocess units. Each subcommand runs one model load and writes
# the result to --out; called by _run_unit_in_subprocess when isolate_recompute is set.
# --------------------------------------------------------------------------------------------------


def _cli_build_config(args: argparse.Namespace, *, run_suffix: str = "") -> RunConfig:
    """Reconstruct a leaf RunConfig for a CLI unit (isolation disabled to avoid re-spawning)."""

    config: RunConfig = RunConfig(
        output_dir=Path(args.output_dir),
        reports_dir=Path(args.reports_dir),
        seed=args.seed,
        run_suffix=run_suffix,
        isolate_recompute=False,
    )
    ensure_directories(config=config)
    register_all_language_names()
    return config


def _cli_base_eval(args: argparse.Namespace) -> None:
    config: RunConfig = _cli_build_config(args)
    languages: tuple[str, ...] = tuple(part for part in args.languages.split(",") if part)
    frame: pd.DataFrame = _evaluate_base_on_languages(model_name=args.model, languages=languages, base_config=config)
    frame.to_csv(args.out, index=False)


def _cli_finetune(args: argparse.Namespace) -> None:
    config: RunConfig = _cli_build_config(args)
    eval_languages: tuple[str, ...] = tuple(part for part in args.eval_languages.split(",") if part)
    frame: pd.DataFrame = _finetune_and_evaluate(
        model_name=args.model,
        eval_languages=eval_languages,
        base_config=config,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.lr,
        run_suffix=args.run_suffix,
        save_model=args.save_model,
    )
    frame.to_csv(args.out, index=False)


def _cli_seed_sweep(args: argparse.Namespace) -> None:
    config: RunConfig = _cli_build_config(args, run_suffix=args.run_suffix)
    history: pd.DataFrame = run_lr_epoch_sweep(base_config=config, learning_rates=[args.lr], max_epochs=args.max_epochs)
    history["seed"] = args.seed
    history.to_csv(args.out, index=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Isolated single-unit compute for the cache-aware studies."
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", required=True)
    subparsers: Any = parser.add_subparsers(dest="unit", required=True)

    base_eval_parser: argparse.ArgumentParser = subparsers.add_parser("base-eval")
    base_eval_parser.add_argument("--model", required=True)
    base_eval_parser.add_argument("--languages", required=True, help="comma-separated language codes")
    base_eval_parser.set_defaults(func=_cli_base_eval)

    finetune_parser: argparse.ArgumentParser = subparsers.add_parser("finetune")
    finetune_parser.add_argument("--model", required=True)
    finetune_parser.add_argument("--eval-languages", required=True, help="comma-separated language codes")
    finetune_parser.add_argument("--epochs", type=int, required=True)
    finetune_parser.add_argument("--lr", type=float, required=True)
    finetune_parser.add_argument("--run-suffix", required=True)
    finetune_parser.add_argument("--save-model", action="store_true")
    finetune_parser.set_defaults(func=_cli_finetune)

    seed_sweep_parser: argparse.ArgumentParser = subparsers.add_parser("seed-sweep")
    seed_sweep_parser.add_argument("--max-epochs", type=int, required=True)
    seed_sweep_parser.add_argument("--lr", type=float, required=True)
    seed_sweep_parser.add_argument("--run-suffix", required=True)
    seed_sweep_parser.set_defaults(func=_cli_seed_sweep)

    return parser


if __name__ == "__main__":
    cli_args: argparse.Namespace = _build_arg_parser().parse_args()
    cli_args.func(cli_args)
