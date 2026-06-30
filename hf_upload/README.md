---
license: apache-2.0
language:
- en
- sw
- uk
- zh
tags:
- text-classification
- multilingual
- emotion-recognition
---

# MMD2 — BRIGHTER Multilingual Emotion Classifiers

Fine-tuned models for the **BRIGHTER** benchmark (SemEval-2025 Task 11):
multilingual multi-label emotion classification.

Submitted as part of the **Mining Media Data 2** course project.

## Task

- **Input:** short text (tweet / comment) in one of 28 languages
- **Output:** one or more emotion labels from {anger, disgust, fear, joy, sadness, surprise}
- **Metric:** macro-F1 over emotion labels

## Models in this repository

| Directory | Base model |
|---|---|
| `mbert/` | `google-bert/bert-base-multilingual-cased` |
| `qwen3.5-0.8b/` | `Qwen/Qwen3.5-0.8B` |
| `qwen3.5-0.8b-chinese-ft/` | `Qwen/Qwen3.5-0.8B` |
| `qwen3.5-2b/` | `Qwen/Qwen3.5-2B` |
| `qwen3.5-2b-chinese-ft/` | `Qwen/Qwen3.5-2B` |
| `qwen3.5-4b/` | `Qwen/Qwen3.5-4B` |
| `qwen3.5-4b-chinese-ft/` | `Qwen/Qwen3.5-4B` |

## Dataset

[BRIGHTER (SemEval-2025 Task 11)](https://brighter-dataset.github.io/)
