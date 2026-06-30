#!/usr/bin/env bash
set -euo pipefail

HF=/home/nik/miniconda3/envs/mmd2-emotion/bin/hf
REPO=MayNiklas/MMD2-brighter-emotion
OUTPUTS=/home/nik/code/github.com/MayNiklas/MMD2-project/outputs

echo "==> Uploading README..."
$HF upload "$REPO" "$(dirname "$0")/README.md" README.md

echo "==> Uploading mBERT..."
$HF upload "$REPO" "$OUTPUTS/model/" mbert/

echo "==> Uploading Qwen3.5-0.8B FT..."
$HF upload "$REPO" "$OUTPUTS/qwen3.5-0.8B_model/" qwen3.5-0.8b/

echo "==> Uploading Qwen3.5-0.8B Chinese FT..."
$HF upload "$REPO" "$OUTPUTS/qwen3.5-0.8B_chinese_ft_Qwen3.5-0.8B_model/" qwen3.5-0.8b-chinese-ft/

echo "==> Uploading Qwen3.5-2B Chinese FT..."
$HF upload "$REPO" "$OUTPUTS/qwen3.5-2B_chinese_ft_Qwen3.5-2B_model/" qwen3.5-2b-chinese-ft/

echo "==> Uploading Qwen3.5-4B Chinese FT..."
$HF upload "$REPO" "$OUTPUTS/qwen3.5-4B_chinese_ft_Qwen3.5-4B_model/" qwen3.5-4b-chinese-ft/

echo "==> Done! https://huggingface.co/$REPO"
