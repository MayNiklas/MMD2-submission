# MMD2 Project - Multilingual Emotion Detection

> Emotion detection as constrained JSON generation with Qwen3.5

- MMD II Project, Topic 3, Team 1
- Christoph Geron, Niklas Steffen, Julian Steffen
- Model weights: [MayNiklas/MMD2-brighter-emotion](https://huggingface.co/MayNiklas/MMD2-brighter-emotion)

This repository fine-tunes `Qwen/Qwen3.5-0.8B` as a **decoder** for the BRIGHTER multi-label
emotion task.

- **Train languages:** English (`eng`), Swahili (`swa`), Ukrainian (`ukr`).
- **Evaluation languages:** the three trained languages plus zero-shot German (`deu`),
  Brazilian Portuguese (`ptbr`), and Yoruba (`yor`).
- **Dataset:** `brighter-dataset/BRIGHTER-emotion-categories` (Hugging Face).

The single notebook `notebooks/brighter_emotion_mvp_qwen.ipynb` is the source for every result,
table, and figure in the report. Running it end-to-end on `full` regenerates everything we hand in.

## Setup

Two environments are supported.

**Conda (Apple Silicon or CUDA):**

```bash
conda env create -f environment.yml
conda activate mmd2-emotion
python -m ipykernel install --user --name mmd2-emotion --display-name "Python (mmd2-emotion)"
```

**Nix (CUDA dev shell):**

```bash
nix develop
```

## Running

Open the notebook and run it top to bottom:

```bash
jupyter notebook notebooks/brighter_emotion_mvp_qwen.ipynb
```

Set `RUN_MODE` in Step 1:

- `"smoke"`  -  2-step backend check on a tiny sample (verifies the pipeline; macro-F1 is ~0).
- `"lite"`  -  bounded real run (seeded subsample, eval capped per language) for a quick sanity pass.
- `"full"`  -  full train split and full official `dev` evaluation for the final reported numbers.

For long unattended runs, execute the notebook headless:

```bash
python scripts/run_notebook.py notebooks/brighter_emotion_mvp_qwen.ipynb --all --inplace --timeout 0
```

## What the notebook produces

1. **Core model**  -  train on eng+swa+ukr, evaluate the six languages (macro-F1, per-label F1),
   base-vs-fine-tuned deltas, and LIME error analysis, with all figures rendered inline.
2. **Size scaling study**  -  base vs fine-tuned macro-F1 across 0.8B/2B/4B/9B, plus a
   28-language x 4-size base heatmap. Headline finding: *scale is the dominant lever*.
3. **Does heritage help?**  -  Chinese (the only CJK language in BRIGHTER), base vs fine-tuned.
4. **Multi-seed noise floor**  -  the run-to-run std (~±0.01) every other delta is measured against.
5. **What we tried that did not work**  -  the closed levers, for honest scope.

Every heavy study is **cache-aware**: it loads a committed CSV from `reports/tables/` if present,
otherwise it recomputes the result and saves it. The small aggregate CSVs that back every table and
figure are committed, so the notebook renders end-to-end on any machine. All artifacts carry a
`qwen3.5` prefix.

## Reproducing the results

The notebook self-bootstraps - it never depends on a cache being present:

- **Review / no GPU.** A fresh clone already ships the backing CSVs in `reports/tables/`, so running
  the notebook (or just opening it) reproduces every table and figure without a GPU and without
  recomputing anything. The core 0.8B training cell still needs a GPU if you want to retrain it.
- **Regenerate from scratch (GPU).** Delete the relevant CSV(s) in `reports/tables/` and re-run; the
  matching study recomputes and re-saves them. Each heavy study isolates **one model load per child
  process** (`RunConfig.isolate_recompute`, default on), so sweeping 0.8B→9B in a single study frees
  VRAM between sizes and fits a 24 GB GPU without the fragmentation OOM that an in-kernel sweep hits.
  A full from-scratch regeneration of the scaling/Chinese/multi-seed studies takes several hours.

The same isolated units are runnable directly, e.g.
`python -m src.brighter_emotion_pipeline_qwen --out <csv> base-eval --model Qwen/Qwen3.5-4B --languages eng,swa,ukr`.

## Protocol

The official BRIGHTER `train` split is divided into train and internal-dev with seed `42069`.
The official `dev` split is reserved for final evaluation only. The primary metric is macro-F1 over
the six emotion labels; per-label F1 and zero-shot language transfer are reported separately.
