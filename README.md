# BioNER-DS — Multi-Entity Biomedical NER with Distant Supervision

Pipeline for training and evaluating biomedical Named Entity Recognition (NER)
models that expand the BC5CDR (Chemical + Disease) schema toward
virus-centric drug repurposing (Chemical + Disease + Virus + Gene).

Six training configurations are supported from a single code base; results are
aggregated across multiple seeds for paper-grade mean ± std reporting.

## Project Structure

```
BioNER-DS/
├── configs/                      # YAML experiment configs (one per scenario)
├── dataset/                      # JSONL datasets (gitignored; user-provided)
│   ├── bc5cdr/                   # BC5CDR train/validation/test
│   └── pubmed/                   # PubMed silver corpus + 100-sentence gold test
├── notebooks/
│   └── run_experiments.ipynb     # One-click experiment runner
├── outputs/                      # Run artifacts (gitignored)
│   ├── checkpoints/<config>/seed_<seed>/best_model/
│   ├── logs/<config>/seed_<seed>.log
│   └── results/<config>/seed_<seed>/eval_<test_set>.json
├── src/                          # Library code (config, data, models, training, eval, inference)
├── train.py                      # Top-level training entry
├── evaluate.py                   # Standalone evaluation entry
├── predict.py                    # Standalone inference entry (KG interface)
└── requirements.txt
```

## Experiment Matrix

| Config | Backbone   | Strategy           | Entities                 |
|--------|------------|--------------------|--------------------------|
| 1      | BERT-base  | single             | Chemical, Disease        |
| 2      | BioBERT    | single             | Chemical, Disease        |
| 3      | PubMedBERT | single             | Chemical, Disease        |
| 4      | PubMedBERT | sequential         | + Virus, Gene (4-entity) |
| 5      | PubMedBERT | joint_uniform      | + Virus, Gene (4-entity) |
| 6      | PubMedBERT | joint_noise_aware  | + Virus, Gene (4-entity) |

Configs 1–3 train on BC5CDR gold labels only. Configs 4–6 add the silver
PubMed corpus through different mixing strategies (sequential, joint
uniform-weight, joint noise-aware-weight).

### Expected behaviours that look like bugs but aren't

- **Config 4 (sequential) reports near-zero BC5CDR F1 on Chemical/Disease**
  after phase 2. This is **catastrophic forgetting** — phase 2 trains on the
  silver corpus where Virus/Gene dominate, and the model erases its
  phase-1 Chemical/Disease knowledge. The pipeline explicitly logs the
  per-entity forgetting score (`Δ F1`) for the paper. Use config 5 (joint
  uniform) as the contrast that preserves both schemas. Do not "fix"
  config 4 — the gap *is* the experimental finding.
- **Chemical/Disease F1 = 0 on `test_pubmed`.** The PubMed gold test only
  contains 4 Chemical and 0 Disease spans (it was annotated for Virus/Gene
  evaluation). Entities with support below `LOW_SUPPORT_THRESHOLD = 10` are
  marked **N/A** in `aggregated_results.md`; the raw numbers remain in the
  JSON. Always report Chemical/Disease from `test_bc5cdr`, not `test_pubmed`.

## Requirements

* Python 3.10–3.14
* GPU recommended: NVIDIA T4 16GB or 3080 Ti 12GB (fp16 enabled by default).
* CPU runs are supported for smoke tests; full training is impractical without
  a GPU.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

PyTorch CUDA users should install the CUDA build manually first, then run the
command above:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements.txt
```

## Dataset Layout

The configs expect five files under `dataset/`:

```
dataset/bc5cdr/bc5cdr_train.jsonl
dataset/bc5cdr/bc5cdr_validation.jsonl
dataset/bc5cdr/bc5cdr_test.jsonl
dataset/pubmed/pubmed_scrapping.jsonl     # silver corpus
dataset/pubmed/pubmed_test.jsonl          # 100-sentence manual gold
```

Each line is a JSON record with `tokens` plus one or more BIO label fields
(`decoded_tags`, `decoded_tags_pseudo_final`). See `specs/02_data_spec.md` for
the full schema and the field-to-config mapping table.

## Running a Single Experiment

Train one config with one seed:

```bash
python train.py --config configs/config_3_pubmedbert.yaml --seed 42
```

Run a quick smoke test (1 epoch, 16 training sentences):

```bash
python train.py --config configs/config_3_pubmedbert.yaml --seed 42 --smoke_test
```

CLI overrides that take precedence over YAML defaults:

```
--seed INT
--output_dir STR
--epochs INT
--batch_size INT
--learning_rate FLOAT
```

## Running All Six Configs (Multi-Seed)

Use `notebooks/run_experiments.ipynb` — it loops over every config and seed
in-kernel (see the "Notebook Entry Point" section below).

For a headless / SSH workflow, loop through the configs from the shell:

```bash
for cfg in configs/config_*.yaml; do
    for seed in 42 1337 2024; do
        python train.py --config "$cfg" --seed "$seed"
    done
    python -m src.utils.aggregate --config "$cfg"
done
```

After every run, evaluation JSONs land under
`outputs/results/<config>/seed_<seed>/`. The aggregator produces:

* `outputs/results/<config>/aggregated_results.json` — machine-readable.
* `outputs/results/<config>/aggregated_results.md` — paper-ready table.

You can re-aggregate at any time:

```bash
python -m src.utils.aggregate --config configs/config_3_pubmedbert.yaml
```

## Standalone Evaluation

Re-score a checkpoint without retraining:

```bash
python evaluate.py \
    --checkpoint outputs/checkpoints/config_4_sequential/seed_42/phase2/best_model \
    --config configs/config_4_sequential.yaml \
    --test_set test_pubmed
```

## Inference for the KG Pipeline

`predict.py` reads either a JSONL with pre-tokenized records or a plain text
file (one sentence per line) and writes predictions in the format expected by
the downstream KG construction step.

```bash
python predict.py \
    --checkpoint outputs/checkpoints/config_4_sequential/seed_42/phase2/best_model \
    --input dataset/pubmed/pubmed_test.jsonl \
    --output outputs/predictions/config4_pubmed.jsonl
```

Output schema (one JSON object per line):

```json
{
  "sentence_id": "0",
  "tokens": ["The", "SARS-CoV-2", "spike", "binds", "to", "ACE2"],
  "predictions": [
    {"entity": "SARS-CoV-2", "type": "Virus", "start_token": 1, "end_token": 1, "score": 0.94},
    {"entity": "ACE2",       "type": "Gene",  "start_token": 5, "end_token": 5, "score": 0.89}
  ]
}
```

## Notebook Entry Point

`notebooks/run_experiments.ipynb` provides a one-click runner that:

1. Sets the working directory to the project root and checks GPU + library
   versions.
2. Optionally installs `requirements.txt`.
3. Verifies dataset files.
4. Lets you pick the seed list and a `SMOKE_TEST` flag once.
5. Calls the trainer functions directly (`train_single`, `train_sequential`,
   `train_joint`) per config — no `bash`, no subprocess, no environment
   crossing. The kernel's Python is used throughout.
6. Renders every `aggregated_results.md` for quick comparison.
7. Optionally runs inference on the PubMed test set via `NERPredictor`.

You can skip any cell without breaking the rest of the notebook. Pick a
kernel that points at the venv where `requirements.txt` is installed.

The same notebook is the recommended entry point on Vast.ai / SSH — open it
through a remote Jupyter session and run the cells. For a non-interactive
shell loop, use the `python train.py` snippet in the "Running All Six Configs"
section above.

## Smoke Tests

Use the `--smoke_test` flag to validate the training pipeline without paying
the full epoch cost. The flag caps each dataset to 16 sentences and forces a
single epoch.

```bash
python train.py --config configs/config_3_pubmedbert.yaml --smoke_test --batch_size 4
python train.py --config configs/config_4_sequential.yaml --smoke_test --batch_size 4
python train.py --config configs/config_6_joint_noise_aware.yaml --smoke_test --batch_size 4
```

The notebook exposes the same option via the `SMOKE_TEST` variable in cell 4.

## Reproducibility

* Every run sets seeds for `random`, `numpy`, `torch`, `torch.cuda`, and the
  DataLoader workers (see `src/utils/seed.py`).
* HuggingFace `TrainingArguments` receives `seed` and `data_seed` explicitly.
* A YAML snapshot of the resolved config is written to
  `outputs/results/<config>/seed_<seed>/config_snapshot.yaml` per run.
* The environment (Python, Torch, CUDA, GPU) is logged at startup.

Bit-exact reproducibility on GPU is not guaranteed due to non-deterministic
CUDA kernels; small numeric drift across runs with the same seed is expected.

## Output Layout (per run)

```
outputs/
├── checkpoints/<config>/seed_<seed>/
│   ├── best_model/                   # single-source / joint
│   ├── phase1/best_model/            # sequential phase 1 checkpoint
│   └── phase2/best_model/            # sequential phase 2 (final) checkpoint
├── logs/<config>/seed_<seed>_*.log
└── results/<config>/
    ├── seed_<seed>/
    │   ├── config_snapshot.yaml
    │   ├── eval_test_bc5cdr.json
    │   └── eval_test_pubmed.json     # configs 4-6 only
    ├── aggregated_results.json
    └── aggregated_results.md
```

## License

See `LICENSE`.
