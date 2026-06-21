<div align="center">

# Multi-Entity Biomedical NER with Distant Supervision
### Comparing Training Strategies for Schema Expansion toward Virus-Centric Drug Repurposing

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.40%2B-FFD21E?logo=huggingface&logoColor=black)](https://github.com/huggingface/transformers)
[![seqeval](https://img.shields.io/badge/seqeval-1.2%2B-4C1?logo=python&logoColor=white)](https://github.com/chakki-works/seqeval)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

BC5CDR ships with two entity types: Chemical and Disease. For virus-centric drug repurposing the model also needs Virus and Gene. This project expands the entity schema by attaching distant-supervision silver labels to a scraped PubMed corpus, then compares several ways of mixing the original gold data with the noisier silver data so that a single biomedical NER model recognises all four entity types at once.

</div>

## Methodology

Six configurations share the same code base. They use the same model factory, the same JSONL loader, and the same BIO-to-subword alignment. What changes between them is the backbone and how the training data is mixed.

### Three single-source baselines (configs 1 through 3)

These configurations train on BC5CDR alone. They isolate the contribution of the backbone before any silver data enters the pipeline.

| Config | Backbone | Strategy | Entities |
|---|---|---|---|
| 1 | `bert-base-uncased` | single source | Chemical, Disease |
| 2 | `dmis-lab/biobert-base-cased-v1.2` | single source | Chemical, Disease |
| 3 | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | single source | Chemical, Disease |

PubMedBERT wins these comparisons cleanly, so it is the backbone used by every 4-entity experiment below.

### Three schema-expansion strategies (configs 4 through 6)

These three configurations test different ways of teaching the model the new Virus and Gene entities while keeping the BC5CDR Chemical and Disease performance intact.

| Config | Backbone | Strategy | What changes |
|---|---|---|---|
| 4 | PubMedBERT | sequential | Phase 1 trains on BC5CDR, phase 2 continues on the silver corpus |
| 5 | PubMedBERT | joint uniform | Gold and silver are concatenated, every example has weight 1.0 |
| 6 | PubMedBERT | joint noise-aware | Gold and silver are concatenated, silver examples carry a smaller weight (0.5) |

**Config 5 (joint uniform) is the main contribution.** Mixing BC5CDR gold and PubMed silver in a single shuffled stream, with the same loss weight on every example, produces the strongest 4-entity model: BC5CDR F1 around 0.83 (Chemical, Disease) and PubMed F1 around 0.97 (Virus, Gene), with all three seeds converging tightly.

Config 4 is reported as the contrast. Sequential fine-tuning learns Virus and Gene from the silver corpus very well, but phase 2 trains on data where Chemical and Disease are rare, so the model overwrites what it learned in phase 1. The BC5CDR F1 collapses almost to zero. This is the textbook signature of catastrophic forgetting, and we surface it on purpose because it motivates the joint approach.

Config 6 is the noise-aware ablation. Downweighting silver examples is the obvious move when you suspect distant-supervision labels are noisy. In practice it underperforms config 5 on both test sets, and one seed out of three collapses to predicting all-O even after raising the silver weight from 0.3 to 0.5. The takeaway is that for this corpus the silver labels are clean enough that downweighting them removes useful signal without buying back meaningful noise reduction. We report the negative result in the paper rather than hiding it.

### Optional hyperparameter sweep

`notebooks/run_experiments_hyperparameter.ipynb` runs three subsampled-silver variants of config 5 (silver:gold ratios 1:1, 2:1, 4:1) and plots an F1 scaling curve against the full-silver 8:1 baseline. Useful when you want to know how much of the silver corpus is actually pulling its weight.

## Dataset

Five JSONL files drive the whole pipeline.

| File | Sentences | Source | Labels |
|---|---|---|---|
| `dataset/bc5cdr/bc5cdr_train.jsonl` | 5,119 | BC5CDR (BioCreative V CDR) | gold Chemical + Disease |
| `dataset/bc5cdr/bc5cdr_validation.jsonl` | 5,218 | BC5CDR | gold Chemical + Disease |
| `dataset/bc5cdr/bc5cdr_test.jsonl` | 5,728 | BC5CDR | gold Chemical + Disease |
| `dataset/pubmed/pubmed_scrapping.jsonl` | 40,946 | PubMed via NCBI Entrez API | silver Virus + Gene (distant supervision) |
| `dataset/pubmed/pubmed_test.jsonl` | 100 | PubMed | manually verified Virus + Gene |

The silver labels come from a dictionary-based distant-supervision pass: every token of the scraped PubMed text is matched against NCBI Taxonomy for virus names and HGNC for human gene symbols. This is cheap, scales to tens of thousands of sentences, and is noisy in the ways distant supervision is always noisy (partial matches, boundary errors, missed mentions). The 100-sentence PubMed test set was manually verified so that Virus and Gene F1 can be reported against a trusted reference.

Each record looks like this:

```json
{
  "tokens": ["The", "SARS-CoV-2", "spike", "binds", "to", "ACE2"],
  "decoded_tags": ["O", "O", "O", "O", "O", "O"],
  "decoded_tags_pseudo_final": ["O", "B-Virus", "O", "O", "O", "B-Gene"]
}
```

`decoded_tags` carries the 5-tag (Chemical + Disease) view used by configs 1 through 3. `decoded_tags_pseudo_final` carries the 9-tag (Chemical + Disease + Virus + Gene) view used by configs 4 through 6. Both label spaces are defined in `src/config.py` as `LABEL2ID_5TAG` and `LABEL2ID_9TAG`.

Both the dataset and every trained checkpoint live on the HuggingFace Hub:

- Dataset: [`lumicero/BioNER-DS`](https://huggingface.co/datasets/lumicero/BioNER-DS)
- Checkpoints: [`lumicero/BioNER-DS`](https://huggingface.co/lumicero/BioNER-DS)

## Running the experiments

### Requirements

- Python 3.12 to 3.14 (this code was created using Python version 3.14)
- A CUDA-capable GPU (the reference runs used an RTX 5060 Laptop)
- The packages in `requirements.txt`

Install PyTorch with the CUDA build that matches your driver, then install the rest:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 torch
pip install -r requirements.txt
```

A CPU-only install will run, but real training on the full dataset is not realistic without a GPU.

### Dataset layout

Download the five JSONL files from the HuggingFace dataset repo and place them under `dataset/`:

```
dataset/
├── bc5cdr/
│   ├── bc5cdr_train.jsonl
│   ├── bc5cdr_validation.jsonl
│   └── bc5cdr_test.jsonl
└── pubmed/
    ├── pubmed_scrapping.jsonl
    └── pubmed_test.jsonl
```

The YAML configs reference these paths relative to the project root, so launching from the repository root (or from `notebooks/` with the auto-cd logic in the notebook setup cell) is enough.

### Running configs 1 through 6

The recommended entry point is `notebooks/run_experiments.ipynb`. It calls the trainer functions directly inside the kernel: no shell crossing, no subprocess, no environment confusion. Open the notebook, pick your seed list in cell 4, then run cells 5 through 10 in order. Each cell handles one configuration across every seed and writes the aggregated mean ± std file when it finishes.

For a headless or SSH workflow, the same thing from the shell:

```bash
for cfg in configs/config_{1,2,3,4,5,6}_*.yaml; do
    for seed in 42 1337 2024; do
        python train.py --config "$cfg" --seed "$seed"
    done
    python -m src.utils.aggregate --config "$cfg"
done
```

If you only want one configuration:

```bash
python train.py --config configs/config_5_joint_uniform.yaml --seed 42
python -m src.utils.aggregate --config configs/config_5_joint_uniform.yaml
```

CLI overrides take precedence over the YAML defaults:

```
--seed INT
--output_dir STR
--epochs INT
--batch_size INT
--learning_rate FLOAT
```

### Re-evaluating an existing checkpoint

```bash
python evaluate.py \
    --checkpoint outputs/checkpoints/config_5_joint_uniform/seed_42/best_model \
    --config configs/config_5_joint_uniform.yaml \
    --test_set test_pubmed
```

### Running inference for downstream use

`predict.py` reads either a JSONL file with pre-tokenized `tokens` or a plain text file (one sentence per line) and writes predictions in the format the downstream Knowledge Graph step consumes:

```bash
python predict.py \
    --checkpoint outputs/checkpoints/config_5_joint_uniform/seed_42/best_model \
    --input dataset/pubmed/pubmed_test.jsonl \
    --output outputs/predictions/config5_pubmed.jsonl
```

Each output line looks like:

```json
{
  "sentence_id": "0",
  "tokens": ["The", "SARS-CoV-2", "spike", "binds", "to", "ACE2"],
  "predictions": [
    {"entity": "SARS-CoV-2", "type": "Virus", "start_token": 1, "end_token": 1, "score": 0.94},
    {"entity": "ACE2", "type": "Gene", "start_token": 5, "end_token": 5, "score": 0.89}
  ]
}
```

## Output layout

```
outputs/
├── checkpoints/<config_name>/seed_<seed>/
│   ├── best_model/                   # single source and joint
│   ├── phase1/best_model/            # sequential phase 1
│   └── phase2/best_model/            # sequential phase 2 (the final model)
├── logs/<config_name>/seed_<seed>_*.log
└── results/<config_name>/
    ├── seed_<seed>/
    │   ├── config_snapshot.yaml
    │   ├── eval_test_bc5cdr.json
    │   └── eval_test_pubmed.json     # configs 4 through 6 only
    ├── aggregated_results.json
    └── aggregated_results.md
```

The aggregator writes both a machine-readable JSON (every metric, every seed, raw values plus mean and std) and a paper-ready Markdown table. Entities with fewer than ten gold spans in a test set are marked N/A in the Markdown so that small-support noise does not contaminate the paper view; the raw zero is preserved in the JSON for auditing. `config_snapshot.yaml` is a frozen copy of the resolved YAML for each run, which makes a result file reproducible even after the config files in `configs/` are edited.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.
