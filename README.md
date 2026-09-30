# Zero-shot and few-shot foundation models versus a small supervised model

We compare a small supervised text classifier (TF-IDF + logistic regression) against two LLMs used without any weight updates, on four-class news-topic classification (World, Sports, Business, Sci/Tech). The question is whether LLM performance justifies its extra cost and latency for routing news articles to topic sections.

## What is compared

| System | Configurations |
| --- | --- |
| TF-IDF + regularized logistic regression (scikit-learn) | Settings selected on the validation split |
| GPT-4o-mini (via OpenRouter) | 0-shot, 12-shot, 48-shot |
| Llama 3.3 70B Instruct (via OpenRouter) | 0-shot, 12-shot, 48-shot |

Few-shot examples are drawn only from the training split and are balanced across classes
(0, 3, and 12 examples per class).

**Metrics:** accuracy (primary), macro-F1, cost per 1,000 predictions, mean/median latency, and
for each LLM the max–min accuracy spread across 0/12/48 examples.

## Setup

Requires Python 3.13.

```
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run from the repository root with the environment activated.

```
python -m src.data            # download the dataset into data/raw/
python -m src.prepare_data    # build the splits. Will skip if already built under data/splits/
```

`python -m src.prepare_data` ends by printing the dataset statistics documented in
[docs/data_preparation.md](docs/data_preparation.md).

## Dataset

- **Source:** AG News, Hugging Face [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news)
  (Zhang, Zhao & LeCun, 2015). 120,000 train + 7,600 test rows, balanced across four classes.
- **Terms:** license listed as unknown on Hugging Face; the original corpus page states
  non-commercial use. Used here for non-commercial research only, and the raw data is not committed in the repository.
- **Pinned version:** Hugging Face commit `eb185aade064a813bc0b7f42de02595523103ca4`. The two
  source files (train, test) are checked against their published SHA-256, so every teammate works
  from identical files.
- **Subset:** 4,000 articles, balanced across the four classes: 3,000 training and 500 validation
  from the source train file, 500 test from the source test file. Articles whose normalized text
  appears more than once across both files are skipped. The row IDs are committed in `data/splits/`.
  Details: [docs/data_preparation.md](docs/data_preparation.md).

## Repository structure

```
data/raw/                   Source files, downloaded automatically (not committed)
data/splits/                Row IDs of the fixed training, validation, and test splits
docs/data_preparation.md    How the subset was built: cleaning, duplicates, selection
src/data.py                 Download, verify, clean, and load the data (load_split)
src/prepare_data.py         One-time selection of the subset, and dataset statistics
requirements.txt            Pinned Python dependencies
```
