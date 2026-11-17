# Zero-shot and few-shot LLMs versus a small supervised model

Can two LLMs, used without any training, beat a simple TF-IDF + logistic regression classifier at sorting
news articles into four topics (World, Sports, Business, Sci/Tech)? And is the difference worth the extra
cost and latency?

## What is compared

- **TF-IDF + logistic regression**, trained on 3,000 labeled articles.
- **GPT-4o-mini** and **Llama 3.3 70B Instruct** through OpenRouter, each with 0, 12, and 48 examples in the prompt.

All seven are scored on the same 500 test articles: accuracy, macro-F1, cost per 1,000 predictions, and
mean/median latency. For each LLM we also report how much accuracy changes between 0, 12, and 48 examples.

## Setup

Requires Python 3.13.

```
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # macOS/Linux: cp .env.example .env
```

Then put your [OpenRouter](https://openrouter.ai) API key in `.env`. Git ignores this file.

## How to run

From the repository root, with the environment activated:

```
python -m src.baseline_tfidf_logreg --score-test   # logistic regression, results in results/baseline/
python -m src.llm_classify                         # both LLMs, results in results/llm/
```

The dataset downloads automatically on the first run. Each command writes one file of predictions per
model (one row per test article) and a `test_results.csv` with the scores. The LLM run makes 3,000 API
calls; if it stops, run it again and it continues where it left off.

## Data

- **AG News** from Hugging Face ([`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news),
  Zhang, Zhao & LeCun, 2015), pinned to one version and checked by SHA-256. Used for non-commercial
  research only, as the original corpus requires.
- **A fixed subset** of 3,000 training, 500 validation, and 500 test articles, balanced across the four
  topics, with duplicate articles removed. The row IDs are in `data/splits/`.
- **The prompts** are in `data/prompts/`. Their examples are the first 12 and 48 training articles, the
  only article text stored in the repository.

How the subset was built: [docs/data_preparation.md](docs/data_preparation.md).

## Repository structure

```
data/splits/     Row IDs of the training, validation, and test articles
data/prompts/    LLM prompts with 0, 12, and 48 examples
src/             Code: data loading, baseline, LLM runs, shared metrics
results/         Outputs of the runs (results/llm/old/ holds an earlier pilot)
docs/            How the data subset was built
```
