"""Zero-shot and few-shot classification of the test split with two LLMs through OpenRouter."""

import csv
import os
import time

import openai
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from src.data import FEWSHOT_SIZES, LABEL_NAMES, REPO_ROOT, load_split
from src.evaluation import INVALID, save_confusion_matrix, summarize

# Name used in file names and tables -> OpenRouter model ID.
MODELS = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
}
SHOTS = (0, *FEWSHOT_SIZES)  # examples in the prompt: data/prompts/<shots>_shot.txt

OPENROUTER_URL = "https://openrouter.ai/api/v1"
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
RESULTS_DIR = REPO_ROOT / "results" / "llm"

MAX_ATTEMPTS = 5
# Errors worth retrying: network problems and timeouts, rate limits, server-side failures.
RETRYABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)

# Columns of each predictions file, one row per test article.
COLUMNS = ["id", "true_label", "predicted_label", "answer", "latency_s", "cost_usd",
           "input_tokens", "output_tokens", "provider", "model_version"]


def ask(client: openai.OpenAI, model: str, prompt: str) -> dict:
    """Send one prompt and return the answer, latency, tokens, cost, provider, and model that answered.

    Temperature 0 and at most 10 output tokens, since the answer should be one topic name. A failed call
    is retried with a growing wait; after the last attempt the error is raised, so a run stops instead of
    recording an API failure as a wrong answer. latency_s covers the successful attempt only.
    """
    client = client.with_options(max_retries=0)  # retries happen here, so latency excludes failed attempts
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=10)
            latency_s = time.perf_counter() - start
            break
        except RETRYABLE as error:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = 2 ** attempt
            print(f"  attempt {attempt} failed ({error}), retrying in {wait} s ...")
            time.sleep(wait)
    return {
        "answer": response.choices[0].message.content or "",
        "latency_s": latency_s,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "cost_usd": getattr(response.usage, "cost", None),  # billed by OpenRouter, in USD credits
        "provider": getattr(response, "provider", None),    # not documented by OpenRouter; empty if absent
        "model_version": response.model,
    }


def parse(answer: str) -> str:
    """Return the topic named in the answer, or INVALID if it names none.

    Case is ignored, and if several topics appear the earliest one wins, so "sports." gives Sports and
    "Business, then World" gives Business. Other wordings such as "Technology" are not mapped to a topic.
    """
    text = answer.lower()
    found = [(text.find(name.lower()), name) for name in LABEL_NAMES if name.lower() in text]
    return min(found)[1] if found else INVALID


def run_configuration(client: openai.OpenAI, name: str, shots: int, test: pd.DataFrame) -> pd.DataFrame:
    """Classify every test article with one model and one prompt, and return all its predictions.

    Each answer is appended to results/llm/test_predictions_<name>_<shots>_shot.csv as soon as it
    arrives. Articles already in that file are skipped, so a finished configuration is not run again and
    an interrupted one continues where it stopped.
    """
    path = RESULTS_DIR / f"test_predictions_{name}_{shots}_shot.csv"
    template = (PROMPTS_DIR / f"{shots}_shot.txt").read_text(encoding="utf-8").strip()
    if template.count("{article}") != 1:
        raise ValueError(f"{shots}_shot.txt must contain {{article}} exactly once.")

    started = path.exists() and path.stat().st_size > 0
    done = set(pd.read_csv(path).id) if started else set()
    todo = test[~test.id.isin(done)]
    label = f"{name}, {shots} examples"
    if todo.empty:
        print(f"{label}: all {len(test)} articles already done, skipping.")
    else:
        print(f"{label}: classifying {len(todo)} articles" + (f" ({len(done)} done earlier)" if done else ""))
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            if not started:
                writer.writeheader()
            for i, row in enumerate(todo.itertuples(), 1):
                reply = ask(client, MODELS[name], template.replace("{article}", row.text))
                writer.writerow({"id": row.id, "true_label": LABEL_NAMES[row.label],
                                 "predicted_label": parse(reply["answer"]), **reply})
                file.flush()
                if i % 50 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")
    return pd.read_csv(path)


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("No OpenRouter key found. Copy .env.example to .env in the repository root and "
                         "put your key after OPENROUTER_API_KEY=.")
    client = openai.OpenAI(base_url=OPENROUTER_URL, api_key=api_key, timeout=60)
    test = load_split("test")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in MODELS:
        for shots in SHOTS:
            predictions = run_configuration(client, name, shots, test)
            row = {"model": name, "labeled_examples": shots, **summarize(predictions)}
            rows.append(row)
            cm = confusion_matrix(predictions.true_label, predictions.predicted_label, labels=LABEL_NAMES)
            save_confusion_matrix(cm, RESULTS_DIR / f"confusion_matrix_test_{name}_{shots}_shot.png",
                                  f"{name}, {shots} examples, test (n={row['n']}), acc {row['accuracy']:.3f}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "test_results.csv", index=False)
    spread = results.groupby("model", sort=False).accuracy.agg(accuracy_min="min", accuracy_max="max")
    spread["spread_points"] = (spread.accuracy_max - spread.accuracy_min) * 100
    spread.reset_index().to_csv(RESULTS_DIR / "test_accuracy_spread.csv", index=False)

    print("\n== Test results ==")
    print(results.to_string(index=False, float_format="%.4f"))
    print("\n== Accuracy spread across 0, 12, and 48 examples ==")
    print(spread.to_string(float_format="%.3f"))
    print(f"\nWrote CSV/PNG outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
