"""Put the baseline and LLM test results side by side: one table with accuracy intervals, per-topic F1
scores, a paired test of each LLM configuration against the baseline, and an accuracy chart."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import LABEL_NAMES, REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"  # the comparison outputs go here, next to baseline/ and llm/
BASELINE_DIR = RESULTS_DIR / "baseline"
LLM_DIR = RESULTS_DIR / "llm"

BASELINE = "TF-IDF + LR"  # model name the baseline script writes

# Columns both test_results.csv files share, in the order of the comparison table.
COLUMNS = ["model", "labeled_examples", "n", "accuracy", "macro_f1", "invalid",
           "cost_per_1000_usd", "latency_mean_s", "latency_median_s"]

Z_95 = 1.96  # standard normal quantile for a 95% interval

# Chart: the two LLMs take the first two categorical hues, the baseline is the gray reference.
DISPLAY_NAMES = {BASELINE: "TF-IDF + LR", "gpt-4o-mini": "GPT-4o-mini", "llama-3.3-70b": "Llama 3.3 70B"}
COLORS = {BASELINE: "#52514e", "gpt-4o-mini": "#2a78d6", "llama-3.3-70b": "#eb6834"}
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def combine_results() -> pd.DataFrame:
    """Return one row per configuration: the baseline first, then each LLM with 0, 12, and 48 examples.

    labeled_examples is how many labeled articles a configuration sees: the training split for the
    baseline, the prompt examples for an LLM.
    """
    tables = []
    for path, command in [(BASELINE_DIR / "test_results.csv", "python -m src.baseline_tfidf_logreg --score-test"),
                          (LLM_DIR / "test_results.csv", "python -m src.llm_classify")]:
        if not path.exists():
            raise SystemExit(f"{path.relative_to(REPO_ROOT)} is missing. Run {command} first.")
        tables.append(pd.read_csv(path))
    return pd.concat(tables, ignore_index=True)[COLUMNS]


def add_accuracy_interval(results: pd.DataFrame) -> pd.DataFrame:
    """Return results with the 95% Wilson interval of each accuracy, in columns placed after accuracy.

    The interval shows how much an accuracy measured on n test articles could move by chance, so
    configurations whose intervals overlap are not clearly different. Wilson rather than the simpler
    accuracy +/- 1.96 standard errors, because it stays within 0-1 and is accurate near either end.
    """
    p, n = results.accuracy, results.n
    center = (p + Z_95**2 / (2 * n)) / (1 + Z_95**2 / n)
    half = Z_95 * np.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / (1 + Z_95**2 / n)
    results = results.copy()
    i = results.columns.get_loc("accuracy") + 1
    results.insert(i, "accuracy_ci_low", center - half)
    results.insert(i + 1, "accuracy_ci_high", center + half)
    return results


def load_predictions(model: str, labeled_examples: int) -> pd.DataFrame:
    """Return one configuration's per-article test predictions, indexed by article ID.

    labeled_examples picks the prompt of an LLM; the baseline has a single predictions file.
    """
    path = (BASELINE_DIR / "test_predictions.csv" if model == BASELINE
            else LLM_DIR / f"test_predictions_{model}_{labeled_examples}_shot.csv")
    return pd.read_csv(path, index_col="id")


def per_class_f1(results: pd.DataFrame) -> pd.DataFrame:
    """Return the F1 score of each topic for every configuration in results, in the same row order.

    Scores come from the per-article prediction files the baseline and LLM runs wrote. As in macro-F1,
    an INVALID answer counts as a miss for its true topic.
    """
    rows = []
    for model, shots in zip(results.model, results.labeled_examples):
        predictions = load_predictions(model, shots)
        f1 = f1_score(predictions.true_label, predictions.predicted_label, labels=LABEL_NAMES,
                      average=None, zero_division=0)
        rows.append({"model": model, "labeled_examples": shots, **dict(zip(LABEL_NAMES, f1))})
    return pd.DataFrame(rows)


def compare_with_baseline(results: pd.DataFrame) -> pd.DataFrame:
    """Return, for each LLM configuration, a paired comparison with the baseline on the same test articles.

    Only articles exactly one of the two got right can show which is better: only_llm_correct and
    only_baseline_correct count them. p_value is McNemar's exact two-sided test: the chance of a split
    at least this uneven if the two were equally accurate. It is paired, so it can detect differences
    that the overlapping accuracy intervals cannot.
    """
    baseline = load_predictions(BASELINE, 0)
    baseline_correct = baseline.true_label == baseline.predicted_label
    rows = []
    for model, shots in zip(results.model, results.labeled_examples):
        if model == BASELINE:
            continue
        llm = load_predictions(model, shots).loc[baseline.index]  # same articles, same order
        llm_correct = llm.true_label == llm.predicted_label
        only_llm, only_baseline = int((llm_correct & ~baseline_correct).sum()), int((~llm_correct & baseline_correct).sum())
        # Under equal accuracy, each disagreement goes either way with probability 1/2 (binomial test).
        n = only_llm + only_baseline
        p_value = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_llm, only_baseline) + 1)) / 2**n)
        rows.append({"model": model, "labeled_examples": shots,
                     "points_above_baseline": (llm_correct.mean() - baseline_correct.mean()) * 100,
                     "only_llm_correct": only_llm, "only_baseline_correct": only_baseline, "p_value": p_value})
    return pd.DataFrame(rows)


def plot_accuracy(results: pd.DataFrame, path: Path) -> None:
    """Save a dot plot of each configuration's test accuracy with its 95% interval.

    Rows follow the table order, with a gap between models. The dashed line marks the baseline's
    accuracy, so each LLM configuration reads against it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group = (results.model != results.model.shift()).cumsum() - 1  # 0 for the first model, 1 for the next, ...
    y = results.index + 0.6 * group
    labels = [f"{DISPLAY_NAMES[m]} (trained on {k:,})" if m == BASELINE else f"{DISPLAY_NAMES[m]}, {k} examples"
              for m, k in zip(results.model, results.labeled_examples)]
    baseline_accuracy = results.accuracy[results.model == BASELINE].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvline(baseline_accuracy, color=MUTED, linestyle=(0, (4, 3)), linewidth=1, zorder=1)
    ax.text(baseline_accuracy, y.min() - 0.75, " baseline", color=MUTED, fontsize=8, va="center")
    for yi, row in zip(y, results.itertuples()):
        color = COLORS[row.model]
        ax.plot([row.accuracy_ci_low, row.accuracy_ci_high], [yi, yi], color=color, linewidth=2,
                solid_capstyle="round", zorder=2)
        ax.plot(row.accuracy, yi, "o", color=color, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3)

    ax.set_yticks(y, labels)
    ax.set_ylim(y.max() + 0.6, y.min() - 1.1)  # top to bottom in table order, room for the line label
    ax.set_xlabel("Test accuracy: dot = measured, line = 95% interval", color=INK, fontsize=9)
    ax.set_title(f"Accuracy on the same {results.n.iloc[0]} test articles", loc="left", color=INK, fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0, colors=INK, labelsize=9)
    ax.tick_params(axis="x", colors=MUTED, labelcolor=INK, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    results = add_accuracy_interval(combine_results())
    per_class = per_class_f1(results)
    vs_baseline = compare_with_baseline(results)

    results.to_csv(RESULTS_DIR / "test_comparison.csv", index=False)
    per_class.to_csv(RESULTS_DIR / "test_per_class_f1.csv", index=False)
    vs_baseline.to_csv(RESULTS_DIR / "test_vs_baseline.csv", index=False)
    plot_accuracy(results, RESULTS_DIR / "test_accuracy.png")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("== All configurations on the test split ==")
    print(results.to_string(index=False, float_format="%.4f"))
    print("\n== F1 per topic ==")
    print(per_class.to_string(index=False, float_format="%.3f"))
    print("\n== Each LLM configuration against the baseline (same articles, McNemar's exact test) ==")
    print(vs_baseline.to_string(index=False, float_format="%.4f"))
    print(f"\nWrote test_comparison.csv, test_per_class_f1.csv, test_vs_baseline.csv, and test_accuracy.png "
          f"to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
