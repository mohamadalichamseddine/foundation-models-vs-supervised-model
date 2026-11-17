"""TF-IDF + logistic regression baseline for AG News."""

import argparse
import itertools
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data import LABEL_NAMES, REPO_ROOT, load_split
from src.evaluation import save_confusion_matrix, summarize

SEED = 42

# Settings searched on the validation split.
GRID = {
    "ngram_range": [(1, 1), (1, 2)],
    "min_df": [1, 2],
    "sublinear_tf": [True, False],
    "C": [0.1, 1.0, 10.0, 100.0],
}


def make_pipeline(ngram_range=(1, 2), min_df=1, sublinear_tf=True, C=10.0) -> Pipeline:
    """Return TF-IDF features followed by L2-regularized multinomial logistic regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=ngram_range,
                                  min_df=min_df, sublinear_tf=sublinear_tf)),
        ("logreg", LogisticRegression(C=C, max_iter=2000, random_state=SEED)),
    ])


def scores(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro")}


def describe_numeric(name: str, values: pd.Series, units: str) -> dict:
    return {
        "Variable": name, "Type": "numeric (int)", "Units": units,
        "n (non-null)": int(values.notna().sum()), "% missing": f"{values.isna().mean():.1%}",
        "Mean / mode": f"{values.mean():.1f}", "Std / count": f"{values.std():.1f}",
        "Min-Max": f"{values.min()}-{values.max()}",
    }


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Target class counts plus the text-derived features the baseline relies on."""
    text = df.text
    counts = df.label.value_counts().sort_index()
    mode = LABEL_NAMES[counts.idxmax()] if counts.nunique() > 1 else "none (balanced)"
    rows = [{
        "Variable": "label (target)", "Type": "categorical, 4 classes", "Units": "class",
        "n (non-null)": int(df.label.notna().sum()), "% missing": f"{df.label.isna().mean():.1%}",
        "Mean / mode": mode,
        "Std / count": "; ".join(f"{LABEL_NAMES[k]} {n:,} ({n / len(df):.0%})" for k, n in counts.items()),
        "Min-Max": f"{df.label.min()}-{df.label.max()}",
    }]
    rows.append({
        "Variable": "text (input)", "Type": "string", "Units": "headline + description",
        "n (non-null)": int(text.notna().sum()), "% missing": f"{(text.isna() | (text.str.strip() == '')).mean():.1%}",
        "Mean / mode": "-", "Std / count": f"{text.nunique():,} unique", "Min-Max": "-",
    })
    rows.append(describe_numeric("text length", text.str.len(), "characters"))
    rows.append(describe_numeric("word count", text.str.split().str.len(), "words"))
    # Distinct TF-IDF terms per article (unigrams, vocabulary fit on the training split only).
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode").fit(df.loc[df.split == "training", "text"])
    nnz = pd.Series(np.diff(vec.transform(text).indptr), index=df.index)
    row = describe_numeric("TF-IDF terms per article", nnz, "non-zero unigram features")
    row["Type"] = f"numeric (int); vocab {len(vec.vocabulary_):,}"
    rows.append(row)
    return pd.DataFrame(rows)


def split_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Size, share, and per-class counts of each split."""
    total = sum(len(df) for df in splits.values())
    return pd.DataFrame([{
        "Split": name, "n": len(df), "%": f"{len(df) / total:.1%}",
        **{c: int((df.label == k).sum()) for k, c in enumerate(LABEL_NAMES)},
    } for name, df in splits.items()])


def grid_search(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        model = make_pipeline(**params).fit(train.text, train.label)
        rows.append({**params, **scores(val.label, model.predict(val.text))})
    # Rank by validation accuracy (primary metric), then macro-F1.
    return pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False, ignore_index=True)


def predict_one_at_a_time(model: Pipeline, test: pd.DataFrame) -> pd.DataFrame:
    """Return one row per test article: true and predicted topic, confidence, latency, and cost.

    Each article is classified and timed on its own, as each LLM call handles one article. confidence is
    the probability of the predicted topic. Prediction runs locally, so cost_usd is 0. The columns match
    the LLM predictions in results/llm/.
    """
    rows = []
    for row in test.itertuples():
        start = time.perf_counter()
        proba = model.predict_proba([row.text])[0]
        latency_s = time.perf_counter() - start
        best = int(proba.argmax())
        rows.append({"id": row.id, "true_label": LABEL_NAMES[row.label],
                     "predicted_label": LABEL_NAMES[model.classes_[best]], "confidence": proba[best],
                     "latency_s": latency_s, "cost_usd": 0.0})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-test", action="store_true", help="also score the held-out test split")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "baseline", help="folder for CSV/PNG outputs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 80)

    splits = {name: load_split(name) for name in ("training", "validation", "test")}
    train, val, test = splits["training"], splits["validation"], splits["test"]
    subset = pd.concat([df.assign(split=name) for name, df in splits.items()], ignore_index=True)

    print("== Summary statistics (4,000-article subset, cleaned text) ==")
    stats = summary_statistics(subset)
    print(stats.to_string(index=False))
    stats.to_csv(args.out / "summary_statistics.csv", index=False)

    print("\n== Splits ==")
    sizes = split_table(splits)
    print(sizes.to_string(index=False))
    sizes.to_csv(args.out / "splits.csv", index=False)

    print("\n== Majority-class reference ==")
    dummy = DummyClassifier(strategy="most_frequent").fit(train.text, train.label)
    ref = scores(val.label, dummy.predict(val.text))
    print(f"validation: accuracy {ref['accuracy']:.3f}, macro-F1 {ref['macro_f1']:.3f}")

    print("\n== Settings selected on validation ==")
    grid = grid_search(train, val)
    grid.to_csv(args.out / "validation_grid.csv", index=False)
    print(grid.head(8).to_string(index=False, float_format="%.3f"))
    best = grid.iloc[0][list(GRID)].to_dict()
    best["min_df"], best["sublinear_tf"] = int(best["min_df"]), bool(best["sublinear_tf"])
    print(f"Selected: {best}")

    print("\n== 5-fold stratified CV on training (selected settings) ==")
    cv = cross_validate(make_pipeline(**best), train.text, train.label,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
                        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"})
    for metric in ("accuracy", "macro_f1"):
        fold = cv[f"test_{metric}"]
        print(f"{metric}: {fold.mean():.3f} +/- {fold.std(ddof=1):.3f}  (folds {np.round(fold, 3).tolist()})")

    start = time.perf_counter()
    model = make_pipeline(**best).fit(train.text, train.label)
    training_time_s = time.perf_counter() - start
    start = time.perf_counter()
    val_pred = model.predict(val.text)
    ms_per_1000 = (time.perf_counter() - start) / len(val) * 1000 * 1000
    val_scores = scores(val.label, val_pred)
    print(f"\n== Validation (trained on 3,000 training articles) ==")
    print(f"accuracy {val_scores['accuracy']:.3f}, macro-F1 {val_scores['macro_f1']:.3f}, "
          f"prediction time {ms_per_1000:.1f} ms per 1,000 articles (CPU, batch)")
    print(classification_report(val.label, val_pred, target_names=LABEL_NAMES, digits=3))
    cm = confusion_matrix(val.label, val_pred)
    print(pd.DataFrame(cm, index=[f"true {c}" for c in LABEL_NAMES], columns=LABEL_NAMES).to_string())
    save_confusion_matrix(cm, args.out / "confusion_matrix_validation.png",
                          f"TF-IDF + LR, validation (n={len(val)}), acc {val_scores['accuracy']:.3f}")

    results = [
        {"model": "Majority class", "config": "most_frequent", "split": "validation", **ref},
        {"model": "TF-IDF + LR", "config": str(best), "split": "5-fold CV on training",
         "accuracy": cv["test_accuracy"].mean(), "accuracy_std": cv["test_accuracy"].std(ddof=1),
         "macro_f1": cv["test_macro_f1"].mean(), "macro_f1_std": cv["test_macro_f1"].std(ddof=1)},
        {"model": "TF-IDF + LR", "config": str(best), "split": "validation", **val_scores},
    ]

    pd.DataFrame(results).to_csv(args.out / "baseline_results.csv", index=False)

    if args.score_test:
        predictions = predict_one_at_a_time(model, test)
        predictions.to_csv(args.out / "test_predictions.csv", index=False)
        result = {"model": "TF-IDF + LR", "labeled_examples": len(train), **summarize(predictions),
                  "training_time_s": training_time_s, "cpu": platform.processor()}
        pd.DataFrame([result]).to_csv(args.out / "test_results.csv", index=False)
        print(f"\n== Test (held out, one article at a time) ==")
        print(f"accuracy {result['accuracy']:.3f}, macro-F1 {result['macro_f1']:.3f}, "
              f"latency mean {result['latency_mean_s'] * 1000:.2f} ms, median {result['latency_median_s'] * 1000:.2f} ms, "
              f"training {training_time_s:.2f} s")
        print(classification_report(predictions.true_label, predictions.predicted_label, labels=LABEL_NAMES, digits=3))
        cm = confusion_matrix(predictions.true_label, predictions.predicted_label, labels=LABEL_NAMES)
        save_confusion_matrix(cm, args.out / "confusion_matrix_test.png",
                              f"TF-IDF + LR, test (n={len(test)}), acc {result['accuracy']:.3f}")

    print(f"\nWrote CSV/PNG outputs to {args.out.resolve()}")


if __name__ == "__main__":
    main()
