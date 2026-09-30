"""Select the fixed 4,000-article subset once and write its row IDs to data/splits/."""

import re

import numpy as np
import pandas as pd

from src.data import FEWSHOT_SIZES, LABEL_NAMES, REVISION, SPLIT_SOURCE, SPLITS_DIR, clean_text, load_raw, load_split

SEED = 42
VALIDATION_PER_CLASS = 125  # from the source train file, taken first
TRAINING_PER_CLASS = 750    # from the source train file, taken next
TEST_PER_CLASS = 125        # from the source test file


def normalize(text: str) -> str:
    """Return the text used to compare articles for duplicates: lowercase, punctuation as spaces, spaces collapsed."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def build_pool() -> pd.DataFrame:
    """Return every article from both source files with columns file, row, label, key, eligible.

    key is the normalized cleaned text. An article is ineligible if its key appears more than once
    across both files; every copy is skipped, including copies with conflicting labels.
    """
    pool = pd.concat(
        [load_raw(file).assign(file=file).rename_axis("row").reset_index() for file in ("train", "test")],
        ignore_index=True,
    )
    pool["key"] = pool.text.map(clean_text).map(normalize)
    pool["eligible"] = ~pool.key.duplicated(keep=False)
    return pool[["file", "row", "label", "key", "eligible"]]


def select_splits(pool: pd.DataFrame) -> dict[str, list[int]]:
    """Return the row IDs of the training, validation, and test splits.

    For each class, the eligible rows of each source file are shuffled with SEED. Validation takes the
    first rows of the train file and training the next ones, so a larger training set later just
    continues down the same shuffled list. Test takes the first rows of the test file. Each split lists
    its IDs with classes interleaved (World, Sports, Business, Sci/Tech, World, ...), so any prefix,
    such as the first 12 or 48 training IDs used as few-shot examples, is balanced.
    """
    rng = np.random.default_rng(SEED)
    per_class = {"training": [], "validation": [], "test": []}
    for label in sorted(pool.label.unique()):
        train_rows = rng.permutation(pool.row[(pool.file == "train") & (pool.label == label) & pool.eligible])
        test_rows = rng.permutation(pool.row[(pool.file == "test") & (pool.label == label) & pool.eligible])
        per_class["validation"].append(train_rows[:VALIDATION_PER_CLASS])
        per_class["training"].append(train_rows[VALIDATION_PER_CLASS:VALIDATION_PER_CLASS + TRAINING_PER_CLASS])
        per_class["test"].append(test_rows[:TEST_PER_CLASS])
    return {split: [int(row) for group in zip(*classes) for row in group] for split, classes in per_class.items()}


def write_splits(splits: dict[str, list[int]]) -> None:
    """Write each split to data/splits/<split>_data_ids.txt, one row ID per line, in order."""
    SPLITS_DIR.mkdir(parents=True)  # raises FileExistsError if the folder exists, so IDs are never overwritten
    for split, ids in splits.items():
        (SPLITS_DIR / f"{split}_data_ids.txt").write_text("".join(f"{row}\n" for row in ids), newline="\n")


def fmt_range(values: pd.Series) -> str:
    """Return "min / median / max" of values."""
    return f"{values.min():,} / {values.median():,.0f} / {values.max():,}"


def fmt_per_class(labels: pd.Series) -> str:
    """Return the number of articles per class, in label order."""
    return " / ".join(f"{n:,}" for n in labels.value_counts().sort_index())


def dataset_statistics(pool: pd.DataFrame) -> None:
    """Print the numbers documented in docs/data_preparation.md, section by section, then the checks."""
    raw = {file: load_raw(file) for file in ("train", "test")}
    cleaned = {file: df.text.map(clean_text) for file, df in raw.items()}
    all_raw, all_cleaned = pd.concat(raw.values()).text, pd.concat(cleaned.values())

    print(f"\n== 1-2. Source dataset (commit {REVISION}) ==")
    print(pd.DataFrame({file: {
        "Rows": f"{len(df):,}",
        "Per class": fmt_per_class(df.label),
        "Characters (min / median / max)": fmt_range(df.text.str.len()),
        "Words (min / median / max)": fmt_range(df.text.str.split().str.len()),
        "Empty texts": (df.text.str.strip() == "").sum(),
        "Exact duplicate texts": df.text.duplicated().sum(),
    } for file, df in raw.items()}).to_string())

    print("\n== 3. Text cleaning (rows) ==")
    leftovers = {
        "HTML code whose & became a space": r"(?<!&)#\d+;",
        "quot;/amp;/lt;/gt; whose & became a space": r"(?<![&\w])(?:quot|amp|lt|gt);",
        "Encoded HTML tags": r"&lt;/?[a-zA-Z]",
        "Backslash (lost line break)": r"\\",
    }
    print(pd.DataFrame({file: {
        **{name: f"{df.text.str.contains(pattern).sum():,}" for name, pattern in leftovers.items()},
        "Changed by clean_text()": f"{(cleaned[file] != df.text).sum():,} ({(cleaned[file] != df.text).mean():.1%})",
    } for file, df in raw.items()}).to_string())
    missing_amp = all_raw.str.count(r"(?<!&)#\d+;|(?<![&\w])(?:quot|amp|lt|gt);").sum()
    after_space = all_raw.str.count(r" (?:#\d+|quot|amp|lt|gt);").sum()
    print(f"Codes missing their &: {missing_amp:,}, preceded by a space: {after_space:,}")
    print(f"Possessives 'Web #39;s': {all_raw.str.count(r'\w #39;s\b').sum():,}, 'Web#39;s': {all_raw.str.count(r'\w#39;s\b').sum():,}")
    remaining = {"#NN;": r"#\d+;", "backslash": r"\\", "HTML tag": r"</?[a-zA-Z][^>]*>", "&name;": r"&\w+;", "space before 's": r"\w 's\b"}
    print("Rows still containing after cleaning:", {name: int(all_cleaned.str.contains(p).sum()) for name, p in remaining.items()})

    print("\n== 4. Duplicates and eligibility ==")
    skipped = pool[~pool.eligible]
    groups = skipped.groupby("key")
    identical = all_cleaned.duplicated(keep=False).sum()
    print(f"Identical after cleaning: {identical:,}, identical only after normalizing: {len(skipped) - identical:,}")
    print(f"Ineligible (skipped): {len(skipped):,}, by file: {skipped.file.value_counts().to_dict()}")
    print(f"Groups: {groups.ngroups:,}, by size: {groups.size().value_counts().sort_index().to_dict()}, "
          f"spanning both files: {(groups.file.nunique() > 1).sum()}, conflicting labels: {(groups.label.nunique() > 1).sum()}")
    print("Eligible per class:")
    print(pool[pool.eligible].pivot_table(index="file", columns="label", values="row", aggfunc="count")
          .rename(columns=dict(enumerate(LABEL_NAMES))).map(lambda n: f"{n:,}").to_string())

    print("\n== 5. Splits ==")
    if not SPLITS_DIR.exists():
        print("Not built yet: run python -m src.prepare_data")
        return
    splits = {split: load_split(split) for split in SPLIT_SOURCE}
    print(f"Seed {SEED}, per class: validation {VALIDATION_PER_CLASS}, training {TRAINING_PER_CLASS}, test {TEST_PER_CLASS}")
    print(pd.DataFrame({split: {
        "Source file": SPLIT_SOURCE[split],
        "Articles": f"{len(df):,}",
        "Per class": fmt_per_class(df.label),
        "Characters (min / median / max)": fmt_range(df.text.str.len()),
        "Words (min / median / max)": fmt_range(df.text.str.split().str.len()),
    } for split, df in splits.items()}).to_string())
    for size in FEWSHOT_SIZES:
        print(f"Few-shot {size} (first {size} training rows), per class: {fmt_per_class(splits['training'].head(size).label)}")

    print("\n== 6. Checks ==")
    expected = {"training": TRAINING_PER_CLASS, "validation": VALIDATION_PER_CLASS, "test": TEST_PER_CLASS}
    ids = {split: df.id.tolist() for split, df in splits.items()}
    selected = pd.concat([pool[pool.file == SPLIT_SOURCE[split]].set_index("row").loc[ids[split]] for split in splits])
    checks = {
        "Split sizes match the settings": all(
            df.label.nunique() == len(LABEL_NAMES) and (df.label.value_counts() == expected[split]).all()
            for split, df in splits.items()),
        "Every selected article is eligible": selected.eligible.all(),
        "No normalized text appears twice across the splits (so no repeated or shared IDs)": selected.key.is_unique,
        "Re-running the selection gives the committed IDs": select_splits(pool) == ids,
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")


if __name__ == "__main__":
    pool = build_pool()
    if SPLITS_DIR.exists():
        print(f"Skipped: {SPLITS_DIR} already exists and the subset is fixed. Delete that folder only to rebuild it on purpose.")
    else:
        write_splits(select_splits(pool))
        print(f"Wrote the training, validation, and test row IDs to {SPLITS_DIR}.")
    dataset_statistics(pool)
