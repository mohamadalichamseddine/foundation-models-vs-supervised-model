# Data preparation

This document describes how the 4,000-article subset used in all experiments was built from AG News.
Only a subset is used, to keep the cost of querying the LLMs manageable. The subset is fixed: its row
IDs are committed in `data/splits/`, and every script loads it with `load_split()` from `src/data.py`.

All numbers below are printed by `python -m src.prepare_data`, section by section, so this document
can be checked or updated from that output.

## 1. Source

| | |
| --- | --- |
| Dataset | AG News, Hugging Face [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) (Zhang, Zhao & LeCun, 2015) |
| Pinned version | Commit `eb185aade064a813bc0b7f42de02595523103ca4` (last modified 2024-03-07) |
| Files | `train-00000-of-00001.parquet` (120,000 rows), `test-00000-of-00001.parquet` (7,600 rows) |
| Columns | `text` (headline followed by description), `label` |
| Labels | 0 World, 1 Sports, 2 Business, 3 Sci/Tech (from the dataset card) |
| Terms | License listed as unknown on Hugging Face; the original corpus page states non-commercial use |

`python -m src.data` downloads both files into `data/raw/` (not committed) and checks each one against
the SHA-256 that Hugging Face publishes for the pinned version:

| File | SHA-256 |
| --- | --- |
| train | `fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee` |
| test | `71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb` |

## 2. Source dataset profile

| | Train file | Test file |
| --- | --- | --- |
| Rows | 120,000 | 7,600 |
| Per class | 30,000 | 1,900 |
| Characters (min / median / max) | 100 / 232 / 1,012 | 100 / 231 / 892 |
| Words (min / median / max) | 8 / 37 / 177 | 11 / 37 / 137 |
| Empty texts | 0 | 0 |
| Exact duplicate texts within the file | 0 | 0 |

## 3. Text cleaning

The text contains formatting leftovers from the original web pages. `clean_text()` in `src/data.py`
repairs them. It runs in memory whenever data is loaded; the raw files are never modified.

| Leftover | Train rows | Test rows | Raw → cleaned |
| --- | --- | --- | --- |
| HTML code whose `&` became a space (`#39;`, `#36;`, `#151;`, …) | 31,357 | 1,991 | `the Web #39;s` → `the Web's`, `past  #36;46` → `past $46` |
| `quot;`, `amp;`, `lt;`, `gt;` whose `&` became a space | 6,181 | 390 | `Sizzler, quot; Aug. 11` → `Sizzler," Aug. 11` |
| Encoded HTML tags | 5,241 | 301 | `&lt;strong&gt;Opinion&lt;/strong&gt;` → `Opinion` |
| Backslash marking a lost line break | 13,146 | 808 | `Friday\but` → `Friday but` |
| **Rows changed by `clean_text()`** | **63,526 (52.9%)** | **3,973 (52.3%)** | |

Rules, in order:

1. Replace a space followed by `#NN;` with `&#NN;`. The space is the lost `&`: 62,005 of the 62,011
   codes missing their `&` are preceded by a space, and possessives always appear as `Web #39;s`,
   never `Web#39;s`.
2. Do the same for `quot;`, `amp;`, `lt;`, and `gt;`.
3. Decode all HTML codes with Python's `html.unescape` (Windows-1252 codes such as `#151;` become `—`).
4. Remove HTML tags, keeping the text inside them.
5. Replace backslashes with spaces and collapse repeated spaces.

After cleaning, no `#NN;` codes, backslashes, or HTML tags remain in either file. Left unchanged:
source tags such as "(Reuters)" and "AP -", which are part of the article text; URLs; 3 rows that
still contain an `&name;` sequence (a double-encoded `&apos;`, the source typo `&nsbp;`, and
`AT &T;`); and 24 rows where the source has a stray space before `'s` (for example `men 's`).

## 4. Duplicates and eligibility

The same article sometimes appears twice with trivial differences, for example `Found in Greece` and
`Found In Greece`, or `KMart` and `Kmart`. If one copy landed in training and the other in test, the
test score would be inflated.

To compare articles, `normalize()` in `src/prepare_data.py` lowercases the cleaned text, replaces
punctuation with spaces, and collapses spaces. **An article is ineligible if its normalized text
appears more than once across both source files, and every copy is skipped.** This also removes
copies with conflicting labels without needing a separate rule.

| | Articles |
| --- | --- |
| Identical after cleaning | 372 |
| Identical only after normalizing | 325 |
| **Ineligible (skipped)** | **697** (663 from the train file, 34 from the test file) |

These form 348 duplicate groups (347 pairs and one triple). 32 groups span both files, and 55 groups
have conflicting labels. The raw files contain no exact duplicates; the copies only become visible
after cleaning and normalizing.

Eligible articles per class:

| | World | Sports | Business | Sci/Tech |
| --- | --- | --- | --- | --- |
| Train file | 29,931 | 29,887 | 29,804 | 29,715 |
| Test file | 1,897 | 1,895 | 1,888 | 1,886 |

## 5. Selection and splits

`python -m src.prepare_data` ran once on 2026-09-29 and wrote the row IDs. It refuses to run again
while `data/splits/` exists.

For each class, the eligible rows of each source file were shuffled with a NumPy generator seeded
with 42. Then:

- **Validation** took the first 125 rows per class of the train file.
- **Training** took the next 750 rows per class of the train file.
- **Test** took the first 125 rows per class of the test file.

Because test comes from a separate file, the training set can grow later by raising
`TRAINING_PER_CLASS`. The selection then continues down the same shuffled lists: the current 3,000
articles stay in, and validation and test do not change.

| File | IDs are rows of | Articles | Per class | Characters (min / median / max) | Words (min / median / max) |
| --- | --- | --- | --- | --- | --- |
| `training_data_ids.txt` | train file | 3,000 | 750 | 77 / 229 / 717 | 11 / 37 / 131 |
| `validation_data_ids.txt` | train file | 500 | 125 | 71 / 224 / 440 | 11 / 37 / 77 |
| `test_data_ids.txt` | test file | 500 | 125 | 101 / 226 / 615 | 17 / 36 / 105 |

An ID is the 0-based row number in its source file. Lengths are measured on the cleaned text.

Each file lists its IDs with classes interleaved (World, Sports, Business, Sci/Tech, World, …), so any
prefix whose length is a multiple of four is balanced. If the test set ever has to shrink, its first
N rows remain balanced.

### Few-shot examples

The few-shot examples are the first 12 and the first 48 rows of the training split: 3 and 12 per
class, with the 12 included in the 48. The project plan called for 10 and 50; neither divides by
four, so they were changed to the nearest balanced counts.

## 6. Checks

`python -m src.prepare_data` ends with these checks, and all pass:

- Split sizes and per-class counts match the settings in `src/prepare_data.py`.
- Every selected article is eligible.
- No normalized text appears twice across the three splits, so no ID repeats within a split and
  no article is shared between splits.
- Re-running the selection produces the committed IDs.
