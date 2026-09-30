"""Download, verify, and load the AG News data used in this project."""

import hashlib
import html
import http.client
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
SPLITS_DIR = REPO_ROOT / "data" / "splits"

# Hugging Face dataset fancyzhx/ag_news, pinned to one commit so everyone downloads the same files.
REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
BASE_URL = f"https://huggingface.co/datasets/fancyzhx/ag_news/resolve/{REVISION}/data"

# File name and SHA-256 of each raw file, as published by Hugging Face for this revision.
RAW_FILES = {
    "train": ("train-00000-of-00001.parquet", "fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee"),
    "test": ("test-00000-of-00001.parquet", "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb"),
}

# Source file each split's row IDs refer to.
SPLIT_SOURCE = {"training": "train", "validation": "train", "test": "test"}

# Class name of each label (0-3), from the dataset card.
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# The few-shot examples are the first rows of the training split.
FEWSHOT_SIZES = (12, 48)


def fetch(url: str, attempts: int = 10) -> bytes:
    """Return the full body at url, retrying because connections to Hugging Face are sometimes reset."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except (OSError, http.client.HTTPException) as error:
            if attempt == attempts:
                raise
            print(f"  attempt {attempt} failed ({error}), retrying ...")
            time.sleep(2)


def download_raw() -> None:
    """Make sure data/raw/ holds both source files with their pinned SHA-256.

    A file that is missing, or present but different, is downloaded again and checked before it is written.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, expected_sha256 in RAW_FILES.values():
        path = RAW_DIR / filename
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256:
                continue
            print(f"{filename} does not match the pinned version, downloading it again ...")
        else:
            print(f"Downloading {filename} ...")
        content = fetch(f"{BASE_URL}/{filename}")
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise RuntimeError(f"The downloaded {filename} does not match the pinned version.")
        path.write_bytes(content)


def load_raw(file: str) -> pd.DataFrame:
    """Return the "train" or "test" source file with columns text and label. The index is the row ID."""
    download_raw()
    return pd.read_parquet(RAW_DIR / RAW_FILES[file][0])


def clean_text(text: str) -> str:
    """Undo AG News formatting leftovers: HTML codes whose "&" became a space, HTML tags, and backslash line breaks."""
    text = re.sub(r" ?(#\d+;)", r"&\1", text)                       # "Web #39;s" -> "Web&#39;s"
    text = re.sub(r" ?(?<![&\w])(quot|amp|lt|gt);", r"&\1;", text)  # ", quot;"   -> ",&quot;"
    text = html.unescape(text)                                     # "&#39;" -> "'", "&lt;" -> "<"
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)                 # "<strong>" -> " "
    text = text.replace("\\", " ")                                 # "Friday\but" -> "Friday but"
    return re.sub(r"\s+", " ", text).strip()


def load_split(split: str) -> pd.DataFrame:
    """Return the "training", "validation", or "test" split with columns id, text (cleaned), label.

    Rows follow the order of data/splits/<split>_data_ids.txt, so the first 12 and 48 rows of the
    training split are the few-shot examples.
    """
    ids = [int(line) for line in (SPLITS_DIR / f"{split}_data_ids.txt").read_text().split()]
    df = load_raw(SPLIT_SOURCE[split]).loc[ids].rename_axis("id").reset_index()
    df["text"] = df.text.map(clean_text)
    return df


if __name__ == "__main__":
    download_raw()
    print(f"Source files in {RAW_DIR} match the pinned version.")
