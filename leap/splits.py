import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def build_splits(label_file, label_column, n_folds=5, seed=42, val_frac=0.25):
    """fold -> {'train': [Slide_ID], 'val': [...], 'test': [...]}.

    Stratified k-fold on `label_column`, with a stratified inner split carving `val_frac`
    of each fold's training rows off as validation.
    """
    df = pd.read_excel(label_file)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = {}
    for fold, (train_idx, test_idx) in enumerate(skf.split(df["Slide_ID"], df[label_column])):
        tr, va = train_test_split(
            train_idx,
            test_size=val_frac,
            stratify=df.iloc[train_idx][label_column],
            random_state=seed,
        )
        folds[fold] = {
            "train": df.iloc[tr]["Slide_ID"].astype(str).tolist(),
            "val": df.iloc[va]["Slide_ID"].astype(str).tolist(),
            "test": df.iloc[test_idx]["Slide_ID"].astype(str).tolist(),
        }
    return folds


def freeze_splits(label_file, label_column, out_json, n_folds=5, seed=42, val_frac=0.25):
    """Write a split to `out_json` atomically and return its path."""
    folds = build_splits(label_file, label_column, n_folds, seed, val_frac)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    payload = {
        "label_file": os.path.basename(str(label_file)),
        "label_column": label_column,
        "n_folds": n_folds,
        "seed": seed,
        "val_frac": val_frac,
        "folds": folds,
    }
    tmp = f"{out_json}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_json)
    return out_json


def load_split(split_json, fold):
    """{'train': [Slide_ID], 'val': [...], 'test': [...]} for one fold."""
    with open(split_json) as f:
        return json.load(f)["folds"][str(fold)]


def load_fold_indices(split_json, fold, labels_df):
    """Map a frozen fold to positional indices into `labels_df`.

    Returns (train_idx, val_idx, test_idx) as integer arrays. Raises if any Slide_ID in the
    split is missing from the label file, rather than silently dropping it.
    """
    record = load_split(split_json, fold)
    ids = labels_df["Slide_ID"].astype(str).tolist()
    id_to_index = {s: i for i, s in enumerate(ids)}
    if len(id_to_index) != len(ids):
        raise ValueError(
            "Duplicate Slide_IDs in the label file; a frozen split cannot be mapped unambiguously."
        )

    def to_indices(key):
        missing = [s for s in record[key] if s not in id_to_index]
        if missing:
            raise ValueError(
                f"{len(missing)} '{key}' Slide_IDs from {split_json} are absent from the label "
                f"file. The label file does not match this split; refusing to proceed."
            )
        return np.array([id_to_index[s] for s in record[key]])

    return to_indices("train"), to_indices("val"), to_indices("test")


def ensure_split(split_json, label_file, label_column, n_folds=5, seed=42, val_frac=0.25):
    """Return `split_json`, freezing it from the label file first if it does not exist."""
    if not os.path.exists(split_json):
        freeze_splits(label_file, label_column, split_json, n_folds, seed, val_frac)
    return split_json
