import json
import os
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from leap.metrics import METRIC_KEYS

PER_FOLD_TRACKER = "metrics_per_fold.csv"
RECAP_TRACKER = "experiment_metrics.csv"


def run_timestamp() -> str:
    """A single timestamp per process, so every fold and cohort of a run is versioned together."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def dump_config(cfg: DictConfig, results_dir: str, cohort: str, experiment_id: str) -> str:
    """Write the fully-resolved config next to that run's predictions."""
    out_dir = os.path.join(results_dir, cohort)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{experiment_id}_config.yaml")
    with open(path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    return path


def write_table(df: pd.DataFrame, path: str) -> str:
    """Write a DataFrame to Parquet, falling back to CSV if no Parquet engine is available."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        alt = path.replace(".parquet", ".csv")
        df.to_csv(alt, index=False)
        return alt


def collect_patch_manifest(dataset, slide_ids: Sequence[str]) -> Dict[str, List[str]]:
    """Slide_ID -> the patch filenames actually scored.

    Valid only for datasets iterated with num_workers=0, so the record was populated in this
    process. The inference draw is seeded per (base_seed, Slide_ID) and worker-independent,
    so the manifest reproduces exactly what was scored.
    """
    return {sid: list(dataset.get_selected_files(sid)) for sid in slide_ids}


def _label_only_rows(slide_ids, labels, split: str) -> List[dict]:
    return [
        {
            "Slide_ID": sid,
            "Labels": int(label),
            "split": split,
            "Probabilities": "",
            "Final Predictions": "",
            "patches": "",
        }
        for sid, label in zip(slide_ids, labels)
    ]


def save_fold_predictions(
    results_dir: str,
    cohort: str,
    fold: int,
    experiment_id: str,
    eval_slide_ids: Sequence[str],
    eval_labels,
    eval_probs,
    eval_preds,
    eval_patches: Dict[str, List[str]],
    train_slide_ids: Optional[Sequence[str]] = None,
    train_labels: Optional[Sequence] = None,
    val_slide_ids: Optional[Sequence[str]] = None,
    val_labels: Optional[Sequence] = None,
) -> str:
    """One CSV per fold covering the whole cohort fold.

    Rows with split == 'test' carry probabilities, thresholded predictions and the patch
    manifest; train and val rows carry Slide_ID and label only. Row order is train, then
    val, then the evaluated slides in dataset order.
    """
    eval_labels = np.asarray(eval_labels).ravel()
    eval_probs = np.asarray(eval_probs).ravel()
    eval_preds = np.asarray(eval_preds).ravel()

    rows: List[dict] = []
    if train_slide_ids is not None:
        rows += _label_only_rows(train_slide_ids, train_labels, "train")
    if val_slide_ids is not None:
        rows += _label_only_rows(val_slide_ids, val_labels, "val")
    for sid, label, prob, pred in zip(eval_slide_ids, eval_labels, eval_probs, eval_preds):
        rows.append({
            "Slide_ID": sid,
            "Labels": int(label),
            "split": "test",
            "Probabilities": float(prob),
            "Final Predictions": int(pred),
            "patches": json.dumps(eval_patches.get(sid, [])),
        })

    df = pd.DataFrame(rows, columns=[
        "Slide_ID", "Labels", "split", "Probabilities", "Final Predictions", "patches",
    ])
    out_dir = os.path.join(results_dir, cohort)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{experiment_id}_fold_{fold}.csv")
    df.to_csv(path, index=False)
    return path


def save_fold_metrics(
    results_dir: str,
    cohort: str,
    fold: int,
    experiment_id: str,
    seed: int,
    threshold: Optional[float],
    metrics: Dict[str, float],
) -> str:
    """Per-fold metrics JSON, stamped with the seed and the operating threshold."""
    out_dir = os.path.join(results_dir, cohort)
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(metrics)
    payload["threshold"] = float(threshold) if threshold is not None else None
    payload["seed"] = int(seed)
    payload["fold"] = int(fold)
    payload["experiment_id"] = experiment_id
    path = os.path.join(out_dir, f"{experiment_id}_fold_{fold}_metrics.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def append_fold_metrics(
    results_dir: str,
    experiment_id: str,
    cohort: str,
    fold: int,
    seed: int,
    threshold: Optional[float],
    metrics: Dict[str, float],
    timestamp: Optional[str] = None,
) -> None:
    """Append one row per (experiment_id, cohort, fold) to the per-fold tracker."""
    path = os.path.join(results_dir, PER_FOLD_TRACKER)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {
        "experiment_id": experiment_id,
        "cohort": cohort,
        "fold": int(fold),
        "seed": int(seed),
        "threshold": round(float(threshold), 6) if threshold is not None else None,
    }
    row.update({k: metrics.get(k) for k in METRIC_KEYS})
    row["timestamp"] = timestamp or run_timestamp()
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )


def append_experiment_recap(results_dir: str, rows: Sequence) -> None:
    """Append per-run averaged metrics. `rows` is an iterable of (experiment_id, cohort, metrics)."""
    path = os.path.join(results_dir, RECAP_TRACKER)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = []
    for experiment_id, cohort, metrics in rows:
        if not metrics:
            continue
        record = {"experiment_id": experiment_id, "cohort": cohort}
        record.update({k: metrics.get(k) for k in METRIC_KEYS})
        out.append(record)
    if not out:
        return
    pd.DataFrame(out, columns=["experiment_id", "cohort"] + list(METRIC_KEYS)).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )


def _summary_writer(log_dir: str):
    """A TensorBoard writer. Imported lazily so the analysis scripts do not need tensorboard."""
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=log_dir)


class TensorBoardLogger:
    """Scalar logger writing one TensorBoard run per fold. A blank `log_dir` disables it."""

    def __init__(self, log_dir: Optional[str], experiment_id: str):
        self.log_dir = log_dir
        self.experiment_id = experiment_id
        self.step = 0
        self.writer = None
        if log_dir:
            self.writer = _summary_writer(log_dir)

    def log_config(self, cfg) -> None:
        if self.writer is not None:
            self.writer.add_text("config", OmegaConf.to_yaml(cfg, resolve=True))

    def log_scalars(self, values: Dict[str, float]) -> None:
        if self.writer is None:
            return
        for tag, value in values.items():
            if value is not None and np.isfinite(value):
                self.writer.add_scalar(tag, value, self.step)
        self.step += 1

    def set_fold(self, fold: int) -> None:
        if self.writer is None:
            return
        self.writer.close()
        self.writer = _summary_writer(
            os.path.join(self.log_dir, self.experiment_id, f"fold_{fold}")
        )
        self.step = 0

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


CHECKPOINT_SUFFIX = ".pt"


def checkpoint_path(checkpoint_dir: str, experiment_id: str, fold: int) -> str:
    """Where a fold's expert weights are written: <dir>/<experiment_id>_fold<k>.pt."""
    return os.path.join(checkpoint_dir, f"{experiment_id}_fold{fold}{CHECKPOINT_SUFFIX}")


def find_checkpoint(checkpoint_dir: str, stem: str, fold: int) -> str:
    """The most recent checkpoint for one fold whose filename contains `stem`.

    Raises FileNotFoundError if none matches, naming the pattern that was searched.
    """
    pattern = os.path.join(checkpoint_dir, f"*{stem}*_fold{fold}{CHECKPOINT_SUFFIX}")
    candidates = sorted(glob(pattern), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint matching {pattern}")
    return candidates[-1]

