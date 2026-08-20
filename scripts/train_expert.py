#!/usr/bin/env python3

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.utils import instantiate
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leap import runs
from leap.config import (build_parser, experiment_id, load_config, option, require,
                         resolve_device)
from leap.data import CytologyDataset
from leap.determinism import seed_everything, worker_init_fn
from leap.metrics import best_threshold, classification_metrics, early_stopping_score
from leap.mil import build_model
from leap.splits import ensure_split, load_fold_indices


def get_datasets_for_fold(cfg, fold):
    """Train, validation and test datasets for one fold of the frozen split."""
    labels_df = pd.read_excel(cfg.data.label_file)
    split_json = ensure_split(
        cfg.data.split_json,
        cfg.data.label_file,
        cfg.data.label_column,
        n_folds=int(cfg.data.folds),
        seed=int(cfg.seed),
    )
    train_idx, val_idx, test_idx = load_fold_indices(split_json, fold, labels_df)

    def make(index_list, deterministic):
        return CytologyDataset(
            label_file=cfg.data.label_file,
            image_folder=cfg.data.image_folder,
            label_column=cfg.data.label_column,
            patches_per_slide=int(cfg.data.patches_per_slide),
            augment=bool(cfg.data.augment),
            index_list=index_list,
            base_seed=int(cfg.data.base_seed),
            deterministic=deterministic,
        )

    # Training re-draws patches every epoch; evaluation uses the per-slide seeded draw so
    # metrics are reproducible.
    return make(train_idx, False), make(val_idx, True), make(test_idx, True)


def train_one_epoch(cfg, model, loader, optimizer, criterion, device, epoch, logger):
    """One training pass with gradient accumulation. Returns (labels, probs, mean loss)."""
    accumulation_steps = max(
        1, int(cfg.data.accumulated_batch_size) // int(cfg.data.batch_size)
    )
    grad_clip = float(option(cfg, "train.grad_clip_norm", 1.0))

    model.train()
    optimizer.zero_grad()
    labels_all, probs_all, losses = [], [], []
    last_index = -1

    bar = tqdm(loader, total=len(loader), desc=f"train epoch {epoch}")
    for i, (images, labels) in enumerate(bar):
        last_index = i
        if len(images) == 0:
            continue
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels.unsqueeze(1)) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item() * accumulation_steps)
        labels_all.append(labels.detach().cpu().numpy())
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        logger.log_scalars({"train/loss": losses[-1]})

    # A trailing step taken when the batch count is not a whole number of accumulation
    # cycles. The gradients were already applied and cleared inside the loop, so this steps
    # on zero gradients; the optimizer's moment estimates still move the parameters. It is
    # kept because it ran in every published training run and the released checkpoints
    # reflect it.
    if last_index >= 0 and (last_index + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return (
        np.concatenate(labels_all).ravel(),
        np.concatenate(probs_all)[:, 0],
        float(np.mean(losses)) if losses else float("nan"),
    )


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, desc="eval"):
    """Run the model over a loader. Returns (labels, probs, mean loss or NaN)."""
    model.eval()
    labels_all, probs_all, losses = [], [], []
    for images, labels in tqdm(loader, total=len(loader), desc=desc, leave=False):
        if len(images) == 0:
            continue
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if criterion is not None:
            losses.append(criterion(logits, labels.unsqueeze(1)).item())
        labels_all.append(labels.cpu().numpy())
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
    return (
        np.concatenate(labels_all).ravel(),
        np.concatenate(probs_all)[:, 0],
        float(np.mean(losses)) if losses else float("nan"),
    )


def round_metrics(metrics, digits=3):
    """Round reported metrics before they are recorded, as the trackers expect."""
    return {k: (v if v is None else round(float(v), digits)) for k, v in metrics.items()}


def report(tag, epoch, loss, metrics, threshold):
    print(
        f"[{tag}] epoch {epoch:3d} | loss {loss:.4f} | AUROC {metrics['roc_auc']:.3f} "
        f"| bACC {metrics['balanced_acc']:.3f} | F1w {metrics['weighted_f1']:.3f} "
        f"| MCC {metrics['mcc']:.3f} | thr {threshold:.3f}"
    )


def save_checkpoint(model, path, device):
    """Write the state dict from CPU tensors, so the file loads without a GPU."""
    torch.save(model.cpu().state_dict(), path)
    model.to(device)


def external_datasets(cfg):
    """name -> CytologyDataset for each configured external cohort."""
    datasets = {}
    for spec in option(cfg, "data.external_cohorts", []):
        if not (os.path.exists(spec.label_file) and os.path.isdir(spec.image_folder)):
            print(f"[{spec.name}] skipped: {spec.label_file} or {spec.image_folder} not found")
            continue
        datasets[spec.name] = CytologyDataset(
            label_file=spec.label_file,
            image_folder=spec.image_folder,
            label_column=cfg.data.label_column,
            patches_per_slide=int(cfg.data.patches_per_slide),
            augment=bool(cfg.data.augment),
            base_seed=int(cfg.data.base_seed),
            deterministic=True,
        )
    return datasets


def train_k_fold(cfg, run_id, device):
    results_dir = cfg.output.results_dir
    cohort = cfg.data.cohort
    val_gate = option(cfg, "train.val_gate")

    logger = runs.TensorBoardLogger(option(cfg, "output.tensorboard_dir"), run_id)
    logger.log_config(cfg)
    runs.dump_config(cfg, results_dir, cohort, run_id)
    timestamp = runs.run_timestamp()

    externals = external_datasets(cfg)
    fold_metrics = {c: defaultdict(list) for c in [cohort] + list(externals)}

    for fold in range(int(cfg.data.folds)):
        print(f"\n{'=' * 70}\nFOLD {fold}\n{'=' * 70}")
        logger.set_fold(fold)

        train_ds, val_ds, test_ds = get_datasets_for_fold(cfg, fold)
        print(f"  slides: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
        print(f"  class counts (train): {train_ds.class_counts()}")

        # Per-epoch patch variation comes from workers re-seeding off this generator, so
        # keep num_workers fixed across runs that must reproduce each other.
        generator = torch.Generator()
        generator.manual_seed(int(cfg.seed))
        sampler = WeightedRandomSampler(
            train_ds.weights, len(train_ds.weights), replacement=True
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=int(cfg.data.batch_size),
            sampler=sampler,
            num_workers=int(cfg.data.num_workers),
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
        val_dl = DataLoader(val_ds, batch_size=int(cfg.data.batch_size), shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=int(cfg.data.batch_size), shuffle=False)

        model = build_model(cfg, device)
        optimizer = instantiate(
            cfg.optimizer,
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.optimizer.lr,
        )
        criterion = nn.BCEWithLogitsLoss()
        ckpt = runs.checkpoint_path(cfg.output.checkpoint_dir, run_id, fold)
        os.makedirs(cfg.output.checkpoint_dir, exist_ok=True)

        best_score, patience_counter, saved = -np.inf, 0, False
        for epoch in range(int(cfg.train.epochs)):
            y, p, loss = train_one_epoch(
                cfg, model, train_dl, optimizer, criterion, device, epoch, logger
            )
            thr = best_threshold(p, y)
            train_metrics = classification_metrics(y, p, thr)
            report("train", epoch, loss, train_metrics, thr)
            logger.log_scalars({f"train/{k}": v for k, v in train_metrics.items()})

            y, p, loss = evaluate(model, val_dl, device, criterion, desc=f"val epoch {epoch}")
            thr = best_threshold(p, y)
            val_metrics = classification_metrics(y, p, thr)
            report(" val ", epoch, loss, val_metrics, thr)
            logger.log_scalars({f"val/{k}": v for k, v in val_metrics.items()})

            if val_gate is not None and all(
                val_metrics[k] >= float(val_gate)
                for k in ("balanced_acc", "auc_pr", "weighted_f1", "roc_auc")
            ):
                save_checkpoint(model, ckpt, device)
                saved = True
                print(f"  validation gate {val_gate} reached; stopping at epoch {epoch + 1}")
                break

            if not option(cfg, "train.early_stopping", True):
                continue
            score = early_stopping_score(val_metrics)
            if np.isfinite(score) and score > best_score:
                best_score, patience_counter = score, 0
                save_checkpoint(model, ckpt, device)
                saved = True
            else:
                patience_counter += 1
                if patience_counter >= int(cfg.train.patience):
                    print(f"  early stopping at epoch {epoch + 1}")
                    break

        if not saved:
            raise SystemExit(
                f"fold {fold}: no epoch produced a finite validation score, nothing saved"
            )
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        print(f"  loaded best checkpoint {os.path.basename(ckpt)}")

        # The operating point comes from validation and is then applied unchanged to the
        # held-out fold and to every external cohort.
        y_val, p_val, _ = evaluate(model, val_dl, device, desc="val (threshold)")
        threshold = best_threshold(p_val, y_val)
        print(f"  operating threshold from validation: {threshold:.4f}")

        y, p, _ = evaluate(model, test_dl, device, desc=f"test fold {fold}")
        metrics = round_metrics(classification_metrics(y, p, threshold))
        print(f"  {cohort} fold {fold}: {metrics}")
        for key, value in metrics.items():
            fold_metrics[cohort][key].append(value)

        runs.save_fold_predictions(
            results_dir, cohort, fold, run_id,
            eval_slide_ids=test_ds.slide_ids, eval_labels=y, eval_probs=p,
            eval_preds=(p >= threshold).astype(int),
            eval_patches=runs.collect_patch_manifest(test_ds, test_ds.slide_ids),
            train_slide_ids=train_ds.slide_ids,
            train_labels=train_ds.labels_df[cfg.data.label_column].tolist(),
            val_slide_ids=val_ds.slide_ids,
            val_labels=val_ds.labels_df[cfg.data.label_column].tolist(),
        )
        runs.save_fold_metrics(results_dir, cohort, fold, run_id, cfg.seed, threshold, metrics)
        runs.append_fold_metrics(
            results_dir, run_id, cohort, fold, cfg.seed, threshold, metrics, timestamp
        )

        for name, dataset in externals.items():
            loader = DataLoader(dataset, batch_size=int(cfg.data.batch_size), shuffle=False)
            y, p, _ = evaluate(model, loader, device, desc=f"{name} fold {fold}")
            metrics = round_metrics(classification_metrics(y, p, threshold))
            print(f"  {name} fold {fold}: {metrics}")
            for key, value in metrics.items():
                fold_metrics[name][key].append(value)
            runs.save_fold_predictions(
                results_dir, name, fold, run_id,
                eval_slide_ids=dataset.slide_ids, eval_labels=y, eval_probs=p,
                eval_preds=(p >= threshold).astype(int),
                eval_patches=runs.collect_patch_manifest(dataset, dataset.slide_ids),
            )
            runs.save_fold_metrics(results_dir, name, fold, run_id, cfg.seed, threshold, metrics)
            runs.append_fold_metrics(
                results_dir, run_id, name, fold, cfg.seed, threshold, metrics, timestamp
            )

    averaged = {
        c: {k: round(sum(v) / len(v), 3) for k, v in m.items() if v}
        for c, m in fold_metrics.items()
    }
    runs.append_experiment_recap(results_dir, [(run_id, c, m) for c, m in averaged.items()])
    logger.close()

    print(f"\n{'=' * 70}")
    for c, m in averaged.items():
        print(f"  {c:14s} AUROC {m.get('roc_auc')}  bACC {m.get('balanced_acc')}")
    print(f"  checkpoints -> {cfg.output.checkpoint_dir}")
    print(f"  predictions -> {results_dir}/<cohort>/{run_id}_fold_<k>.csv")


def main():
    args = build_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    require(
        cfg,
        "seed",
        "data.label_file",
        "data.image_folder",
        "data.label_column",
        "data.patches_per_slide",
        "data.folds",
        "data.cohort",
        "data.split_json",
        "output.checkpoint_dir",
        "output.results_dir",
    )
    seed_everything(int(cfg.seed))
    device = resolve_device(option(cfg, "train.device"))
    run_id = experiment_id(
        cfg,
        str(cfg.extractor._target_).rsplit(".", 1)[-1],
        str(cfg.head._target_).rsplit(".", 1)[-1],
    )
    print(f"experiment {run_id} | device {device}")
    train_k_fold(cfg, run_id, device)


if __name__ == "__main__":
    main()
