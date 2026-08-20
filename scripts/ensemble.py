#!/usr/bin/env python3
import os
import re
import sys
from glob import glob

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leap import runs
from leap.config import build_parser, experiment_id, load_config, option, require
from leap.metrics import (COARSE_GRID, auroc, best_threshold, bootstrap_auroc_ci,
                          classification_metrics)

COMBINERS = ["mean", "lr", "lr_convex"]


def read_predictions(path):
    """Slide_ID, Labels and Probabilities for the test rows of one fold CSV."""
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    return df[["Slide_ID", "Labels", "Probabilities"]]


def resolve_best_run(predictions_dir, cohort, stem, folds):
    """The experiment_id of the best complete run matching `stem`, by pooled out-of-fold AUROC.

    Returns (experiment_id or None, AUROC). None means no run covered every fold, and the
    caller falls back to the newest file per fold.
    """
    candidates = {}
    for path in glob(f"{predictions_dir}/{cohort}/*{stem}*_fold_*.csv"):
        match = re.search(r"_fold_(\d+)\.csv$", os.path.basename(path))
        if match:
            eid = os.path.basename(path)[: match.start()]
            candidates.setdefault(eid, {})[int(match.group(1))] = path
    complete = {e: f for e, f in candidates.items() if all(k in f for k in range(folds))}
    if not complete:
        return None, float("nan")
    best, best_auc = None, -1.0
    for eid, per_fold in complete.items():
        pooled = pd.concat([read_predictions(per_fold[k]) for k in range(folds)], ignore_index=True)
        a = auroc(pooled["Labels"], pooled["Probabilities"])
        if np.isfinite(a) and a > best_auc:
            best, best_auc = eid, a
    return best, best_auc


def load_expert_cohort(predictions_dir, cohort, stem, eid, folds):
    """fold -> predictions frame for one Expert on one cohort."""
    out = {}
    for k in range(folds):
        if eid:
            path = os.path.join(predictions_dir, cohort, f"{eid}_fold_{k}.csv")
            matches = [path] if os.path.exists(path) else []
        else:
            matches = sorted(
                glob(f"{predictions_dir}/{cohort}/*{stem}*_fold_{k}.csv"), key=os.path.getmtime
            )
        if matches:
            out[k] = read_predictions(matches[-1])
    return out


def build_cohort(predictions_dir, cohort, folds, experts, stems, chosen_runs):
    """Merge every Expert's per-fold CSVs on Slide_ID -> fold -> frame with one column per Expert."""
    per_expert = {
        name: load_expert_cohort(predictions_dir, cohort, stems[name], chosen_runs.get(name), folds)
        for name in experts
    }
    if any(not per_expert[name] for name in experts):
        return {}
    shared = sorted(set.intersection(*[set(per_expert[n].keys()) for n in experts]))
    out = {}
    for k in shared:
        merged = None
        for name in experts:
            df = per_expert[name][k].rename(columns={"Probabilities": name})
            merged = df if merged is None else merged.merge(
                df.drop(columns=["Labels"]), on="Slide_ID", how="inner"
            )
        if merged is not None and len(merged):
            out[k] = merged.reset_index(drop=True)
    return out


def fit_logistic(X, y, nonneg=False, C=1.0):
    """Logistic weights and intercept; `nonneg` constrains weights to be non-negative."""
    if not nonneg:
        lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
        lr.fit(X, y)
        return lr.coef_.ravel().astype(float), float(lr.intercept_[0])

    n, d = X.shape
    lam = 1.0 / (C * max(n, 1))

    def objective(theta):
        w, b = theta[:d], theta[d]
        z = X @ w + b
        return float(np.mean(np.logaddexp(0.0, z) - y * z) + lam * np.sum(w * w))

    def jacobian(theta):
        w, b = theta[:d], theta[d]
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        return np.concatenate([X.T @ (p - y) / n + 2.0 * lam * w, [float(np.mean(p - y))]])

    result = minimize(
        objective, np.zeros(d + 1), jac=jacobian, method="L-BFGS-B",
        bounds=[(0.0, None)] * d + [(None, None)],
    )
    return result.x[:d].astype(float), float(result.x[d])


def combiner_fit(kind, X, y):
    if kind == "mean":
        return {"kind": "mean"}
    return {"kind": kind, "weights": fit_logistic(X, y, nonneg=(kind == "lr_convex"))}


def combiner_apply(combiner, X):
    if combiner["kind"] == "mean":
        return X.mean(axis=1)
    w, b = combiner["weights"]
    return 1.0 / (1.0 + np.exp(-(X @ np.asarray(w) + b)))


def crossfit_discovery(kind, folds, active):
    """Cross-fit a combiner over the discovery folds; pooled out-of-fold predictions."""
    keys, rows = sorted(folds), []
    for k in keys:
        held = folds[k]
        if kind == "mean":
            score = held[active].to_numpy().mean(axis=1)
        else:
            fitted = pd.concat([folds[j] for j in keys if j != k], ignore_index=True)
            combiner = combiner_fit(
                kind, fitted[active].to_numpy(), fitted["Labels"].to_numpy().astype(int)
            )
            score = combiner_apply(combiner, held[active].to_numpy())
        frame = held[["Slide_ID", "Labels"]].copy()
        frame["score"], frame["fold"] = score, k
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def score_frame(df, column):
    frame = df[["Slide_ID", "Labels"]].copy()
    frame["score"] = np.asarray(df[column], dtype=float)
    return frame


def expert_frames(name, discovery_folds, externals, discovery):
    frames = {discovery: {k: score_frame(discovery_folds[k], name) for k in discovery_folds}}
    for cohort, folds in externals.items():
        frames[cohort] = {k: score_frame(folds[k], name) for k in folds}
    return frames


def combiner_frames(kind, discovery_folds, discovery_pool, externals, discovery, active):
    pooled = crossfit_discovery(kind, discovery_folds, active)
    frames = {
        discovery: {
            k: g[["Slide_ID", "Labels", "score"]].copy() for k, g in pooled.groupby("fold")
        }
    }
    full = combiner_fit(
        kind, discovery_pool[active].to_numpy(), discovery_pool["Labels"].to_numpy().astype(int)
    )
    for cohort, folds in externals.items():
        per_fold = {}
        for k in sorted(folds):
            frame = folds[k][["Slide_ID", "Labels"]].copy()
            frame["score"] = combiner_apply(full, folds[k][active].to_numpy())
            per_fold[k] = frame
        frames[cohort] = per_fold
    return frames, full


def round4(value):
    """Round a reported metric to four decimals, leaving NaN alone."""
    return value if value is None or not np.isfinite(value) else round(float(value), 4)


def mean_of_folds(per_fold):
    values = [auroc(f["Labels"], f["score"]) for f in per_fold.values()]
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def model_metrics(frames, discovery, external_names, rng, n_boot):
    """Metrics for one model: pooled cross-fit out-of-fold on discovery, fold-averaged externally.

    One threshold, chosen on the discovery out-of-fold predictions, is applied to every cohort.
    """
    discovery_df = pd.concat(
        [frames[discovery][k] for k in sorted(frames[discovery])], ignore_index=True
    )
    threshold = best_threshold(
        discovery_df["score"].to_numpy(), discovery_df["Labels"].to_numpy().astype(int),
        COARSE_GRID,
    )

    def pack(df, per_fold, mode):
        y, p = df["Labels"].to_numpy().astype(int), df["score"].to_numpy()
        lo, hi = bootstrap_auroc_ci(y, p, rng, n_boot)
        metrics = {k: round4(v) for k, v in classification_metrics(y, p, threshold).items()}
        return dict(
            metrics, ci_low=round4(lo), ci_high=round4(hi),
            auroc_mean_of_folds=round4(mean_of_folds(per_fold)),
            n=len(df), n_positive=int(y.sum()), mode=mode,
        )

    results = {discovery: pack(discovery_df, frames[discovery], "cv_pool")}
    for cohort in external_names:
        per_fold = frames[cohort]
        stacked = pd.concat(per_fold.values(), ignore_index=True)
        averaged = (
            stacked.groupby("Slide_ID")
            .agg(Labels=("Labels", "first"), score=("score", "mean"))
            .reset_index()
        )
        results[cohort] = pack(averaged, per_fold, "zeroshot_avg")
    return results, threshold


def select_headline(results, discovery, mode):
    """Pick the reported model. Selection reads discovery AUROC only."""
    names = list(results)

    def discovery_auroc(name):
        value = results[name][discovery]["roc_auc"]
        return -1.0 if value is None or not np.isfinite(value) else value

    if mode in results:
        return mode
    if mode in ("expert", "best_expert"):
        return max([n for n in names if n.startswith("expert:")], key=discovery_auroc)
    if mode == "auto":                       # one-standard-error rule
        reference = results["mean"][discovery]
        se = (
            (reference["ci_high"] - reference["ci_low"]) / 3.92
            if np.isfinite(reference["ci_high"]) else 0.0
        )
        best, best_auc = "mean", reference["roc_auc"]
        for kind in ("lr_convex", "lr"):
            value = results[kind][discovery]["roc_auc"]
            if np.isfinite(value) and value > reference["roc_auc"] + se and value > best_auc:
                best, best_auc = kind, value
        return best

    # auto_all: highest discovery AUROC over every model; ties favour mean, then a single
    # Expert, then the convex stacker, then the unconstrained one.
    def preference(name):
        if name == "mean":
            return 0
        if name.startswith("expert:"):
            return 1
        return 2 if name == "lr_convex" else 3

    return max(names, key=lambda n: (discovery_auroc(n), -preference(n)))


def write_fold_csv(directory, fold, frame, threshold):
    os.makedirs(directory, exist_ok=True)
    score = frame["score"].to_numpy()
    pd.DataFrame({
        "Slide_ID": frame["Slide_ID"].tolist(),
        "Labels": frame["Labels"].astype(int).tolist(),
        "Ensemble_Probabilities": score,
        "Final Predictions": (score >= threshold).astype(int),
    }).to_csv(os.path.join(directory, f"ensemble_fold_{fold}.csv"), index=False)


REPORT_COLUMNS = ["auroc", "ci_low", "ci_high", "balanced_acc", "auc_pr", "mcc",
                  "auroc_mean_of_folds", "n", "n_pos", "mode"]


def report_row(record):
    """The reported columns of one model on one cohort."""
    renamed = dict(record, auroc=record["roc_auc"], n_pos=record["n_positive"])
    return {k: renamed[k] for k in REPORT_COLUMNS}


def write_comparison(out_dir, label, results, headline, cohorts, gate, active):
    rows = []
    for name, per_cohort in results.items():
        for cohort in cohorts:
            if cohort not in per_cohort:
                continue
            rows.append(dict(
                target=label, model=name, is_headline=(name == headline),
                expert_gate_margin=gate, active_experts="+".join(active), cohort=cohort,
                **report_row(per_cohort[cohort]),
            ))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "combiner_comparison.csv"), index=False)


def print_summary(label, results, headline, cohorts):
    names = [n for n in results if n.startswith("expert:")] + COMBINERS
    width = max(len(n) for n in names) + 2
    print("=" * (width + 14 * len(cohorts)))
    for metric, title in [
        ("roc_auc", "AUROC (pooled cross-fit out-of-fold on discovery, fold-averaged externally)"),
        ("balanced_acc", "balanced accuracy (discovery-derived threshold, applied unchanged)"),
    ]:
        print(f"\n{label}: {title}")
        print("model".ljust(width) + "".join(c[:12].ljust(14) for c in cohorts))
        for name in names:
            line = name.ljust(width)
            for cohort in cohorts:
                value = results[name].get(cohort, {}).get(metric)
                line += (
                    "n/a".ljust(14)
                    if value is None or not np.isfinite(value)
                    else f"{value:.3f}".ljust(14)
                )
            print(line + ("   <- reported" if name == headline else ""))
    print("=" * (width + 14 * len(cohorts)))


def main():
    args = build_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    require(
        cfg, "seed", "folds", "data.label_column", "data.discovery_cohort",
        "data.predictions_dir", "experts", "output.results_dir",
    )

    rng = np.random.default_rng(int(cfg.seed))
    label = cfg.data.label_column
    discovery = cfg.data.discovery_cohort
    zero_shot = list(option(cfg, "data.zero_shot_cohorts", []))
    predictions_dir = cfg.data.predictions_dir
    folds = int(cfg.folds)
    n_boot = int(option(cfg, "n_bootstrap", 2000))
    stems = dict(cfg.experts)
    experts = sorted(stems)

    # Gate knobs are parsed before anything is computed so the run folder can be tagged with
    # the margin; runs that differ only in the gate then never collide.
    min_expert_auroc = float(option(cfg, "min_expert_auroc", 0.5))
    configured_margin = option(cfg, "expert_gate_margin")
    auto_gate = isinstance(configured_margin, str) and configured_margin.lower() == "auto"
    gate_margin = None if (configured_margin is None or auto_gate) else float(configured_margin)
    if gate_margin is not None:
        gate, gate_tag = f"{gate_margin:g}", f"_gate{gate_margin:g}"
    elif auto_gate:
        gate, gate_tag = "auto", "_gateauto"
    else:
        gate, gate_tag = "none", ""

    run_id = experiment_id(cfg) + gate_tag
    out_dir = os.path.join(cfg.output.results_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'=' * 78}")
    print(f"Ensembling  target={label}  discovery={discovery}  zero_shot={zero_shot}")
    print(f"experts={experts}  predictions={predictions_dir}  (no GPU)\n{'=' * 78}")

    chosen_runs = {}
    for name in experts:
        eid, a = resolve_best_run(predictions_dir, discovery, stems[name], folds)
        chosen_runs[name] = eid
        if eid is None:
            print(f"  [{name}] no complete run found; using the newest file per fold")
        else:
            print(f"  [{name}] run {eid} (discovery out-of-fold AUROC {a:.3f})")

    discovery_folds = build_cohort(predictions_dir, discovery, folds, experts, stems, chosen_runs)
    if not discovery_folds:
        raise SystemExit(
            f"no {discovery} Expert predictions under {predictions_dir}/{discovery}"
        )
    discovery_pool = pd.concat(
        [discovery_folds[k] for k in sorted(discovery_folds)], ignore_index=True
    )
    print(f"{discovery} out-of-fold: {len(discovery_pool)} slides over "
          f"{len(discovery_folds)} folds, {int(discovery_pool['Labels'].sum())} positive")

    externals = {}
    for cohort in zero_shot:
        folds_data = build_cohort(predictions_dir, cohort, folds, experts, stems, chosen_runs)
        if folds_data:
            externals[cohort] = folds_data
            print(f"{cohort}: {len(folds_data[sorted(folds_data)[0]])} slides "
                  f"x {len(folds_data)} folds (zero-shot)")
        else:
            print(f"[skip] {cohort}: no Expert predictions under {predictions_dir}/{cohort}")
    cohorts = [discovery] + list(externals)

    # The gate decides which Experts feed the mean and the stackers. It reads discovery
    # AUROC only. Single Experts are always evaluated, gated or not.
    expert_auroc = {e: auroc(discovery_pool["Labels"], discovery_pool[e]) for e in experts}
    best_expert = max(expert_auroc, key=lambda e: -1.0 if np.isnan(expert_auroc[e]) else expert_auroc[e])
    best_auroc = expert_auroc[best_expert]
    expert_ci = {
        e: bootstrap_auroc_ci(
            discovery_pool["Labels"].to_numpy(), discovery_pool[e].to_numpy(), rng, n_boot
        )
        for e in experts
    } if auto_gate else {}

    def keep(name):
        value = expert_auroc[name]
        if not np.isfinite(value) or value < min_expert_auroc:
            return False
        if gate_margin is not None and (best_auroc - value) > gate_margin:
            return False
        if auto_gate:
            # Drop only an Expert whose discovery interval lies wholly below the best one's.
            high, best_low = expert_ci[name][1], expert_ci[best_expert][0]
            if np.isfinite(high) and np.isfinite(best_low) and high < best_low:
                return False
        return True

    active = [e for e in experts if keep(e)] or [best_expert]
    print(f"[gate] best discovery Expert {best_expert} ({best_auroc:.3f}); "
          f"floor {min_expert_auroc}; relative {gate}")
    if set(active) != set(experts):
        print(f"[gate] active for mean/stackers: {active} "
              f"(dropped {sorted(set(experts) - set(active))})")

    models = {f"expert:{e}": expert_frames(e, discovery_folds, externals, discovery) for e in experts}
    for kind in COMBINERS:
        frames, combiner = combiner_frames(
            kind, discovery_folds, discovery_pool, externals, discovery, active
        )
        models[kind] = frames
        if kind != "mean":
            w, b = combiner["weights"]
            weights = {name: round(float(v), 3) for name, v in zip(active, w)}
            print(f"[{kind}] weights {weights} intercept {round(b, 3)}")

    results, thresholds = {}, {}
    for name, frames in models.items():
        results[name], thresholds[name] = model_metrics(
            frames, discovery, list(externals), rng, n_boot
        )

    headline = select_headline(results, discovery, str(option(cfg, "headline", "mean")))

    for name, frames in models.items():
        safe = name.replace(":", "_")
        for cohort in cohorts:
            for fold, frame in frames[cohort].items():
                write_fold_csv(
                    os.path.join(out_dir, "by_model", safe, cohort), fold, frame, thresholds[name]
                )
    for cohort in cohorts:
        for fold, frame in models[headline][cohort].items():
            write_fold_csv(os.path.join(out_dir, cohort), fold, frame, thresholds[headline])

    write_comparison(out_dir, label, results, headline, cohorts, gate, active)
    pd.DataFrame([
        dict(target=label, cohort=c, model=headline, expert_gate_margin=gate,
             active_experts="+".join(active), n_folds=folds, **report_row(results[headline][c]))
        for c in cohorts
    ]).to_csv(os.path.join(out_dir, "ensemble_recap.csv"), index=False)

    runs.dump_config(cfg, out_dir, cohorts[0], run_id)
    timestamp = runs.run_timestamp()
    for cohort in cohorts:
        for fold, frame in sorted(models[headline][cohort].items()):
            metrics = {
                k: round4(v) for k, v in classification_metrics(
                    frame["Labels"], frame["score"].to_numpy(), thresholds[headline]
                ).items()
            }
            runs.save_fold_metrics(
                out_dir, cohort, fold, run_id, int(cfg.seed), thresholds[headline], metrics
            )
            runs.append_fold_metrics(
                cfg.output.results_dir, run_id, cohort, fold, int(cfg.seed),
                thresholds[headline], metrics, timestamp,
            )
    runs.append_experiment_recap(
        cfg.output.results_dir, [(run_id, c, results[headline][c]) for c in cohorts]
    )

    print_summary(label, results, headline, cohorts)
    print(f"reported model = {headline} (gate {gate}, active {'+'.join(active)})")
    print(f"  -> {out_dir}/<cohort>/ensemble_fold_<k>.csv")
    print(f"  -> {out_dir}/combiner_comparison.csv")


if __name__ == "__main__":
    main()
