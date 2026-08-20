#!/usr/bin/env python3

import json
import os
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import logrank_test
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leap import runs
from leap.config import (build_parser, experiment_id, load_config, option, require,
                         resolve_device)
from leap.data import CytologyDataset
from leap.determinism import seed_everything
from leap.metrics import cindex, median_split_logrank, rank01
from leap.mil import load_model, slide_embedding
from leap.survival import penultimate_activations, risk_scores, train_survival_model


def build_datasets(cfg):
    """cohort name -> CytologyDataset, refusing duplicate Slide_IDs."""
    datasets = {}
    for name, spec in cfg.cohorts.items():
        dataset = CytologyDataset(
            label_file=spec.label_file,
            image_folder=spec.image_folder,
            label_column=cfg.data.event_column,
            patches_per_slide=int(cfg.data.patches_per_slide),
            augment=False,
            base_seed=int(cfg.seed),
            deterministic=True,
        )
        ids = dataset.slide_ids
        if len(set(ids)) != len(ids):
            raise SystemExit(
                f"{name}: {len(ids) - len(set(ids))} duplicate Slide_IDs in {spec.label_file}; "
                "a survival split cannot be built unambiguously."
            )
        datasets[name] = dataset
        print(f"  {name}: {len(dataset)} slides")
    return datasets


def embed_cohort(label_file, dataset, model, device, reserved):
    """Slide embeddings plus Time, Event and clinical covariates, in dataset order.

    Row i corresponds to dataset.slide_ids[i]. Returns (features_frame, full_frame), where
    features_frame holds Slide_ID, Feature_*, Time and Event, and full_frame adds every
    non-reserved column of the label file as a clinical covariate.
    """
    labels_df = pd.read_excel(label_file)
    clinical_cols = [c for c in labels_df.columns if c not in reserved]
    lookup = labels_df.set_index("Slide_ID")
    time_col, event_col = reserved[1], reserved[2]

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rows, dim = [], None
    for i, (patches, _) in tqdm(enumerate(loader), total=len(loader), leave=False):
        slide_id = dataset.slide_ids[i]
        embedding = slide_embedding(model, patches[0].to(device))
        dim = embedding.shape[0]
        record = lookup.loc[slide_id]
        rows.append(
            [slide_id] + embedding.tolist()
            + [record[time_col], record[event_col]]
            + [record[c] for c in clinical_cols]
        )

    feature_cols = [f"Feature_{i}" for i in range(dim)]
    df = pd.DataFrame(
        rows, columns=["Slide_ID"] + feature_cols + ["Time", "Event"] + clinical_cols
    )
    return df[["Slide_ID"] + feature_cols + ["Time", "Event"]].copy(), df


def standardise(discovery_df, external_dfs):
    """Fit a scaler on the discovery embeddings and apply it to the externals.

    Constant-zero feature columns are dropped first.
    """
    features = discovery_df[[c for c in discovery_df.columns if c.startswith("Feature_")]]
    features = features.loc[:, (features != 0).any(axis=0)]
    columns = features.columns.tolist()

    scaler = StandardScaler()
    out_discovery = pd.DataFrame(scaler.fit_transform(features), columns=columns)
    out_discovery["Time"] = discovery_df["Time"].values
    out_discovery["Event"] = discovery_df["Event"].values

    out_externals = []
    for df in external_dfs:
        scaled = pd.DataFrame(scaler.transform(df[columns]), columns=columns)
        scaled["Time"], scaled["Event"] = df["Time"].values, df["Event"].values
        out_externals.append(scaled)
    return out_discovery, out_externals


def fit_cox(frame, penalizer, what):
    """Fit a Cox model, turning a convergence failure into an actionable message.

    Collinear covariates make the Newton step singular. That is most likely when several
    Experts order the slides almost identically, which is exactly what well-trained Experts
    on the same target do, so the fix is a ridge penalty rather than dropping a covariate.
    """
    try:
        return CoxPHFitter(penalizer=penalizer).fit(
            frame, duration_col="Time", event_col="Event"
        )
    except (ConvergenceError, np.linalg.LinAlgError) as exc:
        raise SystemExit(
            f"the {what} Cox model did not converge with cox_penalizer={penalizer}. "
            f"Its covariates are near-collinear; set a small cox_penalizer (e.g. 0.1) in the "
            f"config and rerun.\n  lifelines reported: {exc}"
        )


def risk_matrix(risks, expert_names):
    return pd.DataFrame(
        np.column_stack([risks[n] for n in expert_names]),
        columns=[f"Model_{i}" for i in range(len(expert_names))],
    )


def summarise(df, column, aggregation):
    """Per-fold, per-patient rows -> headline numbers for one risk column.

    aggregation='oof' is for the discovery cohort, whose test folds are disjoint, so each
    patient appears once: split within each fold, then pool. aggregation='consensus' is for
    external cohorts, where every fold model scores the same patients: rank-normalise per
    fold, average per patient, then run one log-rank at the true sample size.
    """
    df = df[np.isfinite(df[column])]
    if df.empty or df.Event.sum() < 1:
        return {}

    per_fold = [cindex(g.Time, g.Event, g[column]) for _, g in df.groupby("fold")]
    out = {
        "c_index_per_fold": [None if np.isnan(c) else round(c, 4) for c in per_fold],
        "c_index_mean": round(float(np.nanmean(per_fold)), 4),
        "c_index_std": round(float(np.nanstd(per_fold)), 4),
    }

    if aggregation == "oof":
        median = df.groupby("fold")[column].transform("median")
        high = (df[column] > median).to_numpy()
        time = df.Time.to_numpy(float)
        event = df.Event.to_numpy(float)
        result = logrank_test(
            time[~high], time[high],
            event_observed_A=event[~high], event_observed_B=event[high],
        )
        out.update(
            logrank_p=float(result.p_value), n=int(df.Slide_ID.nunique()),
            events=int(event.sum()), aggregation="oof_pooled",
        )
    else:
        consensus = (
            df.assign(_rank=df.groupby("fold")[column].transform(rank01))
            .groupby("Slide_ID")
            .agg(risk=("_rank", "mean"), Time=("Time", "first"), Event=("Event", "first"))
            .reset_index()
        )
        out.update(median_split_logrank(consensus.Time, consensus.Event, consensus.risk))
        out["c_index_consensus"] = cindex(consensus.Time, consensus.Event, consensus.risk)
        out["aggregation"] = "rank_avg_consensus"

        # The same log-rank computed the way an earlier version of this pipeline did it, over
        # every (fold, patient) row rather than one row per patient. It counts each external
        # event once per fold, so it is not a valid test; it is reported so the two can be
        # compared directly.
        median = df.groupby("fold")[column].transform("median")
        high = (df[column] > median).to_numpy()
        time = df.Time.to_numpy(float)
        event = df.Event.to_numpy(float)
        out["logrank_p_LEGACY_5x_pooled"] = float(logrank_test(
            time[~high], time[high],
            event_observed_A=event[~high], event_observed_B=event[high],
        ).p_value)
    return out


def run(cfg, run_id, device):
    out_dir = os.path.join(cfg.output.results_dir, run_id, "tables")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    reserved = ("Slide_ID", cfg.data.time_column, cfg.data.event_column)
    # Blank keys fall through to train_survival_model's own defaults.
    head_params = {k: v for k, v in cfg.survival_head.items() if v is not None}
    penalizer = float(option(cfg, "cox_penalizer", 0.0))

    print("Building datasets")
    datasets = build_datasets(cfg)
    cohorts = list(cfg.cohorts.keys())
    discovery, externals = cohorts[0], cohorts[1:]
    print(f"Discovery: {discovery} | external: {externals}")

    patients, cindex_rows, cox_rows, univariate_rows, embedding_rows = [], [], [], [], []

    for fold in range(int(cfg.folds)):
        print(f"\n{'=' * 70}\nFOLD {fold}\n{'=' * 70}")

        experts = {}
        for name, spec in cfg.experts.items():
            checkpoint = runs.find_checkpoint(cfg.data.checkpoint_dir, spec.stem, fold)
            experts[name] = load_model(spec, checkpoint, device)
            print(f"  expert {name}: {os.path.basename(checkpoint)}")
        expert_names = list(experts)

        features_by_cohort, full_by_cohort = {}, {}
        for cohort, spec in cfg.cohorts.items():
            features_by_cohort[cohort], full_by_cohort[cohort] = {}, {}
            for name, model in experts.items():
                print(f"  embedding {cohort} / {name}")
                features, full = embed_cohort(
                    spec.label_file, datasets[cohort], model, device, reserved
                )
                features_by_cohort[cohort][name] = features
                full_by_cohort[cohort][name] = full

        base = {c: full_by_cohort[c][expert_names[0]].reset_index(drop=True) for c in cohorts}
        slide_ids = {c: base[c]["Slide_ID"].to_numpy() for c in cohorts}
        clinical_cols = [
            c for c in base[discovery].columns
            if not c.startswith("Feature_") and c not in reserved
        ]
        print(f"  clinical covariates ({len(clinical_cols)}): {clinical_cols}")
        for cohort in externals:
            missing = [c for c in clinical_cols if c not in base[cohort].columns]
            if missing:
                raise SystemExit(f"{cohort} label file is missing columns {missing}")

        for name in expert_names:
            scaled_discovery, scaled_externals = standardise(
                features_by_cohort[discovery][name],
                [features_by_cohort[c][name] for c in externals],
            )
            features_by_cohort[discovery][name] = scaled_discovery
            for cohort, scaled in zip(externals, scaled_externals):
                features_by_cohort[cohort][name] = scaled

        X = {
            c: {n: features_by_cohort[c][n].filter(like="Feature_").to_numpy() for n in expert_names}
            for c in cohorts
        }
        times = {c: features_by_cohort[c][expert_names[0]]["Time"].to_numpy() for c in cohorts}
        events = {c: features_by_cohort[c][expert_names[0]]["Event"].to_numpy() for c in cohorts}

        # The discovery partition is derived here rather than read from a frozen split,
        # because the survival stage stratifies on the event indicator, not on the
        # classification label the Experts were split by.
        skf = StratifiedKFold(n_splits=int(cfg.folds), shuffle=True, random_state=int(cfg.seed))
        train_idx, test_idx = list(skf.split(X[discovery][expert_names[0]], events[discovery]))[fold]
        print(f"  discovery split: {len(train_idx)} train / {len(test_idx)} test")

        risks = {"train": {}, "test": {}, "full": {c: {} for c in externals}}
        for name in expert_names:
            print(f"  fitting survival head [{name}]")
            head = train_survival_model(
                X[discovery][name][train_idx],
                {"Time": times[discovery][train_idx], "Event": events[discovery][train_idx]},
                device=device,
                **head_params,
            )
            risks["train"][name] = risk_scores(head, X[discovery][name][train_idx], device)
            risks["test"][name] = risk_scores(head, X[discovery][name][test_idx], device)
            for cohort in externals:
                risks["full"][cohort][name] = risk_scores(head, X[cohort][name], device)

            for cohort, rows in [(discovery, test_idx)] + [(c, None) for c in externals]:
                features = X[cohort][name] if rows is None else X[cohort][name][rows]
                ids = slide_ids[cohort] if rows is None else slide_ids[cohort][rows]
                hidden = penultimate_activations(head, features, device)
                frame = pd.DataFrame(hidden, columns=[f"e{i}" for i in range(hidden.shape[1])])
                frame.insert(0, "risk", risk_scores(head, features, device))
                frame.insert(0, "Slide_ID", ids)
                frame.insert(0, "split", "test" if cohort == discovery else "full")
                frame.insert(0, "model", name)
                frame.insert(0, "fold", fold)
                frame.insert(0, "cohort", cohort)
                embedding_rows.append(frame)

        # Stage 1: combine the per-Expert risks into the pathology risk score.
        train_matrix = risk_matrix(risks["train"], expert_names)
        train_matrix["Time"] = times[discovery][train_idx]
        train_matrix["Event"] = events[discovery][train_idx]
        stage1 = fit_cox(train_matrix, penalizer, "stage-1 Expert combination")

        leap_risk = {
            (discovery, "train"): stage1.predict_partial_hazard(
                risk_matrix(risks["train"], expert_names)
            ).to_numpy(),
            (discovery, "test"): stage1.predict_partial_hazard(
                risk_matrix(risks["test"], expert_names)
            ).to_numpy(),
        }
        for cohort in externals:
            leap_risk[(cohort, "full")] = stage1.predict_partial_hazard(
                risk_matrix(risks["full"][cohort], expert_names)
            ).to_numpy()

        # Stage 2: add the clinical covariates.
        stage2_train = base[discovery].iloc[train_idx][["Time", "Event"] + clinical_cols].copy()
        stage2_train["Risk_Score"] = leap_risk[(discovery, "train")]
        complete_train = stage2_train.dropna()
        print(f"  stage-2 Cox: {len(complete_train)}/{len(stage2_train)} complete-case rows")
        stage2 = fit_cox(complete_train, penalizer, "stage-2 clinical")
        summary = stage2.summary.reset_index().rename(columns={"index": "covariate"})
        summary.insert(0, "fold", fold)
        cox_rows.append(summary)

        held_out = {}
        for cohort, rows, split in (
            [(discovery, train_idx, "train"), (discovery, test_idx, "test")]
            + [(c, None, "full") for c in externals]
        ):
            source = base[cohort] if rows is None else base[cohort].iloc[rows]
            frame = source[["Slide_ID", "Time", "Event"] + clinical_cols].reset_index(drop=True)
            frame.insert(0, "split", split)
            frame.insert(0, "fold", fold)
            frame.insert(0, "cohort", cohort)
            for name in expert_names:
                frame[f"risk_{name}"] = (
                    risks[split][name] if cohort == discovery
                    else risks["full"][cohort][name]
                )
            frame["risk_leap"] = leap_risk[(cohort, split)]
            frame["complete"] = frame[clinical_cols].notna().all(axis=1).to_numpy()

            covariates = frame[clinical_cols].copy()
            covariates["Risk_Score"] = frame["risk_leap"].to_numpy()
            frame["risk_full"] = np.nan
            complete = frame["complete"].to_numpy()
            if complete.any():
                frame.loc[complete, "risk_full"] = stage2.predict_partial_hazard(
                    covariates[complete]
                ).to_numpy()
            patients.append(frame)
            if split != "train":
                held_out[cohort] = frame

        for cohort, frame in held_out.items():
            for variant, column in [("leap_alone", "risk_leap"),
                                    ("clinical_enhanced", "risk_full")]:
                for population, subset in [("all", frame), ("complete", frame[frame.complete])]:
                    cindex_rows.append(dict(
                        cohort=cohort, fold=fold, variant=variant, population=population,
                        c_index=cindex(subset.Time, subset.Event, subset[column]),
                        n=int(np.isfinite(subset[column]).sum()),
                        events=int(subset.Event.sum()),
                    ))

        # Univariate Cox, one covariate at a time. Risk_Score is inside this loop, so the
        # pathology bar and the clinical bars come from the same rows under the same protocol.
        if option(cfg, "univariate_cox", True):
            fitted = complete_train
            held = base[discovery].iloc[test_idx][["Time", "Event"] + clinical_cols].copy()
            held["Risk_Score"] = leap_risk[(discovery, "test")]
            held = held.dropna()
            for covariate in [c for c in fitted.columns if c not in ("Time", "Event")]:
                try:
                    model = CoxPHFitter(penalizer=penalizer).fit(
                        fitted[[covariate, "Time", "Event"]],
                        duration_col="Time", event_col="Event",
                    )
                    predicted = model.predict_partial_hazard(held[[covariate]]).to_numpy()
                    univariate_rows.append(dict(
                        fold=fold, covariate=covariate,
                        c_index=cindex(held.Time, held.Event, predicted),
                        logrank_p=median_split_logrank(held.Time, held.Event, predicted)["p"],
                        hr=float(np.exp(model.params_[covariate])), n=int(len(held)),
                    ))
                except Exception as exc:
                    print(f"    skipped univariate {covariate}: {exc}")

    all_patients = pd.concat(patients, ignore_index=True)
    all_patients.to_csv(f"{out_dir}/patients.csv", index=False)
    pd.DataFrame(cindex_rows).to_csv(f"{out_dir}/cindex.csv", index=False)
    pd.concat(cox_rows, ignore_index=True).to_csv(f"{out_dir}/cox_stage2.csv", index=False)
    if univariate_rows:
        pd.DataFrame(univariate_rows).to_csv(f"{out_dir}/univariate.csv", index=False)
    runs.write_table(
        pd.concat(embedding_rows, ignore_index=True), f"{out_dir}/leapsurv_emb.parquet"
    )

    results = {
        "experiment_id": run_id, "discovery": discovery, "folds": int(cfg.folds),
        "experts": list(cfg.experts.keys()), "cohorts": {},
    }
    for cohort in cohorts:
        subset = all_patients[
            (all_patients.cohort == cohort)
            & (all_patients.split != "train")
            & all_patients.complete
        ]
        aggregation = "oof" if cohort == discovery else "consensus"
        results["cohorts"][cohort] = {
            variant: summarise(subset, column, aggregation)
            for variant, column in [("leap_alone", "risk_leap"),
                                    ("clinical_enhanced", "risk_full")]
        }
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'=' * 70}\nwrote {out_dir}\n")
    for cohort in cohorts:
        for variant in ("leap_alone", "clinical_enhanced"):
            record = results["cohorts"][cohort].get(variant) or {}
            if record:
                print(f"  {cohort:10s} {variant:18s} "
                      f"C = {record['c_index_mean']:.3f} +/- {record['c_index_std']:.3f}   "
                      f"n = {record.get('n')}   log-rank p = "
                      f"{record.get('logrank_p', record.get('p', float('nan'))):.2e}")


def main():
    args = build_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    require(
        cfg, "seed", "folds", "cohorts", "experts", "data.checkpoint_dir",
        "data.patches_per_slide", "data.time_column", "data.event_column",
        "output.results_dir", "survival_head",
    )
    # This pipeline does not set CUBLAS_WORKSPACE_CONFIG, so deterministic cuBLAS is not
    # requested; asking for it would make every matrix multiply raise.
    seed_everything(int(cfg.seed), deterministic_algorithms=False)
    device = resolve_device(option(cfg, "device"))
    run_id = experiment_id(cfg)
    print(f"experiment {run_id} | device {device}")
    run(cfg, run_id, device)


if __name__ == "__main__":
    main()
