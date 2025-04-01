import sys
import os
from typing import Optional
import hydra
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd

from utils.helpers import (
    seed_everything, 
    is_performant,
    compute_early_stopping_score,
    save_model,
    update_metric_file,
    save_results_to_csv,
    update_experiment_recap_csv,
    create_roc_curve
)
from utils.logger import MetricsLogger
from datasets.datasets import CytologyDataset
from models.LEAP_expert import build_LEAP_pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
script_dir = os.path.dirname(os.path.abspath(__file__))
cfgs_pth = os.path.abspath(os.path.join(script_dir, "..", "configs"))



def get_datasets_for_fold(cfg, fold):
    discovery_index = cfg.data.cohorts.index(cfg.data.discovery_cohort)
    labels_df = pd.read_excel(cfg.data.label_files[discovery_index])
    skf = StratifiedKFold(n_splits=cfg.data.folds, shuffle=True, random_state=cfg.seed)
    splits = list(skf.split(labels_df['Slide_ID'], labels_df[cfg.data.label_column]))
    train_idx, test_idx = splits[fold]
    
    train_labels = labels_df.iloc[train_idx][cfg.data.label_column]
    train_slide_ids = labels_df.iloc[train_idx]['Slide_ID']
    train_idx_strat, val_idx_strat = train_test_split(
        train_idx, test_size=0.25, stratify=train_labels, random_state=cfg.seed
    )
    
    train_dataset = CytologyDataset(
        label_file=cfg.data.label_files[discovery_index],
        image_folder=cfg.data.image_folders[discovery_index],
        label_column=cfg.data.label_column,
        tile_number=cfg.data.tile_number,
        transform=None,
        augment=cfg.data.augment,
        index_list=train_idx_strat
    )
    
    val_dataset = CytologyDataset(
        label_file=cfg.data.label_files[discovery_index],
        image_folder=cfg.data.image_folders[discovery_index],
        label_column=cfg.data.label_column,
        tile_number=cfg.data.tile_number,
        transform=None,
        augment=cfg.data.augment,
        index_list=val_idx_strat  
    )
    
    test_dataset = CytologyDataset(
        label_file=cfg.data.label_files[discovery_index],
        image_folder=cfg.data.image_folders[discovery_index],
        label_column=cfg.data.label_column,
        tile_number=cfg.data.tile_number,
        transform=None,
        augment=cfg.data.augment,
        index_list=test_idx
    )
    
    print(f"Length of train dataset: {len(train_dataset)} with weights {train_dataset.weights}")
    print(f"Length of val dataset: {len(val_dataset)} with weights {val_dataset.weights}")
    print(f"Length of test dataset: {len(test_dataset)} with weights {test_dataset.weights}")
    
    return train_dataset, val_dataset, test_dataset

def train_one_epoch(
    cfg: DictConfig,
    model: nn.Module,
    dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    logger: MetricsLogger,
):
    accumulation_steps =cfg.data.complete_batch_size // cfg.data.batch_size  # Effective batch size =32, small batch size = 4
    
    all_labels = []
    all_preds = []
    all_probs = []
    bar = tqdm(dl, total=len(dl), desc=f"Train Epoch {epoch}")
    model.train()

    optimizer.zero_grad()

    for i, batch in enumerate(bar):
        if len(batch[0]) == 0:
            continue
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
    
        # Forward pass
        output_logits = model(images)
        output_probs = torch.sigmoid(output_logits)
        preds = (output_probs >= 0.5).long()

        # Calculate loss
        loss = criterion(output_logits, labels.unsqueeze(1))
        loss = loss / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dl):  
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  
            optimizer.zero_grad() 

        acc = (preds.squeeze() == labels).float().mean()
        all_labels.extend(labels.cpu().numpy())
        
        # Handle different cases for preds:
        preds_np = preds.squeeze().cpu().numpy()
        if preds_np.ndim == 0:
            all_preds.append(preds_np.item())
        else:
            all_preds.extend(preds_np)
            
        all_probs.extend(output_probs.detach().cpu().numpy())
        logger.log_dict({"train/loss": loss.item() * accumulation_steps})
        logger.log_dict({"train/acc": acc.item()})

    # Final check for remaining gradients if not updated
    if (i + 1) % accumulation_steps != 0:
        optimizer.step() 
        optimizer.zero_grad()

    # Epoch metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    thresholds = np.arange(0.0, 1.0, 0.00001)
    best_threshold = 0.5
    best_bacc = 0

    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)
        bacc = balanced_accuracy_score(all_labels, preds)

        if bacc > best_bacc:
            best_bacc = bacc
            best_threshold = threshold


    final_preds = (all_probs >= best_threshold).astype(int)    
    balanced_acc = balanced_accuracy_score(all_labels, final_preds)
    mcc = matthews_corrcoef(all_labels, final_preds)
    weighted_f1 = f1_score(all_labels, final_preds, average="weighted")
    macro_f1 = f1_score(all_labels, final_preds, average="macro")
    
    if cfg.data.n_classes == 2 and len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 0])
        auc_pr = average_precision_score(all_labels, all_probs[:, 0])
        curve_np = create_roc_curve(all_labels, all_probs[:, 0])
        logger.log_image("train/roc_curve", curve_np)
    else:
        auc_pr = 0.1
        roc_auc = 0.5
    
    #logging
    logger.log_dict({"train/balanced_acc": balanced_acc})
    logger.log_dict({"train/roc_auc": roc_auc})
    logger.log_dict({"train/mcc": mcc})
    logger.log_dict({"train/weighted_f1": weighted_f1})
    logger.log_dict({"train/macro_f1": macro_f1})
    logger.log_dict({"train/auc_pr": auc_pr})
    
    metrics = {
        "roc_auc": roc_auc,
        "balanced_acc": balanced_acc,
        "mcc": mcc,
        "auc_pr": auc_pr,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
    }
    return metrics



def val_one_epoch(
    cfg: DictConfig,
    model: nn.Module,
    dl: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: Optional[MetricsLogger] = None,
    return_preds: bool = False,
    mode: str = "val",
    fold: Optional[int] = None,
):
    all_labels = []
    all_preds = []
    all_probs = []
    desc = f"{mode.capitalize()} Epoch {epoch}" if mode == "val" else f"Test set {fold if fold is not None else ''}"
    bar = tqdm(dl, total=len(dl), desc=desc)
    model.eval()

    with torch.no_grad():
        for batch in bar:
            if len(batch[0]) == 0:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            output_logits = model(images)
            output_probs = torch.sigmoid(output_logits)
            preds = (output_probs >= 0.5).long()

            # Calculate loss
            loss = criterion(output_logits, labels.unsqueeze(1))

            # Metrics calculation
            acc = (preds.squeeze() == labels).float().mean()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(np.atleast_1d(preds.squeeze().cpu().numpy()))
            all_probs.extend(output_probs.detach().cpu().numpy())
            
            if logger:
                logger.log_dict({"val/loss": loss.item()})
                logger.log_dict({"val/acc": acc.item()})

        # Epoch metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        thresholds = np.arange(0.0, 1.0, 0.00001)
        best_threshold = 0.5
        best_bacc = 0

        for threshold in thresholds:
            preds = (all_probs >= threshold).astype(int)
            bacc = balanced_accuracy_score(all_labels, preds)

            if bacc > best_bacc:
                best_bacc = bacc
                best_threshold = threshold


        final_preds = (all_probs >= best_threshold).astype(int)        
        balanced_acc = balanced_accuracy_score(all_labels, final_preds)
        mcc = matthews_corrcoef(all_labels, final_preds)
        weighted_f1 = f1_score(all_labels, final_preds, average="weighted")
        macro_f1 = f1_score(all_labels, final_preds, average="macro")

        if cfg.data.n_classes == 2 and len(np.unique(all_labels)) > 1:
            auc_pr = average_precision_score(all_labels, all_probs[:, 0])
            roc_auc = roc_auc_score(all_labels, all_probs[:, 0])
            curve_np = create_roc_curve(all_labels, all_probs[:, 0])
            if logger:
                logger.log_image("val/roc_curve", curve_np)
        else:
            auc_pr = 0.1
            roc_auc = 0.5

        #logging
        if logger:
            logger.log_dict({"val/balanced_acc": balanced_acc})
            logger.log_dict({"val/roc_auc": roc_auc})
            logger.log_dict({"val/mcc": mcc})
            logger.log_dict({"val/weighted_f1": weighted_f1})
            logger.log_dict({"val/macro_f1": macro_f1})
            logger.log_dict({"val/auc_pr": auc_pr})

        metrics = {
            "roc_auc": roc_auc,
            "balanced_acc": balanced_acc,
            "mcc": mcc,
            "auc_pr": auc_pr,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
        }
        if return_preds:
            return metrics, final_preds, all_probs, all_labels
        return metrics
    
    
    

def run_k_fold(cfg: DictConfig):
    cohorts = cfg.data.cohorts

    # Logger initialization
    logger = instantiate(cfg.logger)
    logger.log_cfg(OmegaConf.to_container(cfg, resolve=True))
    performant_model_found = False  # Flag to save the first model with sufficient performance
    
    # To collect metrics for averaging
    fold_metrics = {cohort: defaultdict(list) for cohort in cohorts}

    for fold in range(cfg.data.folds):
        print(f"fold is {fold}")
        logger.set_fold(fold, cfg)
        
        # Get train, validation, and test datasets for the fold
        print(f"Getting Datasets:")
        train, val, test = get_datasets_for_fold(cfg, fold)
        sampler = WeightedRandomSampler(train.weights, len(train.weights), replacement=True)
        
        print("Getting Dataloaders:")
        train_dl = DataLoader(train, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, sampler=sampler)
        val_dl = DataLoader(val, batch_size=cfg.data.batch_size, shuffle=False)
        test_dl = DataLoader(test, batch_size=cfg.data.batch_size, shuffle=False)

        
        model = build_LEAP_pipeline(cfg).to(cfg.train.device)
        print("Model built!")
        
        optimizer = instantiate(cfg.optimizer, filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.optimizer.lr)
        criterion = instantiate(cfg.criterion)

        if cfg.train.early_stopping:
            patience = cfg.train.patience
            best = 0
            patience_counter = 0

        # TRAINING
        print("TRAINING...")
        for epoch in range(cfg.train.epochs):
            _ = train_one_epoch(cfg, model, train_dl, optimizer, criterion, cfg.train.device, epoch, logger)
            metrics = val_one_epoch(cfg, model, val_dl, criterion, cfg.train.device, epoch, logger)
            
            if not performant_model_found and is_performant(metrics, cfg.train.performant_model_thresholds):
                print("TRIGGERED PERFORMANT MODEL, SAVING...")
                save_model(cfg, model, fold)
                performant_model_found = True

            if performant_model_found:
                print(f"Good model found, stopping at : {epoch + 1} epochs.")
                break

            if cfg.train.early_stopping:
                new = compute_early_stopping_score(metrics, cfg.train.early_stopping_formula)
                if new > best:
                    best = new
                    patience_counter = 0
                    save_model(cfg, model, fold)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
                    
                    
        # TESTING:
        print("TESTING...")
        model.load_state_dict(torch.load(f"{cfg.checkpoints.save_dir}/{cfg.logger.experiment_id}_{fold}.pth"))
        print("Best model Weights successfully loaded!!")
        
        test_metrics, final_preds, test_probs, test_labels = val_one_epoch(cfg=cfg,model=model,dl=test_dl,criterion=criterion,device=cfg.train.device,epoch=0,logger=None,return_preds=True,mode="test",fold=fold)
        
        #Collect the metrics for the experiment file update:
        for metric, value in test_metrics.items():
            fold_metrics[cfg.data.discovery_cohort][metric].append(value)
        
        print("Saving Metrics on discovery test set...")
        print(f"Metrics are : {test_metrics}")
        update_metric_file(model_name=f"{cfg.logger.experiment_id}_fold{fold}", cohort=cfg.data.discovery_cohort,cohorts = cohorts, metrics=test_metrics, metricfile = f"{cfg.experiment_recap.fold_metricfile}/Fold_Metrics.csv")
        save_results_to_csv(cfg, test_labels, test_probs[:, 0], final_preds, pth = cfg.experiment_recap.prediction_folder, cohort = cfg.data.discovery_cohort , fold = fold)
        print("Done!")
            
            
        print("Running Inference on Validation Cohorts...")
        for cohort in cfg.data.cohorts:
            if cohort == cfg.data.discovery_cohort:
                continue
            print(f"Cohort is {cohort}")
            cohort_index = cfg.data.cohorts.index(cohort)
            
            dataset = CytologyDataset(
                label_file=cfg.data.label_files[cohort_index],
                image_folder=cfg.data.image_folders[cohort_index],
                label_column=cfg.data.label_column,
                tile_number=cfg.data.tile_number,
                transform=None,
                augment=cfg.data.augment,
            )
            
            dl = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.data.batch_size, shuffle=False
            )
                        
            inf_metrics, final_preds, test_probs, test_labels = val_one_epoch(cfg=cfg,model=model,dl=dl,criterion=criterion,device=cfg.train.device,epoch=0,logger=None,return_preds=True,mode="test",fold=cohort)

            for metric, value in inf_metrics.items():
                fold_metrics[cohort][metric].append(value)
            
            print("Saving Metrics...")
            print(f"Inference metrics are : {inf_metrics}")
            update_metric_file(model_name = f"{cfg.logger.experiment_id}_fold{fold}", cohort = cohort , cohorts = cohorts,metrics = inf_metrics, metricfile = f"{cfg.experiment_recap.fold_metricfile}/Fold_Metrics.csv")
            save_results_to_csv(cfg, test_labels, test_probs[:, 0], final_preds, pth = cfg.experiment_recap.prediction_folder, cohort = cohort, fold = fold)
            print("Done!")
            
        performant_model_found = False
        
    averaged_metrics = {
        cohort: {metric: round(sum(values) / len(values), 3) for metric, values in metrics.items()}
        for cohort, metrics in fold_metrics.items()
    }
    update_experiment_recap_csv(model_name=f"{cfg.logger.experiment_id}",cohort_metrics=averaged_metrics,save_path=f"{cfg.experiment_recap.fold_metricfile}/Experiment_Metrics.csv")
    



@hydra.main(version_base=None, config_path=cfgs_pth, config_name=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    pipeline_name = f"{cfg.extractor._target_}_{cfg.head._target_}"
    cfg.logger.experiment_id = f"{cfg.logger.experiment_id}.{pipeline_name}"
        
    print(f"CONFIG EXPERIMENT ID IS : {cfg.logger.experiment_id }")
    run_k_fold(cfg)
    
    
if __name__ == "__main__":
    main()
