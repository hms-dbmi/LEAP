import sys
import os
import hydra
import numpy as np
from collections import defaultdict
import torch
from omegaconf import DictConfig
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from utils.helpers import (
    seed_everything, 
    save_results_to_csv,
    update_experiment_recap_csv,
)
from datasets.datasets import CytologyDataset
from models.LEAP_expert import build_LEAP_pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
script_dir = os.path.dirname(os.path.abspath(__file__))
cfgs_pth = os.path.abspath(os.path.join(script_dir, "..", "configs"))


def get_model_probs(cfg, test,model,fold):
    test_dl = DataLoader(test, batch_size=4, shuffle=False)
    device = cfg.device
    all_labels = []
    all_probs = []
    bar = tqdm(test_dl, total=len(test_dl), desc=f"Getting data of {model} for fold{fold}")
    model.eval()

    with torch.no_grad():
        for batch in bar:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            output_logits = model(images)
            output_probs = torch.sigmoid(output_logits)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(output_probs.detach().cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        return all_probs , all_labels

def get_datasets_for_fold(cfg, fold):
    labels_df = pd.read_excel(cfg.data.label_file)
    skf = StratifiedKFold(n_splits=cfg.data.folds, shuffle=True, random_state=cfg.seed)
    splits = list(skf.split(labels_df['Slide_ID'], labels_df[cfg.data.label_column]))
    train_idx, test_idx = splits[fold]

    train_dataset = CytologyDataset(
        label_file=cfg.data.label_file,
        image_folder=cfg.data.image_folder,
        label_column=cfg.data.label_column,
        tile_number=cfg.data.tile_number,
        transform=None,
        augment=cfg.data.augment,
        index_list=train_idx
    )

    test_dataset = CytologyDataset(
        label_file=cfg.data.label_file,
        image_folder=cfg.data.image_folder,
        label_column=cfg.data.label_column,
        tile_number=cfg.data.tile_number,
        transform=None,
        augment=cfg.data.augment,
        index_list=test_idx
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    return train_dataset, test_dataset

    

@hydra.main(version_base=None, config_path=cfgs_pth, config_name=None)
def main(cfg: DictConfig):
    print("Seeding everything...")
    seed_everything(cfg.seed)
    
    model_dict = {}
    fold_metrics = {cfg.data.cohort: defaultdict(list)}

    for fold in range(cfg.data.folds):
        print(f"Processing fold {fold}...")
        
        # Loading models:
        for model_name, model_cfg in cfg.model_to_ensemble.items():  
            print(f"model name is {model_name} and model config is {model_cfg}")
            model = build_LEAP_pipeline(model_cfg).to(cfg.device)
            print(f"Instantiating {model_name}...")
            model.load_state_dict(torch.load(f"{model_cfg.weights_path}_{fold}.pth"))
            model.eval()
            print(f"Loaded weights for {model_name}.")
            model_dict[model_name] = model.to(cfg.device)
        
        train_dataset, test_dataset = get_datasets_for_fold(cfg, fold)
        
        all_train_probs = []
        all_test_probs = []

        #Retrieving trained models probs:
        for model_name, model in model_dict.items():
            
            train_probs, train_labels = get_model_probs(cfg, train_dataset, model, fold)
            test_probs, test_labels = get_model_probs(cfg, test_dataset, model, fold)
            all_train_probs.append(train_probs[:, 0])
            all_test_probs.append(test_probs[:, 0])
            

        all_train_probs = np.stack(all_train_probs, axis=-1).squeeze()
        all_test_probs = np.stack(all_test_probs, axis=-1).squeeze()
        
        # TRAINING:
        print("TRAINING META LEARNER...")
        meta_learner = LogisticRegression(penalty="l2",tol=1e-4, C=1 ,solver="lbfgs",max_iter=100,verbose=0)
        for i in tqdm(range(meta_learner.max_iter), desc="Training Progress"):
            meta_learner.fit(all_train_probs, train_labels)
            
        
        # TESTING
        test_meta_probs = meta_learner.predict_proba(all_test_probs)[:, 1]
        thresholds = np.arange(0.0, 1.0, 0.00001)
        best_threshold = 0.5
        best_bacc = 0
        final_preds = []

        for threshold in thresholds:
            preds = (test_meta_probs >= threshold).astype(int)
            bacc = balanced_accuracy_score(test_labels, preds)
            if bacc > best_bacc:
                best_bacc = bacc
                best_threshold = threshold


        final_preds = (test_meta_probs >= best_threshold).astype(int)
        balanced_acc = balanced_accuracy_score(test_labels, final_preds)
        print(f"Balanced Accuracy: {balanced_acc}")
        roc_auc = roc_auc_score(test_labels, test_meta_probs)
        print(f"AUC-ROC: {roc_auc:.4f}")
        
        inf_metrics = {
                    "roc_auc": round(roc_auc,3),
                    "balanced_acc": round(balanced_acc,3),
                }
                
        for metric, value in inf_metrics.items():
            fold_metrics[cfg.data.cohort][metric].append(value)
            
        save_results_to_csv(cfg, test_labels, test_meta_probs, final_preds, pth= cfg.experiment_recap.prediction_folder,cohort=cfg.data.cohort, fold=fold)
               
    averaged_metrics = {
        cfg.data.cohort: {
            metric: round(sum(values) / len(values), 3)
            for metric, values in fold_metrics[cfg.data.cohort].items()
        }
    }
    update_experiment_recap_csv(model_name=f"{cfg.logger.experiment_id}",cohort_metrics=averaged_metrics,save_path=f"{cfg.experiment_recap.fold_metricfile}/Experiment_Metrics.csv")



if __name__ == "__main__":
    main()
