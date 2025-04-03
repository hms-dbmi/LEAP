import io
import os
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import random

def seed_everything(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def create_roc_curve(labels: np.array, probs: np.array) -> np.array:
    roc_auc = roc_auc_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    img_array = plot_to_image(fig)
    fig.clf()
    return img_array

def plot_to_image(figure, dpi=300):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi)
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image).transpose(2, 0, 1)
    
    
def is_performant(metrics, thresholds):
    return all(metrics[k] >= thresholds[k] for k in thresholds)


def compute_early_stopping_score(metrics, formula: str):
    allowed_names = {k: metrics[k] for k in metrics}
    return eval(formula, {"__builtins__": None}, allowed_names)


def save_model(cfg, model, fold):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = cfg.checkpoints.save_dir
    save_dir = os.path.join(base_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    model_pth = f"{cfg.logger.experiment_id}_{fold}.pth"
    torch.save(model.cpu().state_dict(), os.path.join(save_dir, model_pth))
    model.to(cfg.train.device)
    
def update_metric_file(model_name, cohort, cohorts, metrics, metricfile):
    metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])

    if os.path.exists(metricfile):
        df = pd.read_csv(metricfile)
    else:
        df = pd.DataFrame(columns=['MODEL'] + [model_name])
        df['MODEL'] = cohorts  

    if model_name not in df.columns:
        df[model_name] = ""

    if cohort not in df['MODEL'].values:
        new_row = pd.DataFrame({'MODEL': [cohort], model_name: [metrics_str]})
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df.loc[df['MODEL'] == cohort, model_name] = metrics_str

    folder = os.path.dirname(metricfile)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    df.to_csv(metricfile, index=False)

    
def save_results_to_csv(cfg, all_labels, all_probs, final_preds, pth, cohort, fold):
    
    final_preds_flat = [pred[0] if isinstance(pred, (list, tuple)) else pred for pred in final_preds] # Make sure pred is flat for csv
    
    results_df = pd.DataFrame({
        'Labels': all_labels,
        'Probabilities': all_probs,
        'Final Predictions': final_preds_flat
    })

    output_dir = os.path.join(pth, cohort)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{cfg.logger.experiment_id}_fold{fold}.csv")
    results_df.to_csv(output_file, index=False)
    
    

    
def update_experiment_recap_csv(model_name, cohort_metrics, save_path):
    data = []
    for cohort, metrics in cohort_metrics.items():
        row = {
            "model_name": model_name,
            "cohort": cohort,
        }
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        existing_df = pd.read_csv(save_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        updated_df = df
    
    updated_df.to_csv(save_path, index=False)
    print(f"Averaged metrics saved to {save_path}")
