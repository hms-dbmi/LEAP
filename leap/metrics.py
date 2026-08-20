import numpy as np
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

METRIC_KEYS = ["roc_auc", "balanced_acc", "mcc", "auc_pr", "weighted_f1", "macro_f1"]


# Stand-in values reported when a fold carries a single label class, so AUROC and average
# precision are undefined. They are placeholders, not measurements; a stratified split with at
# least one positive per fold never reaches them.
DEGENERATE_ROC_AUC = 0.5
DEGENERATE_AUC_PR = 0.1


def auroc(y, p):
    """AUROC, or NaN if `y` holds a single class."""
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def auprc(y, p):
    """Average precision, or NaN if `y` holds a single class."""
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


# The operating point is searched over a fixed grid rather than over the observed
# probabilities. The grid value matters: the threshold is chosen on one split and applied
# unchanged to another, so a probability in the other split can fall between an observed
# value and the grid point just below it and be classified differently.
FINE_GRID = np.arange(0.0, 1.0, 0.00001)
COARSE_GRID = np.linspace(0.0, 1.0, 1001)


def best_threshold(p, y, grid=FINE_GRID):
    """Grid threshold on `p` maximising balanced accuracy against `y`.

    Predictions are `p >= threshold`. Ties resolve to the lowest grid value, and a grid on
    which no threshold scores above zero returns 0.5.
    """
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y).ravel().astype(int)

    # Balanced accuracy is the mean recall over the classes present in `y`. Counting with
    # searchsorted evaluates the whole grid at once and matches a per-threshold loop exactly.
    positives = np.sort(p[y == 1])
    negatives = np.sort(p[y == 0])
    recalls = []
    if len(positives):
        recalls.append(
            (len(positives) - np.searchsorted(positives, grid, side="left")) / len(positives)
        )
    if len(negatives):
        recalls.append(np.searchsorted(negatives, grid, side="left") / len(negatives))
    if not recalls:
        return 0.5

    scores = np.mean(recalls, axis=0)
    return float(grid[int(np.argmax(scores))]) if scores.max() > 0 else 0.5


def classification_metrics(y, p, threshold):
    """The six reported classification metrics at a fixed threshold.

    When `y` holds a single class the threshold-free entries fall back to
    DEGENERATE_ROC_AUC and DEGENERATE_AUC_PR.
    """
    y = np.asarray(y).ravel().astype(int)
    p = np.asarray(p).ravel()
    preds = (p >= threshold).astype(int)
    two_classes = len(np.unique(y)) > 1
    return {
        "roc_auc": auroc(y, p) if two_classes else DEGENERATE_ROC_AUC,
        "balanced_acc": float(balanced_accuracy_score(y, preds)),
        "mcc": float(matthews_corrcoef(y, preds)),
        "auc_pr": auprc(y, p) if two_classes else DEGENERATE_AUC_PR,
        "weighted_f1": float(f1_score(y, preds, average="weighted")),
        "macro_f1": float(f1_score(y, preds, average="macro")),
    }


def bootstrap_auroc_ci(y, p, rng, n_boot=2000):
    """Percentile bootstrap 95% CI for AUROC, resampling slides with replacement."""
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    if len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    n, scores = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) > 1:
            scores.append(roc_auc_score(y[idx], p[idx]))
    if not scores:
        return (float("nan"), float("nan"))
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def early_stopping_score(metrics):
    """Weighted composite of validation metrics used to select the best epoch.

    Emphasises the threshold-free metrics: AUPRC is weighted 2.5, AUROC 2, balanced accuracy
    and weighted F1 1 each, normalised by the total weight.
    """
    return (
        metrics["balanced_acc"]
        + 2.5 * metrics["auc_pr"]
        + 2.0 * metrics["roc_auc"]
        + metrics["weighted_f1"]
    ) / 7.5


def _clean_survival(time, event, risk):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    keep = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    return time[keep], event[keep], risk[keep]


def cindex(time, event, risk):
    """Harrell's concordance index. `risk` is a log-hazard, so higher means worse prognosis.

    Returns NaN when there are fewer than 3 usable rows, no events, or a constant risk.
    """
    time, event, risk = _clean_survival(time, event, risk)
    if len(time) < 3 or event.sum() < 1 or np.allclose(risk, risk[0]):
        return float("nan")
    return float(concordance_index(time, -risk, event))


def rank01(x):
    """Rank-normalise to [0, 1].

    Puts separately-trained fold models, whose log-hazards live on different scales, on a
    common footing before averaging.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.zeros_like(x)
    return (rankdata(x, method="average") - 1) / (len(x) - 1)


def median_split_logrank(time, event, risk):
    """Split at the median risk and log-rank the two arms.

    Expects one row per patient: a frame carrying the same patient once per fold counts every
    event once per fold and inflates the test. Returns a dict with the p-value and group sizes.
    """
    time, event, risk = _clean_survival(time, event, risk)
    if len(risk) < 4 or event.sum() < 1 or np.allclose(risk, risk[0]):
        return {"p": float("nan"), "n": int(len(risk)), "events": int(event.sum())}
    high = risk > np.median(risk)
    if high.sum() == 0 or (~high).sum() == 0:
        return {"p": float("nan"), "n": int(len(risk)), "events": int(event.sum())}
    result = logrank_test(
        time[~high], time[high], event_observed_A=event[~high], event_observed_B=event[high]
    )
    return {
        "p": float(result.p_value),
        "n": int(len(risk)),
        "events": int(event.sum()),
        "n_high": int(high.sum()),
        "n_low": int((~high).sum()),
    }
