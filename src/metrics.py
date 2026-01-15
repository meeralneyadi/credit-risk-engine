import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss


def ks_statistic(y_true, y_prob) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]

    pos = (y_true_sorted == 1).astype(int)
    neg = (y_true_sorted == 0).astype(int)

    cdf_pos = np.cumsum(pos) / max(pos.sum(), 1)
    cdf_neg = np.cumsum(neg) / max(neg.sum(), 1)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(acc - conf)
    return float(ece)


def eval_proba(y_true, y_prob) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ks": float(ks_statistic(y_true, y_prob)),
        "ece_10bin": float(expected_calibration_error(y_true, y_prob, n_bins=10)),
    }
