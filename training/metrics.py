from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)


# Computes a full suite of classification metrics from raw probability scores.
# FNR (false negative rate) is the primary IDS metric because missing attacks
# is more costly than false positives. Also returns AUC-PR and AUC-ROC.
def compute_metrics(
    y_true: list | np.ndarray,
    y_scores: list | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fnr = fn / (fn + tp + 1e-9)
    fpr = fp / (fp + tn + 1e-9)

    return dict(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        fnr=float(fnr),
        fpr=float(fpr),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        auc_pr=float(average_precision_score(y_true, y_scores)),
        auc_roc=float(roc_auc_score(y_true, y_scores)),
    )
