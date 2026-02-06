"""Evaluation metrics for multiclass medMNIST classification."""

from __future__ import annotations

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Compute classification metrics from logits and labels.

    Parameters
    ----------
    logits : torch.Tensor
        Logits tensor with shape ``(num_samples, num_classes)``.
    labels : torch.Tensor
        Ground-truth labels with shape ``(num_samples,)`` or ``(num_samples, 1)``.

    Returns
    -------
    dict[str, float]
        Dictionary containing accuracy, balanced accuracy, macro F1, and
        micro/macro one-vs-rest AUC when computable.
    """
    probs = torch.softmax(logits.detach().cpu(), dim=1).numpy()
    preds = probs.argmax(axis=1)
    y_true = labels.view(-1).detach().cpu().numpy()

    acc = accuracy_score(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)

    auc_micro = float("nan")
    auc_macro = float("nan")
    try:
        num_classes = probs.shape[1]
        if num_classes == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
            auc_micro = float(auc)
            auc_macro = float(auc)
        elif num_classes > 2:
            y_true_ovr = label_binarize(y_true, classes=list(range(num_classes)))
            auc_micro = float(roc_auc_score(y_true_ovr, probs, average="micro", multi_class="ovr"))
            auc_macro = float(roc_auc_score(y_true_ovr, probs, average="macro", multi_class="ovr"))
    except ValueError:
        pass

    return {
        "acc": float(acc),
        "bal_acc": float(bal_acc),
        "f1": float(macro_f1),
        "auc_micro": auc_micro,
        "auc_macro": auc_macro,
    }
