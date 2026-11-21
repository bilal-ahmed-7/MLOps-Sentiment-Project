"""Evaluation metrics, quality gate and plots."""
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred) -> dict:
    """Binary classification metrics (positive class = 1)."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def passes_quality_gate(metrics: dict, min_f1: float) -> bool:
    """True if the model is good enough to proceed to registration/deployment."""
    return metrics["f1"] >= min_f1


def confusion_matrix_figure(y_true, y_pred, labels=("negative", "positive")) -> Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = Figure(figsize=(4, 4))
    ax = fig.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=list(labels))
    ax.set_yticks([0, 1], labels=list(labels))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), value in [((i, j), cm[i, j]) for i in range(2) for j in range(2)]:
        ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout()
    return fig
