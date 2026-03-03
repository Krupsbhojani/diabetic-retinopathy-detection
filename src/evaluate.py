"""
evaluate.py — Evaluation utilities for Diabetic Retinopathy Detection
QWK, confusion matrix, ROC curves, per-class AUC.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from pathlib import Path

LABEL_MAP   = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}
CLASS_NAMES = [LABEL_MAP[i] for i in range(5)]


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa — the primary APTOS 2019 metric."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def print_full_report(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    """Print accuracy, QWK, AUC and per-class classification report."""
    accuracy  = (y_pred == y_true).mean()
    qwk       = compute_qwk(y_true, y_pred)
    onehot    = np.eye(5)[y_true]
    macro_auc = roc_auc_score(onehot, y_prob, multi_class='ovr', average='macro')

    print('=' * 50)
    print('  EVALUATION REPORT')
    print('=' * 50)
    print(f'  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)')
    print(f'  QWK       : {qwk:.4f}')
    print(f'  Macro AUC : {macro_auc:.4f}')
    print('=' * 50)
    print('\nPer-class report:')
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """Plot raw and row-normalized confusion matrices side by side."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ['Confusion Matrix (Counts)', 'Confusion Matrix (Normalized)'],
        ['d', '.2f']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, linewidths=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
    """Plot per-class OvR ROC curves."""
    onehot = np.eye(5)[y_true]
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    plt.figure(figsize=(9, 7))

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(onehot[:, i], y_prob[:, i])
        auc = roc_auc_score(onehot[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — One-vs-Rest', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
