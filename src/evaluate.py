"""
Evaluate trained models and generate plots.
Usage: python -m src.evaluate
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score,
    matthews_corrcoef, average_precision_score,
    precision_recall_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, PLOTS_DIR, METRICS_DIR,
    BEAT_CLASSES, SEGMENT_LENGTH, CNN_NUM_CLASSES,
    AE_HIDDEN_SIZE, AE_LATENT_SIZE, AE_NUM_LAYERS, AE_DROPOUT,
)
from src.dataset import make_classifier_loaders, make_anomaly_eval_loader
from src.models.autoencoder import LSTMAutoencoder
from src.models.classifier import ECGClassifier


# ── helpers ──────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _class_names():
    return [f"{v}({k})" for k, v in sorted(BEAT_CLASSES.items(), key=lambda x: x[1])]


def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _print_separator(title=''):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'='*pad} {title} {'='*(width - pad - len(title) - 2)}")
    else:
        print('=' * width)


# ── training history ──────────────────────────────────────────────────────────

def plot_training_history(history_path: str, title: str, out_path: str):
    with open(history_path) as f:
        h = json.load(f)

    train_keys = [k for k in h if 'train' in k or (k == 'train')]
    val_keys   = [k for k in h if 'val' in k   or (k == 'val')]
    pairs = []
    for tk in train_keys:
        vk = tk.replace('train', 'val')
        if vk in h:
            pairs.append((tk, vk))
        else:
            pairs.append((tk, None))
    if not pairs:
        pairs = [(k, None) for k in list(h.keys())[:2]]

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (tk, vk) in zip(axes, pairs):
        epochs = range(1, len(h[tk]) + 1)
        ax.plot(epochs, h[tk], label=tk, linewidth=1.8)
        if vk and vk in h:
            ax.plot(epochs, h[vk], label=vk, linewidth=1.8, linestyle='--')
        ax.set_title(tk.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, out_path)


# ── classifier evaluations ────────────────────────────────────────────────────

def _collect_classifier_outputs(model, val_loader, device):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, class_names, out_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues', vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title('Confusion Matrix (normalized)')
    axes[1].set_ylabel('True')
    axes[1].set_xlabel('Predicted')

    fig.suptitle('Classifier Confusion Matrices', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_per_class_metrics(labels, preds, class_names, out_path):
    precision = precision_score(labels, preds, average=None, zero_division=0)
    recall    = recall_score(labels, preds, average=None, zero_division=0)
    f1        = f1_score(labels, preds, average=None, zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label='Precision', color='steelblue', alpha=0.85)
    ax.bar(x,         recall,    width, label='Recall',    color='darkorange', alpha=0.85)
    ax.bar(x + width, f1,        width, label='F1-Score',  color='green',      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision / Recall / F1', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.01, f'{p:.2f}', ha='center', fontsize=7)
        ax.text(i,         r + 0.01, f'{r:.2f}', ha='center', fontsize=7)
        ax.text(i + width, f + 0.01, f'{f:.2f}', ha='center', fontsize=7)

    plt.tight_layout()
    _savefig(fig, out_path)


def plot_sample_beats(val_loader, labels_arr, preds_arr, class_names, out_path):
    """Show one correctly-classified sample beat per class."""
    n_classes = len(class_names)
    samples = {}

    dataset = val_loader.dataset
    for i in range(len(dataset)):
        x, y = dataset[i]
        y = int(y)
        if y not in samples and preds_arr[i] == y:
            samples[y] = x.numpy().squeeze()
        if len(samples) == n_classes:
            break

    if not samples:
        return

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 3), sharey=True)
    if n_classes == 1:
        axes = [axes]

    for cls_idx, ax in enumerate(axes):
        if cls_idx in samples:
            ax.plot(samples[cls_idx], linewidth=1.2, color='steelblue')
        ax.set_title(class_names[cls_idx], fontsize=10)
        ax.set_xlabel('Sample')
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('Amplitude (norm.)')
    fig.suptitle('Sample ECG Beats per Class (correctly classified)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, out_path)


def _compute_and_print_classifier_metrics(labels, preds, probs, class_names):
    acc     = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro',    zero_division=0)
    wgt_f1   = f1_score(labels, preds, average='weighted', zero_division=0)
    mcc      = matthews_corrcoef(labels, preds)

    try:
        if probs.shape[1] == 2:
            auc_ovr = roc_auc_score(labels, probs[:, 1])
        else:
            auc_ovr = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except Exception:
        auc_ovr = float('nan')

    _print_separator('Classifier Performance Summary')
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1         : {macro_f1:.4f}")
    print(f"  Weighted F1      : {wgt_f1:.4f}")
    print(f"  Matthews CC (MCC): {mcc:.4f}")
    print(f"  AUC-ROC (OvR)    : {auc_ovr:.4f}")
    print()
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    metrics = {
        'accuracy':    round(acc, 6),
        'macro_f1':    round(macro_f1, 6),
        'weighted_f1': round(wgt_f1, 6),
        'mcc':         round(mcc, 6),
        'auc_roc_ovr': round(float(auc_ovr), 6),
    }
    return metrics


def evaluate_classifier():
    device = get_device()
    model_path = os.path.join(MODELS_DIR, 'classifier_best.pt')
    if not os.path.exists(model_path):
        print("Classifier model not found. Train first.")
        return

    model = ECGClassifier(seq_len=SEGMENT_LENGTH, num_classes=CNN_NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    _, val_loader = make_classifier_loaders()
    labels, preds, probs = _collect_classifier_outputs(model, val_loader, device)

    class_names = _class_names()

    metrics = _compute_and_print_classifier_metrics(labels, preds, probs, class_names)

    _print_separator('Classifier Plots')
    plot_confusion_matrix(labels, preds, class_names,
                          os.path.join(PLOTS_DIR, 'classifier_confusion_matrix.png'))
    plot_per_class_metrics(labels, preds, class_names,
                           os.path.join(PLOTS_DIR, 'classifier_per_class_metrics.png'))
    plot_sample_beats(val_loader, labels, preds, class_names,
                      os.path.join(PLOTS_DIR, 'classifier_sample_beats.png'))

    history_path = os.path.join(METRICS_DIR, 'classifier_history.json')
    if os.path.exists(history_path):
        plot_training_history(history_path, 'Classifier Training',
                              os.path.join(PLOTS_DIR, 'classifier_history.png'))

    out = os.path.join(METRICS_DIR, 'classifier_metrics.json')
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON: {out}")


# ── autoencoder evaluations ───────────────────────────────────────────────────

def plot_roc_curve(labels, errors, auc, out_path):
    fpr, tpr, _ = roc_curve(labels, errors)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f'AUC-ROC = {auc:.4f}', linewidth=2, color='steelblue')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Autoencoder ROC Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_pr_curve(labels, errors, ap, out_path):
    precision, recall, _ = precision_recall_curve(labels, errors)
    baseline = labels.mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f'AUC-PR = {ap:.4f}', linewidth=2, color='darkorange')
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1,
               label=f'Baseline = {baseline:.3f}')
    ax.fill_between(recall, precision, alpha=0.1, color='darkorange')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Autoencoder Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_error_distribution(errors, labels, threshold, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(errors[labels == 0], bins=100, alpha=0.65, label='Normal',  density=True, color='steelblue')
    ax.hist(errors[labels == 1], bins=100, alpha=0.65, label='Anomaly', density=True, color='tomato')
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                   label=f'Threshold = {threshold:.5f}')
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plot
    ax = axes[1]
    data_to_plot = [errors[labels == 0], errors[labels == 1]]
    bp = ax.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    colors = ['steelblue', 'tomato']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    if threshold is not None:
        ax.axhline(threshold, color='black', linestyle='--', linewidth=1.5,
                   label=f'Threshold = {threshold:.5f}')
        ax.legend()
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.set_title('Error Box Plot by Class')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Autoencoder Reconstruction Error Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_scatter_errors(errors, labels, threshold, out_path):
    """Scatter plot of per-sample reconstruction errors colored by true label."""
    n = len(errors)
    idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(idx[labels == 0], errors[labels == 0], s=3, alpha=0.4,
               color='steelblue', label='Normal', rasterized=True)
    ax.scatter(idx[labels == 1], errors[labels == 1], s=3, alpha=0.6,
               color='tomato', label='Anomaly', rasterized=True)
    if threshold is not None:
        ax.axhline(threshold, color='black', linestyle='--', linewidth=1.5,
                   label=f'Threshold = {threshold:.5f}')
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.set_title('Per-Sample Reconstruction Error', fontsize=13, fontweight='bold')
    ax.legend(markerscale=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, out_path)


def _compute_and_print_ae_metrics(labels, errors, threshold):
    auc_roc = roc_auc_score(labels, errors)
    auc_pr  = average_precision_score(labels, errors)

    _print_separator('Autoencoder Performance Summary')
    print(f"  AUC-ROC          : {auc_roc:.4f}")
    print(f"  AUC-PR           : {auc_pr:.4f}")

    metrics = {'auc_roc': round(float(auc_roc), 6), 'auc_pr': round(float(auc_pr), 6)}

    if threshold is not None:
        preds  = (errors > threshold).astype(int)
        f1     = f1_score(labels, preds, zero_division=0)
        prec   = precision_score(labels, preds, zero_division=0)
        rec    = recall_score(labels, preds, zero_division=0)
        acc    = accuracy_score(labels, preds)
        mcc    = matthews_corrcoef(labels, preds)
        print(f"  Threshold        : {threshold:.6f}")
        print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision        : {prec:.4f}")
        print(f"  Recall           : {rec:.4f}")
        print(f"  F1-Score         : {f1:.4f}")
        print(f"  Matthews CC (MCC): {mcc:.4f}")
        metrics.update({
            'threshold':  round(float(threshold), 8),
            'accuracy':   round(acc, 6),
            'precision':  round(prec, 6),
            'recall':     round(rec, 6),
            'f1':         round(f1, 6),
            'mcc':        round(float(mcc), 6),
        })

    return metrics


def evaluate_autoencoder():
    device = get_device()
    model_path     = os.path.join(MODELS_DIR, 'autoencoder_best.pt')
    threshold_path = os.path.join(MODELS_DIR, 'ae_threshold.npy')

    if not os.path.exists(model_path):
        print("Autoencoder model not found. Train first.")
        return

    model = LSTMAutoencoder(
        seq_len=SEGMENT_LENGTH,
        hidden_size=AE_HIDDEN_SIZE,
        latent_size=AE_LATENT_SIZE,
        num_layers=AE_NUM_LAYERS,
        dropout=AE_DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    threshold = float(np.load(threshold_path)[0]) if os.path.exists(threshold_path) else None

    eval_loader = make_anomaly_eval_loader()
    errors, labels = [], []
    with torch.no_grad():
        for x, y in eval_loader:
            err = model.reconstruction_error(x.to(device)).cpu().numpy()
            errors.extend(err)
            labels.extend(y.numpy())

    errors = np.array(errors)
    labels = np.array(labels)

    metrics = _compute_and_print_ae_metrics(labels, errors, threshold)

    _print_separator('Autoencoder Plots')
    plot_roc_curve(labels, errors, metrics['auc_roc'],
                   os.path.join(PLOTS_DIR, 'autoencoder_roc.png'))
    plot_pr_curve(labels, errors, metrics['auc_pr'],
                  os.path.join(PLOTS_DIR, 'autoencoder_pr_curve.png'))
    plot_error_distribution(errors, labels, threshold,
                            os.path.join(PLOTS_DIR, 'autoencoder_error_dist.png'))
    plot_scatter_errors(errors, labels, threshold,
                        os.path.join(PLOTS_DIR, 'autoencoder_error_scatter.png'))

    history_path = os.path.join(METRICS_DIR, 'autoencoder_history.json')
    if os.path.exists(history_path):
        plot_training_history(history_path, 'Autoencoder Training',
                              os.path.join(PLOTS_DIR, 'autoencoder_history.png'))

    out = os.path.join(METRICS_DIR, 'autoencoder_metrics.json')
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON: {out}")


if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    evaluate_classifier()
    evaluate_autoencoder()
