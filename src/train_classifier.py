"""
Train 1D CNN classifier for arrhythmia classification.
Usage: python -m src.train_classifier
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, METRICS_DIR,
    CNN_EPOCHS, CNN_LR, CNN_NUM_CLASSES, SEGMENT_LENGTH,
)
from src.dataset import make_classifier_loaders
from src.models.classifier import ECGClassifier


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(x)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(x)
    return total_loss / total, correct / total


def train():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = make_classifier_loaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = ECGClassifier(seq_len=SEGMENT_LENGTH, num_classes=CNN_NUM_CLASSES).to(device)
    optimizer = Adam(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    epoch_bar = tqdm(range(1, CNN_EPOCHS + 1), desc='Classifier', unit='epoch')
    for epoch in epoch_bar:
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        epoch_bar.set_postfix(tr_loss=f'{tr_loss:.4f}', tr_acc=f'{tr_acc:.4f}',
                              va_loss=f'{va_loss:.4f}', va_acc=f'{va_acc:.4f}')

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'classifier_best.pt'))
            epoch_bar.set_postfix(tr_loss=f'{tr_loss:.4f}', tr_acc=f'{tr_acc:.4f}',
                                  va_loss=f'{va_loss:.4f}', va_acc=f'{va_acc:.4f}',
                                  saved='*')

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'classifier_last.pt'))
    with open(os.path.join(METRICS_DIR, 'classifier_history.json'), 'w') as f:
        json.dump(history, f)

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print("Classifier training complete.")


if __name__ == '__main__':
    train()
