"""
Train the LSTM Autoencoder on normal ECG beats only.
Usage: python -m src.train_autoencoder
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODELS_DIR, METRICS_DIR,
    AE_HIDDEN_SIZE, AE_LATENT_SIZE, AE_NUM_LAYERS, AE_DROPOUT,
    AE_EPOCHS, AE_LR, AE_ANOMALY_PERCENTILE, SEGMENT_LENGTH,
)
from src.dataset import make_autoencoder_loaders, make_anomaly_eval_loader
from src.models.autoencoder import LSTMAutoencoder


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x in loader:
        x = x.to(device)
        recon = model(x)
        total_loss += criterion(recon, x).item() * len(x)
    return total_loss / len(loader.dataset)


def train():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = make_autoencoder_loaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = LSTMAutoencoder(
        seq_len=SEGMENT_LENGTH,
        hidden_size=AE_HIDDEN_SIZE,
        latent_size=AE_LATENT_SIZE,
        num_layers=AE_NUM_LAYERS,
        dropout=AE_DROPOUT,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=AE_LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    best_val = float('inf')
    history = {'train': [], 'val': []}

    epoch_bar = tqdm(range(1, AE_EPOCHS + 1), desc='Autoencoder', unit='epoch')
    for epoch in epoch_bar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        epoch_bar.set_postfix(train=f'{train_loss:.6f}', val=f'{val_loss:.6f}',
                              best=f'{best_val:.6f}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'autoencoder_best.pt'))
            epoch_bar.set_postfix(train=f'{train_loss:.6f}', val=f'{val_loss:.6f}',
                                  best=f'{best_val:.6f}', saved='*')

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'autoencoder_last.pt'))
    with open(os.path.join(METRICS_DIR, 'autoencoder_history.json'), 'w') as f:
        json.dump(history, f)

    # Compute anomaly threshold from validation normal beats
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'autoencoder_best.pt'),
                                     map_location=device))
    model.eval()
    errors = []
    for x in val_loader:
        errors.append(model.reconstruction_error(x.to(device)).cpu().numpy())
    errors = np.concatenate(errors)
    threshold = float(np.percentile(errors, AE_ANOMALY_PERCENTILE))
    np.save(os.path.join(MODELS_DIR, 'ae_threshold.npy'), np.array([threshold]))
    print(f"\nAnomaly threshold (p{AE_ANOMALY_PERCENTILE}): {threshold:.6f}")
    print("Autoencoder training complete.")


if __name__ == '__main__':
    train()
