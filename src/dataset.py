import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PROCESSED_DIR, TRAIN_RATIO, RANDOM_SEED, AE_BATCH_SIZE, CNN_BATCH_SIZE


class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        # X: (N, L) → (N, 1, L) for Conv1d
        self.X = torch.tensor(X[:, None, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def load_processed():
    X = np.load(os.path.join(DATA_PROCESSED_DIR, 'X.npy'))
    y = np.load(os.path.join(DATA_PROCESSED_DIR, 'y.npy'))
    return X, y


def make_classifier_loaders(batch_size: int = CNN_BATCH_SIZE):
    X, y = load_processed()
    dataset = ECGDataset(X, y)
    n_train = int(len(dataset) * TRAIN_RATIO)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def make_autoencoder_loaders(batch_size: int = AE_BATCH_SIZE):
    """Train autoencoder only on normal beats (class 0 = N)."""
    X, y = load_processed()
    normal_mask = (y == 0)
    X_normal = X[normal_mask]

    dataset = ECGDataset(X_normal)
    n_train = int(len(dataset) * TRAIN_RATIO)
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def make_anomaly_eval_loader(batch_size: int = 512):
    """Return a loader over all data (normal + anomalous) for evaluation."""
    X, y = load_processed()
    # Binary label: 0 = normal, 1 = anomaly
    y_binary = (y != 0).astype(np.int64)
    dataset = ECGDataset(X, y_binary)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
