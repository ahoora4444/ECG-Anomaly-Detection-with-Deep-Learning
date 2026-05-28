"""
LSTM Autoencoder for unsupervised ECG anomaly detection.
High reconstruction error → anomaly.
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, latent_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        # x: (B, 1, L) → (B, L, 1)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        latent = self.fc(out[:, -1, :])  # last time step
        return latent


class LSTMDecoder(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int, output_size: int,
                 seq_len: int, num_layers: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, latent):
        # Repeat latent across time steps
        h = self.fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(h)
        recon = self.out(out)              # (B, L, 1)
        return recon.permute(0, 2, 1)     # (B, 1, L)


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, hidden_size: int = 64, latent_size: int = 32,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = LSTMEncoder(1, hidden_size, latent_size, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_size, hidden_size, 1, seq_len, num_layers, dropout)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error."""
        with torch.no_grad():
            recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))
