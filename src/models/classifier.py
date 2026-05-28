"""
1D CNN classifier for multi-class arrhythmia detection.
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, stride: int = 1,
                 pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.block(x)


class ECGClassifier(nn.Module):
    def __init__(self, seq_len: int, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  32, kernel=7, pool=2),
            ConvBlock(32, 64, kernel=5, pool=2),
            ConvBlock(64, 128, kernel=5, pool=2),
            ConvBlock(128, 256, kernel=3, pool=2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)
