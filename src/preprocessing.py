"""
Load raw MIT-BIH records, extract beat segments, and save as NumPy arrays.
"""

import os
import sys
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR,
    MITBIH_RECORDS, BEAT_CLASSES, NORMAL_LABELS,
    SEGMENT_LENGTH, LEAD, SAMPLE_RATE,
)


def bandpass_filter(signal: np.ndarray, low: float = 0.5, high: float = 45.0,
                    fs: float = SAMPLE_RATE) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def extract_beats(record_name: str, half_len: int = SEGMENT_LENGTH // 2):
    """Return (segments, labels) arrays for one MIT-BIH record."""
    path = os.path.join(DATA_RAW_DIR, record_name)
    try:
        record = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, 'atr')
    except Exception as e:
        print(f"  Skipping {record_name}: {e}", file=sys.stderr)
        return np.array([]), np.array([])

    signal = record.p_signal[:, LEAD].astype(np.float32)
    signal = bandpass_filter(signal)

    # Normalize per-record
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)

    segments, labels = [], []
    for idx, sym in zip(ann.sample, ann.symbol):
        if sym not in BEAT_CLASSES:
            continue
        start = idx - half_len
        end = idx + half_len
        if start < 0 or end > len(signal):
            continue
        segments.append(signal[start:end])
        labels.append(BEAT_CLASSES[sym])

    if not segments:
        return np.array([]), np.array([])

    return np.stack(segments).astype(np.float32), np.array(labels, dtype=np.int64)


def build_dataset():
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    all_segments, all_labels = [], []
    records = [r for r in MITBIH_RECORDS
               if os.path.exists(os.path.join(DATA_RAW_DIR, r + '.dat'))]

    if not records:
        print("No records found. Run download_data.py first.")
        return

    print(f"Processing {len(records)} records...")
    for rec in tqdm(records):
        segs, labs = extract_beats(rec)
        if len(segs):
            all_segments.append(segs)
            all_labels.append(labs)

    X = np.concatenate(all_segments, axis=0)   # (N, SEGMENT_LENGTH)
    y = np.concatenate(all_labels, axis=0)      # (N,)

    print(f"\nTotal beats extracted: {len(X)}")
    for cls_name, cls_idx in sorted(BEAT_CLASSES.items(), key=lambda x: x[1]):
        count = (y == cls_idx).sum()
        print(f"  Class {cls_idx} ({cls_name}): {count}")

    np.save(os.path.join(DATA_PROCESSED_DIR, 'X.npy'), X)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'y.npy'), y)
    print(f"\nSaved X.npy and y.npy to {DATA_PROCESSED_DIR}")


if __name__ == '__main__':
    build_dataset()
