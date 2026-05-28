"""
ECG Anomaly Detection - Main Entry Point

Steps:
  1. python main.py download     → Download MIT-BIH dataset from PhysioNet
  2. python main.py preprocess   → Extract and preprocess beat segments
  3. python main.py train-ae     → Train LSTM Autoencoder (anomaly detection)
  4. python main.py train-cnn    → Train CNN Classifier (arrhythmia classification)
  5. python main.py evaluate     → Evaluate models and generate plots
  6. python main.py all          → Run all steps in order
"""

import sys
import os


def download():
    from download_data import download_mitbih
    download_mitbih()


def preprocess():
    from src.preprocessing import build_dataset
    build_dataset()


def train_ae():
    from src.train_autoencoder import train
    train()


def train_cnn():
    from src.train_classifier import train
    train()


def evaluate():
    from src.evaluate import evaluate_classifier, evaluate_autoencoder
    from config import PLOTS_DIR
    os.makedirs(PLOTS_DIR, exist_ok=True)
    evaluate_classifier()
    evaluate_autoencoder()


STEPS = {
    'download': download,
    'preprocess': preprocess,
    'train-ae': train_ae,
    'train-cnn': train_cnn,
    'evaluate': evaluate,
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == 'all':
        for name, fn in STEPS.items():
            print(f"\n{'='*50}")
            print(f"  STEP: {name}")
            print('='*50)
            fn()
    elif cmd in STEPS:
        STEPS[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(STEPS)} , all")
        sys.exit(1)


if __name__ == '__main__':
    main()
