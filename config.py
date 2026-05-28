import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# MIT-BIH dataset records
MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124', '200',
    '201', '202', '203', '205', '207', '208', '209', '210',
    '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234',
]

# Beat annotation mapping to class indices
BEAT_CLASSES = {
    'N': 0,  # Normal
    'L': 1,  # Left bundle branch block
    'R': 2,  # Right bundle branch block
    'V': 3,  # Premature ventricular contraction
    'A': 4,  # Atrial premature contraction
}

# Normal beat labels (used for autoencoder training)
NORMAL_LABELS = {'N', 'L', 'R', 'e', 'j'}

# Signal processing
SAMPLE_RATE = 360          # Hz
SEGMENT_LENGTH = 360       # samples per beat segment (1 second)
LEAD = 0                   # MLII lead index

# Autoencoder config
AE_HIDDEN_SIZE = 64
AE_LATENT_SIZE = 32
AE_NUM_LAYERS = 2
AE_DROPOUT = 0.2
AE_EPOCHS = 50
AE_BATCH_SIZE = 64
AE_LR = 1e-3
AE_ANOMALY_PERCENTILE = 95  # threshold percentile on reconstruction error

# Classifier config
CNN_EPOCHS = 30
CNN_BATCH_SIZE = 64
CNN_LR = 1e-3
CNN_NUM_CLASSES = len(BEAT_CLASSES)

# Train/test split
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
