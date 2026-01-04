"""
=============================================================================
CONFIG - Cau hinh chung cho toan bo du an
=============================================================================
Muc dich: Dam bao tat ca 3 models (CNN, LSTM, Hybrid) dung chung:
- Cung tap features
- Cung cach chia train/val/test
- Cung hyperparameters co ban
- Uu tien GPU

Author: IoT Security Research Team
Date: 2026-01-03
"""

import torch
import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "backend" / "models"
OUTPUTS_DIR = BASE_DIR / "training" / "outputs"
LOGS_DIR = BASE_DIR / "training" / "logs"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"  # New: preprocessed npy files

# Tao thu muc neu chua ton tai
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DEVICE - Uu tien GPU
# =============================================================================
def get_device():
    """
    Lay device toi uu nhat (GPU > CPU)

    Returns:
        torch.device: cuda neu co GPU, nguoc lai cpu
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] Su dung: {gpu_name}")
        print(f"[GPU] VRAM: {gpu_memory:.1f} GB")

        # Toi uu GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        device = torch.device('cpu')
        print("[CPU] Khong tim thay GPU, su dung CPU")
        print("[WARNING] Training tren CPU se cham hon nhieu!")

    return device

DEVICE = get_device()

# =============================================================================
# FEATURES - 15 dac trung Bot-IoT
# =============================================================================
KEEP_FEATURES = [
    'pkts',     # Total packets in the flow
    'bytes',    # Total bytes in the flow
    'dur',      # Duration of the flow
    'mean',     # Mean packet size
    'stddev',   # Standard deviation of packet size
    'sum',      # Sum of packet sizes
    'min',      # Minimum packet size
    'max',      # Maximum packet size
    'spkts',    # Source to destination packets
    'dpkts',    # Destination to source packets
    'sbytes',   # Source to destination bytes
    'dbytes',   # Destination to source bytes
    'rate',     # Packets per second
    'srate',    # Source packets per second
    'drate',    # Destination packets per second
]

N_FEATURES = len(KEEP_FEATURES)  # 15
LABEL_COLUMN = 'attack'  # 0 = Normal, 1 = Attack

# =============================================================================
# DATA SPLIT - Dam bao dong nhat giua cac thanh vien
# =============================================================================
RANDOM_STATE = 42  # QUAN TRONG: Dung chung de ket qua co the tai tao
TEST_SIZE = 0.2    # 20% cho test
VAL_SIZE = 0.1     # 10% cua train cho validation

# =============================================================================
# SEQUENCE PARAMETERS
# =============================================================================
TIME_STEPS = 20    # So timesteps trong moi sequence
STRIDE = 1         # Buoc nhay khi tao sequences

# =============================================================================
# TRAINING HYPERPARAMETERS - GPU OPTIMIZED
# =============================================================================
BATCH_SIZE = 512           # Tăng từ 64 lên 512 để tận dụng GPU tốt hơn
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5        # L2 regularization

# Gradient Accumulation - Effective batch = BATCH_SIZE * ACCUMULATION_STEPS
ACCUMULATION_STEPS = 2     # Effective batch size = 512 * 2 = 1024

# Early Stopping
PATIENCE = 10              # So epochs cho phep khong cai thien
MIN_DELTA = 1e-4           # Nguong cai thien toi thieu

# DataLoader optimization
PREFETCH_FACTOR = 4        # Số batches prefetch trước
PERSISTENT_WORKERS = False # True nếu num_workers > 0

# =============================================================================
# MODEL NAMES
# =============================================================================
MODEL_NAMES = ['CNN', 'LSTM', 'Hybrid']
MODEL_NAMES_ALL = ['CNN', 'LSTM', 'Hybrid', 'Parallel']  # Include Parallel Hybrid

# =============================================================================
# PRINT CONFIG
# =============================================================================
def print_config():
    """In ra cau hinh hien tai"""
    print("=" * 60)
    print("CAU HINH DU AN")
    print("=" * 60)
    print(f"Device:        {DEVICE}")
    print(f"Features:      {N_FEATURES} dac trung")
    print(f"Time Steps:    {TIME_STEPS}")
    print(f"Batch Size:    {BATCH_SIZE}")
    print(f"Epochs:        {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Random State:  {RANDOM_STATE}")
    print(f"Test Size:     {TEST_SIZE * 100}%")
    print(f"Val Size:      {VAL_SIZE * 100}%")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
