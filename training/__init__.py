"""
Training module cho Bot-IoT DDoS Detection

Modules:
- config: Cau hinh chung (GPU, features, hyperparameters)
- data_loader: Load va xu ly du lieu
- models: Dinh nghia 3 models (CNN, LSTM, Hybrid)
- trainer: Training class
- evaluate: Danh gia models
- visualize: Ve bieu do

Usage:
    # Train all models
    python train_all.py --data path/to/botiot.csv

    # Evaluate models
    python evaluate.py

    # Generate plots
    python visualize.py
"""

from .config import (
    DEVICE, KEEP_FEATURES, N_FEATURES, TIME_STEPS,
    BATCH_SIZE, EPOCHS, MODEL_NAMES
)
from .models import CNN1D, LSTMModel, HybridCNNLSTM, get_model
from .data_loader import load_and_preprocess_data, load_test_set_only
from .trainer import Trainer, train_model
