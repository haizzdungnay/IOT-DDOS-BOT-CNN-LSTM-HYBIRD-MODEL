"""
Bot-IoT Multi-Model Replay Detector
====================================
Real-time Traffic Replay Demo for 3 Deep Learning Models:
- CNN 1D
- LSTM
- Hybrid CNN-LSTM

Author: IoT Security Research Team
Date: 2026-01-02
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import time
import threading
from collections import deque
from datetime import datetime
import logging

logger = logging.getLogger('ReplayDetector')


# =============================================================================
# MODEL DEFINITIONS (Copy từ train_hybrid_standard.py)
# =============================================================================

class HybridCNNLSTM(nn.Module):
    """🔥 Hybrid CNN→LSTM Model (CHUẨN IEEE PAPERS) - NO POOLING"""
    
    def __init__(self, n_features: int, time_steps: int):
        super(HybridCNNLSTM, self).__init__()
        
        # CNN Block (NO POOLING!)
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout_cnn = nn.Dropout(0.3)
        
        # LSTM Block
        lstm_hidden = 64
        self.lstm = nn.LSTM(128, lstm_hidden, num_layers=2, 
                           batch_first=True, dropout=0.3)
        self.dropout_lstm = nn.Dropout(0.3)
        
        # Dense Block
        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc_out = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout_cnn(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout_cnn(x)
        
        # LSTM
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.dropout_lstm(x)
        
        # Classifier
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc_out(x)
        return x.squeeze(-1)


class CNN1D(nn.Module):
    """CNN-1D Model (Baseline)"""
    
    def __init__(self, n_features: int, time_steps: int):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)
        
        return x.squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM Model (Baseline)"""
    
    def __init__(self, n_features: int, time_steps: int):
        super(LSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(n_features, 128, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(128, 64, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(64, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout2(x)
        
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)
        
        return x.squeeze(-1)


# =============================================================================
# REPLAY DETECTOR CLASS
# =============================================================================

class ReplayDetector:
    """
    Multi-Model Replay Detector for Bot-IoT Dataset
    
    Features:
    - Loads 3 models simultaneously (CNN, LSTM, Hybrid)
    - Replays traffic from CSV file
    - Maintains sliding window buffer (20 timesteps)
    - Real-time predictions for all 3 models
    - Thread-safe operations
    """
    
    def __init__(self, models_dir='backend/models', data_dir='data'):
        """
        Initialize detector with 3 models
        
        Args:
            models_dir: Directory containing model weights
            data_dir: Directory containing CSV test data
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Bot-IoT Feature names (15 features)
        self.feature_names = [
            'pkts', 'bytes', 'dur', 'mean', 'stddev', 'sum', 'min', 'max',
            'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'srate', 'drate'
        ]
        
        # Sliding window buffer (20 timesteps)
        self.TIME_STEPS = 20
        self.buffer = deque(maxlen=self.TIME_STEPS)
        
        # Stats
        self.stats = {
            'total_packets': 0,
            'true_attacks': 0,
            'true_normal': 0,
            'consensus_attacks': 0,
            # Model predictions
            'cnn_correct': 0,
            'cnn_wrong': 0,
            'lstm_correct': 0,
            'lstm_wrong': 0,
            'hybrid_correct': 0,
            'hybrid_wrong': 0
        }
        
        # Thread control
        self.running = False
        self.replay_thread = None
        
        # Load scaler and models
        self._load_scaler()
        self._load_models()
        
        logger.info(f"✅ ReplayDetector initialized on {self.device}")
    
    def _load_scaler(self):
        """Load StandardScaler from preprocessing"""
        scaler_path = f"{self.models_dir}/scaler_standard.pkl"
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler loaded from {scaler_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load scaler: {e}")
            raise
    
    def _load_models(self):
        """Load 3 PyTorch models"""
        self.models = {}
        
        model_configs = {
            'CNN': (CNN1D, 'CNN_best.pt'),
            'LSTM': (LSTMModel, 'LSTM_best.pt'),
            'Hybrid': (HybridCNNLSTM, 'Hybrid_CNN_LSTM_best.pt')
        }
        
        for name, (ModelClass, filename) in model_configs.items():
            try:
                model_path = f"{self.models_dir}/{filename}"
                model = ModelClass(n_features=15, time_steps=20)
                
                # Load weights
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                
                self.models[name] = model
                logger.info(f"✅ {name} model loaded from {model_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {name} model: {e}")
                raise
        
        logger.info(f"✅ All 3 models loaded successfully!")
    
    def _prepare_csv_data(self, csv_path):
        """
        Load and prepare CSV data for replay
        
        Returns:
            features: numpy array (n_samples, 15)
            labels: numpy array (n_samples,)
            metadata: dict with additional info
        """
        logger.info(f"📂 Loading CSV from {csv_path}...")
        
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            
            # Extract features and labels
            features = df[self.feature_names].values
            labels = df['attack'].values
            
            # Metadata (for display)
            metadata = {
                'stime': df['stime'].values if 'stime' in df.columns else None,
                'srcip': df['srcip'].values if 'srcip' in df.columns else None,
                'dstip': df['dstip'].values if 'dstip' in df.columns else None,
            }
            
            logger.info(f"✅ Loaded {len(features)} samples")
            logger.info(f"   Normal: {(labels == 0).sum():,} | Attack: {(labels == 1).sum():,}")
            
            return features, labels, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load CSV: {e}")
            raise
    
    def predict(self, sequence):
        """
        Predict using all 3 models
        
        Args:
            sequence: numpy array (20, 15) - sequence of 20 timesteps
        
        Returns:
            dict: {
                'CNN': {'pred': 0/1, 'prob': float, 'confidence': float},
                'LSTM': {...},
                'Hybrid': {...}
            }
        """
        # Convert to tensor
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # (1, 20, 15)
        
        results = {}
        
        with torch.no_grad():
            for name, model in self.models.items():
                try:
                    # Forward pass
                    logits = model(seq_tensor)
                    prob = torch.sigmoid(logits).item()
                    pred = 1 if prob > 0.5 else 0
                    confidence = prob if pred == 1 else (1 - prob)
                    
                    results[name] = {
                        'pred': pred,
                        'prob': prob,
                        'confidence': confidence
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Prediction error for {name}: {e}")
                    results[name] = {'pred': 0, 'prob': 0.5, 'confidence': 0.5}
        
        return results
    
    def start_replay(self, csv_filename='demo_test.csv', speed=0.1, callback=None):
        """
        Start replaying traffic from CSV
        
        Args:
            csv_filename: CSV file to replay
            speed: Delay between packets (seconds)
            callback: Function to call with results (for web updates)
        """
        if self.running:
            logger.warning("⚠️  Replay already running!")
            return
        
        csv_path = f"{self.data_dir}/{csv_filename}"
        
        # Load data
        features, labels, metadata = self._prepare_csv_data(csv_path)
        
        # Start replay thread
        self.running = True
        self.replay_thread = threading.Thread(
            target=self._replay_loop,
            args=(features, labels, metadata, speed, callback),
            daemon=True
        )
        self.replay_thread.start()
        
        logger.info(f"🚀 Replay started: {csv_filename} (speed={speed}s)")
    
    def _replay_loop(self, features, labels, metadata, speed, callback):
        """
        Main replay loop (runs in separate thread)
        """
        self.buffer.clear()
        self.stats = {k: 0 for k in self.stats}
        
        for i in range(len(features)):
            if not self.running:
                break
            
            # Get current row
            row = features[i]
            true_label = int(labels[i])
            
            # Scale features
            row_scaled = self.scaler.transform([row])[0]
            
            # Add to buffer
            self.buffer.append(row_scaled)
            
            # Update stats
            self.stats['total_packets'] += 1
            if true_label == 1:
                self.stats['true_attacks'] += 1
            else:
                self.stats['true_normal'] += 1
            
            # Predict when buffer is full
            predictions = None
            if len(self.buffer) == self.TIME_STEPS:
                # Create sequence
                sequence = np.array(list(self.buffer))  # (20, 15)
                
                # Predict with all 3 models
                predictions = self.predict(sequence)
                
                # Track correct/wrong for each model
                for model_name, pred_data in predictions.items():
                    pred = pred_data['pred']
                    prefix = model_name.lower()
                    
                    if pred == true_label:
                        self.stats[f'{prefix}_correct'] += 1
                    else:
                        self.stats[f'{prefix}_wrong'] += 1
                
                # Consensus (2 out of 3 agree)
                attack_votes = sum([p['pred'] for p in predictions.values()])
                if attack_votes >= 2:
                    self.stats['consensus_attacks'] += 1
            
            # Callback for web updates
            if callback and predictions:
                callback_data = {
                    'packet_id': i,
                    'timestamp': time.time(),
                    'true_label': true_label,
                    'predictions': predictions,
                    'stats': self.stats.copy(),
                    'metadata': {
                        'srcip': metadata['srcip'][i] if metadata['srcip'] is not None else 'N/A',
                        'dstip': metadata['dstip'][i] if metadata['dstip'] is not None else 'N/A',
                    }
                }
                callback(callback_data)
            
            # Sleep to simulate real-time
            time.sleep(speed)
        
        logger.info(f"✅ Replay finished: {self.stats['total_packets']} packets processed")
        self.running = False
    
    def stop_replay(self):
        """Stop replay"""
        if not self.running:
            return
        
        self.running = False
        if self.replay_thread:
            self.replay_thread.join(timeout=5)
        
        logger.info("⏹️  Replay stopped")
    
    def get_stats(self):
        """Get current statistics"""
        return self.stats.copy()
