"""
Bot-IoT Multi-Model Replay Detector
====================================
Real-time Traffic Replay Demo for Deep Learning Models:
- CNN 1D
- LSTM
- Hybrid CNN-LSTM
- Parallel Hybrid

Author: IoT Security Research Team
Date: 2026-01-03
"""

import pandas as pd
import numpy as np
import torch
import joblib
import time
import threading
from collections import deque
from pathlib import Path
import logging
import sys
import os

# Add training directory to path for model imports
BACKEND_DIR = Path(__file__).parent
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "training"))

# Import models from training module (avoid code duplication)
try:
    from models import CNN1D, LSTMModel, HybridCNNLSTM, ParallelHybridCNNLSTM, get_model
    MODELS_IMPORTED = True
except ImportError:
    MODELS_IMPORTED = False
    logging.warning("Could not import models from training module, using fallback")

logger = logging.getLogger('ReplayDetector')


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
        """
        Load PyTorch models từ training module

        Model configs:
        - CNN: CNN1D → CNN_best.pt
        - LSTM: LSTMModel → LSTM_best.pt
        - Hybrid: HybridCNNLSTM → Hybrid_CNN_LSTM_best.pt

        Note: Parallel_Hybrid và Hybrid dùng chung weights (Hybrid_CNN_LSTM_best.pt)
        """
        self.models = {}

        # Model configurations: name -> (model_type, weight_file)
        model_configs = {
            'CNN': ('CNN', 'CNN_best.pt'),
            'LSTM': ('LSTM', 'LSTM_best.pt'),
            'Hybrid': ('Hybrid', 'Hybrid_CNN_LSTM_best.pt'),
        }

        for display_name, (model_type, filename) in model_configs.items():
            model_path = Path(self.models_dir) / filename

            if not model_path.exists():
                logger.warning(f"⚠️  Model file not found: {model_path}")
                continue

            try:
                # Use get_model from training module if available
                if MODELS_IMPORTED:
                    model = get_model(model_type, n_features=15, time_steps=20)
                else:
                    # Fallback - should not happen in normal use
                    raise ImportError("Models not imported")

                # Load weights
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()

                self.models[display_name] = model
                logger.info(f"✅ {display_name} model loaded from {model_path}")

            except Exception as e:
                logger.error(f"❌ Failed to load {display_name} model: {e}")
                raise

        logger.info(f"✅ Loaded {len(self.models)} models successfully!")
    
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
