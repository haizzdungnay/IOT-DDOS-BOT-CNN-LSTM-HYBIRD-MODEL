"""
Backend API Routes for Advanced Dashboard
==========================================
Provides APIs for:
- Model Evaluation
- Training Management
- History & Reports
- Dataset Management
- Compare Results

Author: IoT Security Research Team
Date: 2026-01-03
"""

import os
import sys
import json
import pickle
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Blueprint, jsonify, request
import numpy as np
import torch

# Add training path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training'))

# Import config
try:
    from training.config import DEVICE, N_FEATURES, TIME_STEPS, PROCESSED_DATA_DIR
except ImportError:
    # Fallback defaults
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N_FEATURES = 15
    TIME_STEPS = 20
    PROCESSED_DATA_DIR = 'processed_data'

api = Blueprint('api', __name__)

# =============================================================================
# CONSTANTS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'backend' / 'models'
TRAINING_OUTPUT_DIR = BASE_DIR / 'training' / 'outputs'
LOGS_DIR = BASE_DIR / 'training' / 'logs'
HISTORY_FILE = BASE_DIR / 'data' / 'training_history.json'
DATA_DIR = BASE_DIR / 'data'

# Training state
training_state = {
    'running': False,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_model': '',
    'logs': [],
    'start_time': None,
    'pid': None
}

# Custom dataset path
custom_dataset_path = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_training_history():
    """Load training history from JSON file"""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'trainings': [], 'evaluations': []}


def save_training_history(history):
    """Save training history to JSON file"""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)


def get_model_metrics(model_path, model_name):
    """Get evaluation metrics for a single model"""
    try:
        # Check if evaluation results exist
        eval_file = LOGS_DIR / 'evaluation_results_processed.json'
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                results = json.load(f)
                if model_name in results:
                    return results[model_name]
        
        # Fallback to old results
        eval_file = LOGS_DIR / 'evaluation_results.json'
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                results = json.load(f)
                if model_name in results:
                    return results[model_name]
        
        return None
    except Exception as e:
        return None


def format_metrics_for_display(metrics):
    """Format metrics for frontend display"""
    if not metrics:
        return None
    
    return {
        'accuracy': round(metrics.get('accuracy', 0) * 100, 2),
        'precision': round(metrics.get('precision', 0) * 100, 2),
        'recall': round(metrics.get('recall', 0) * 100, 2),
        'f1_score': round(metrics.get('f1_score', 0) * 100, 2),
        'fpr': round(metrics.get('fpr', 0) * 100, 2),
        'fnr': round(metrics.get('fnr', 0) * 100, 2),
        'roc_auc': round(metrics.get('roc_auc', 0), 4),
        'confusion_matrix': metrics.get('confusion_matrix', [[0,0],[0,0]])
    }


# =============================================================================
# MODEL EVALUATION APIs
# =============================================================================

def detect_model_name(filename: str) -> str:
    """Extract model name from filename like CNN_best.pt, LSTM_v2.pt, etc."""
    name = filename.replace('.pt', '').replace('.pth', '')
    # Remove common suffixes
    for suffix in ['_best', '_final', '_v1', '_v2', '_v3', '_last']:
        name = name.replace(suffix, '')
    # Handle special cases
    if 'Hybrid_CNN_LSTM' in name or 'CNN_LSTM' in name:
        return 'Hybrid'
    if 'Parallel_Hybrid' in name or 'Parallel' in name:
        return 'Parallel'
    if 'LSTM_CNN' in name:
        return 'LSTM_CNN'
    return name.split('_')[0]  # Take first part as model name

def scan_models_directory():
    """Scan models directory and return available models dynamically"""
    models = {}
    
    if not MODELS_DIR.exists():
        return models
    
    # Scan for .pt and .pth files
    for filepath in MODELS_DIR.glob('*.pt'):
        filename = filepath.name
        # Skip scaler files
        if 'scaler' in filename.lower():
            continue
        
        model_name = detect_model_name(filename)
        
        # If model already exists, prefer 'best' version
        if model_name in models:
            if 'best' in filename.lower():
                models[model_name] = filename
        else:
            models[model_name] = filename
    
    # Also scan .pth files
    for filepath in MODELS_DIR.glob('*.pth'):
        filename = filepath.name
        if 'scaler' in filename.lower():
            continue
        
        model_name = detect_model_name(filename)
        if model_name not in models:
            models[model_name] = filename
    
    return models

@api.route('/api/models/list', methods=['GET'])
def list_models():
    """Get list of available models with their status - DYNAMIC SCANNING"""
    models = []
    
    # Dynamically scan models directory
    model_files = scan_models_directory()
    
    for name, filename in sorted(model_files.items()):
        path = MODELS_DIR / filename
        exists = path.exists()
        
        # Get metrics if available
        metrics = get_model_metrics(str(path), name) if exists else None
        
        models.append({
            'name': name,
            'filename': filename,
            'path': str(path),
            'exists': exists,
            'size_mb': round(path.stat().st_size / (1024*1024), 2) if exists else 0,
            'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None,
            'metrics': format_metrics_for_display(metrics)
        })
    
    return jsonify({
        'status': 'ok',
        'models': models,
        'device': str(DEVICE)
    })


@api.route('/api/models/evaluate', methods=['POST'])
def evaluate_models():
    """Run evaluation on selected models"""
    global training_state
    
    if training_state['running']:
        return jsonify({'status': 'error', 'message': 'Training/Evaluation already in progress'}), 400
    
    data = request.get_json()
    models = data.get('models', ['CNN', 'LSTM', 'Hybrid', 'Parallel'])
    
    def run_evaluation():
        global training_state
        training_state['running'] = True
        training_state['current_model'] = 'Evaluating...'
        training_state['logs'] = []
        
        try:
            # Run evaluate_processed.py
            cmd = [
                sys.executable, 
                str(BASE_DIR / 'training' / 'evaluate_processed.py'),
                '--models', *models,
                '--model-dir', str(MODELS_DIR)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(BASE_DIR)  # Chạy từ thư mục gốc project
            )
            
            training_state['pid'] = process.pid
            
            for line in process.stdout:
                training_state['logs'].append(line.strip())
                if 'ĐÁNH GIÁ MODEL:' in line:
                    training_state['current_model'] = line.split(':')[-1].strip()
            
            process.wait()
            
            # Save to history
            history = load_training_history()
            history['evaluations'].append({
                'id': len(history['evaluations']) + 1,
                'timestamp': datetime.now().isoformat(),
                'models': models,
                'type': 'evaluation',
                'status': 'completed' if process.returncode == 0 else 'failed'
            })
            save_training_history(history)
            
        except Exception as e:
            training_state['logs'].append(f"Error: {str(e)}")
        finally:
            training_state['running'] = False
            training_state['pid'] = None
    
    thread = threading.Thread(target=run_evaluation)
    thread.start()
    
    return jsonify({
        'status': 'ok',
        'message': f'Evaluation started for models: {models}'
    })


@api.route('/api/models/metrics', methods=['GET'])
def get_all_metrics():
    """Get evaluation metrics for all models"""
    results = {}
    
    # Try processed results first
    eval_file = LOGS_DIR / 'evaluation_results_processed.json'
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            data = json.load(f)
            for model_name, metrics in data.items():
                results[model_name] = format_metrics_for_display(metrics)
    
    return jsonify({
        'status': 'ok',
        'metrics': results,
        'source': str(eval_file) if eval_file.exists() else None
    })


# =============================================================================
# TRAINING MANAGEMENT APIs
# =============================================================================

@api.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training"""
    global training_state
    
    if training_state['running']:
        return jsonify({'status': 'error', 'message': 'Training already in progress'}), 400
    
    data = request.get_json()
    models = data.get('models', ['CNN', 'LSTM', 'Hybrid', 'Parallel'])
    epochs = data.get('epochs', 30)
    batch_size = data.get('batch_size', 64)
    learning_rate = data.get('learning_rate', 0.001)
    dataset_path = data.get('dataset_path', None)  # Custom dataset path
    
    def run_training():
        global training_state
        training_state['running'] = True
        training_state['total_epochs'] = epochs
        training_state['current_epoch'] = 0
        training_state['logs'] = []
        training_state['start_time'] = datetime.now().isoformat()
        
        try:
            cmd = [
                sys.executable,
                str(BASE_DIR / 'training' / 'train_processed.py'),
                '--models', *models,
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--lr', str(learning_rate)
            ]
            
            if dataset_path:
                cmd.extend(['--data-dir', dataset_path])
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(BASE_DIR / 'training')
            )
            
            training_state['pid'] = process.pid
            
            for line in process.stdout:
                training_state['logs'].append(line.strip())
                
                # Parse progress
                if 'Epoch' in line and '/' in line:
                    try:
                        parts = line.split('Epoch')[1].split('/')[0]
                        training_state['current_epoch'] = int(parts.strip())
                        training_state['progress'] = int(
                            (training_state['current_epoch'] / training_state['total_epochs']) * 100
                        )
                    except:
                        pass
                
                if 'Training' in line and 'Model' in line:
                    training_state['current_model'] = line.split(':')[-1].strip()
            
            process.wait()
            
            # Save to history
            history = load_training_history()
            history['trainings'].append({
                'id': len(history['trainings']) + 1,
                'timestamp': datetime.now().isoformat(),
                'models': models,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'dataset_path': dataset_path or 'processed_data',
                'status': 'completed' if process.returncode == 0 else 'failed',
                'duration_seconds': (datetime.now() - datetime.fromisoformat(training_state['start_time'])).total_seconds()
            })
            save_training_history(history)
            
        except Exception as e:
            training_state['logs'].append(f"Error: {str(e)}")
        finally:
            training_state['running'] = False
            training_state['progress'] = 100 if training_state['current_epoch'] == training_state['total_epochs'] else 0
            training_state['pid'] = None
    
    thread = threading.Thread(target=run_training)
    thread.start()
    
    return jsonify({
        'status': 'ok',
        'message': f'Training started for models: {models}'
    })


@api.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop current training"""
    global training_state
    
    if not training_state['running']:
        return jsonify({'status': 'error', 'message': 'No training in progress'}), 400
    
    if training_state['pid']:
        try:
            import signal
            os.kill(training_state['pid'], signal.SIGTERM)
        except:
            pass
    
    training_state['running'] = False
    training_state['pid'] = None
    
    return jsonify({'status': 'ok', 'message': 'Training stopped'})


@api.route('/api/training/status', methods=['GET'])
def training_status():
    """Get current training status"""
    return jsonify({
        'status': 'ok',
        'training': {
            'running': training_state['running'],
            'progress': training_state['progress'],
            'current_epoch': training_state['current_epoch'],
            'total_epochs': training_state['total_epochs'],
            'current_model': training_state['current_model'],
            'logs': training_state['logs'][-50:],  # Last 50 lines
            'start_time': training_state['start_time']
        }
    })


# =============================================================================
# HISTORY & REPORTS APIs
# =============================================================================

@api.route('/api/history', methods=['GET'])
def get_history():
    """Get training and evaluation history"""
    history = load_training_history()
    return jsonify({
        'status': 'ok',
        'history': history
    })


@api.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear history"""
    save_training_history({'trainings': [], 'evaluations': []})
    return jsonify({'status': 'ok', 'message': 'History cleared'})


@api.route('/api/reports/<report_type>', methods=['GET'])
def get_report(report_type):
    """Get specific report"""
    report_files = {
        'cnn': 'CNN_classification_report_processed.txt',
        'lstm': 'LSTM_classification_report_processed.txt',
        'hybrid': 'Hybrid_classification_report_processed.txt'
    }
    
    if report_type.lower() not in report_files:
        return jsonify({'status': 'error', 'message': 'Invalid report type'}), 400
    
    report_path = LOGS_DIR / report_files[report_type.lower()]
    
    if not report_path.exists():
        return jsonify({'status': 'error', 'message': 'Report not found'}), 404
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    return jsonify({
        'status': 'ok',
        'report': content,
        'filename': report_files[report_type.lower()],
        'modified': datetime.fromtimestamp(report_path.stat().st_mtime).isoformat()
    })


# =============================================================================
# DATASET MANAGEMENT APIs
# =============================================================================

@api.route('/api/dataset/info', methods=['GET'])
def get_dataset_info():
    """Get information about current dataset"""
    info = {
        'processed_data': None,
        'raw_data': None
    }
    
    # Check processed_data
    processed_dir = BASE_DIR / 'processed_data'
    if processed_dir.exists():
        try:
            X_train = np.load(processed_dir / 'X_train_seq.npy', mmap_mode='r')
            X_val = np.load(processed_dir / 'X_val_seq.npy', mmap_mode='r')
            X_test = np.load(processed_dir / 'X_test_seq.npy', mmap_mode='r')
            y_train = np.load(processed_dir / 'y_train_seq.npy')
            
            # Load config
            config = {}
            config_path = processed_dir / 'config.pkl'
            if config_path.exists():
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)
            
            # Class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            class_dist = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
            
            info['processed_data'] = {
                'path': str(processed_dir),
                'train_samples': int(X_train.shape[0]),
                'val_samples': int(X_val.shape[0]),
                'test_samples': int(X_test.shape[0]),
                'time_steps': int(X_train.shape[1]),
                'features': int(X_train.shape[2]),
                'class_distribution': class_dist,
                'feature_names': config.get('feature_names', []),
                'total_size_mb': round(
                    (X_train.nbytes + X_val.nbytes + X_test.nbytes) / (1024*1024), 2
                )
            }
        except Exception as e:
            info['processed_data'] = {'error': str(e)}
    
    # Check raw demo data
    demo_file = DATA_DIR / 'demo_test.csv'
    if demo_file.exists():
        import pandas as pd
        try:
            df = pd.read_csv(demo_file)
            info['raw_data'] = {
                'path': str(demo_file),
                'samples': len(df),
                'columns': list(df.columns),
                'size_mb': round(demo_file.stat().st_size / (1024*1024), 2)
            }
        except Exception as e:
            info['raw_data'] = {'error': str(e)}
    
    return jsonify({
        'status': 'ok',
        'dataset': info
    })


@api.route('/api/dataset/set-path', methods=['POST'])
def set_dataset_path():
    """Set custom dataset path for training - supports CSV files or folder with .npy files"""
    global custom_dataset_path
    data = request.get_json()
    path = data.get('path', '')
    
    if not path:
        return jsonify({'status': 'error', 'message': 'Path is required'}), 400
    
    # Normalize path
    path = path.strip().strip('"').strip("'")
    
    if not os.path.exists(path):
        return jsonify({'status': 'error', 'message': f'Path does not exist: {path}'}), 400
    
    dataset_type = None
    info = {}
    
    # Check if it's a CSV file
    if os.path.isfile(path) and path.lower().endswith('.csv'):
        dataset_type = 'csv'
        file_size = os.path.getsize(path)
        info = {
            'type': 'csv',
            'file': os.path.basename(path),
            'size_mb': round(file_size / (1024*1024), 2),
            'path': path
        }
        custom_dataset_path = path
        
    # Check if it's a folder with .npy files
    elif os.path.isdir(path):
        required_files = ['X_train_seq.npy', 'y_train_seq.npy', 'X_val_seq.npy', 'y_val_seq.npy', 
                          'X_test_seq.npy', 'y_test_seq.npy']
        
        found_npy = [f for f in required_files if os.path.exists(os.path.join(path, f))]
        missing_npy = [f for f in required_files if not os.path.exists(os.path.join(path, f))]
        
        # Check for CSV files in folder
        csv_files = [f for f in os.listdir(path) if f.lower().endswith('.csv')]
        
        if len(found_npy) == len(required_files):
            dataset_type = 'preprocessed'
            info = {
                'type': 'preprocessed',
                'files': found_npy,
                'path': path
            }
            custom_dataset_path = path
        elif csv_files:
            dataset_type = 'csv_folder'
            info = {
                'type': 'csv_folder',
                'csv_files': csv_files[:10],  # Show first 10
                'total_csv': len(csv_files),
                'path': path,
                'note': 'Folder contains CSV files. Select a specific CSV or preprocess first.'
            }
            custom_dataset_path = path
        else:
            return jsonify({
                'status': 'error',
                'message': f'Folder must contain preprocessed .npy files or CSV files. Missing: {missing_npy}'
            }), 400
    else:
        return jsonify({
            'status': 'error',
            'message': 'Path must be a CSV file or a folder containing preprocessed data'
        }), 400
    
    return jsonify({
        'status': 'ok',
        'message': f'Dataset path validated: {path}',
        'dataset_type': dataset_type,
        'info': info,
        'path': path
    })


# =============================================================================
# COMPARE RESULTS APIs
# =============================================================================

@api.route('/api/compare', methods=['GET'])
def compare_results():
    """Compare old vs new evaluation results"""
    results = {
        'old': None,
        'new': None
    }
    
    # Load old results (from training/outputs)
    old_file = LOGS_DIR / 'evaluation_results.json'
    if old_file.exists():
        with open(old_file, 'r') as f:
            results['old'] = json.load(f)
    
    # Load new results (from processed data)
    new_file = LOGS_DIR / 'evaluation_results_processed.json'
    if new_file.exists():
        with open(new_file, 'r') as f:
            results['new'] = json.load(f)
    
    # Calculate improvements
    improvements = {}
    if results['old'] and results['new']:
        for model in results['new'].keys():
            if model in results['old']:
                old_metrics = results['old'][model]
                new_metrics = results['new'][model]
                
                improvements[model] = {
                    'accuracy_change': round(
                        (new_metrics.get('accuracy', 0) - old_metrics.get('accuracy', 0)) * 100, 2
                    ),
                    'fpr_change': round(
                        (new_metrics.get('fpr', 0) - old_metrics.get('fpr', 0)) * 100, 2
                    ),
                    'fnr_change': round(
                        (new_metrics.get('fnr', 0) - old_metrics.get('fnr', 0)) * 100, 2
                    ),
                    'roc_auc_change': round(
                        new_metrics.get('roc_auc', 0) - old_metrics.get('roc_auc', 0), 4
                    )
                }
    
    return jsonify({
        'status': 'ok',
        'comparison': {
            'old': results['old'],
            'new': results['new'],
            'improvements': improvements
        }
    })


# =============================================================================
# SYSTEM INFO APIs
# =============================================================================

@api.route('/api/system/info', methods=['GET'])
def system_info():
    """Get system information"""
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            'name': torch.cuda.get_device_name(0),
            'memory_total': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            'memory_allocated': round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        }
    
    return jsonify({
        'status': 'ok',
        'system': {
            'device': str(DEVICE),
            'gpu_available': torch.cuda.is_available(),
            'gpu_info': gpu_info,
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'base_dir': str(BASE_DIR)
        }
    })
