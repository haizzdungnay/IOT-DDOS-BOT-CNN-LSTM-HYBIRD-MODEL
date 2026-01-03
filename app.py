"""
Flask App - Bot-IoT Multi-Model DDoS Detection System
Features:
- Dashboard với metrics và ranking
- Training management (train mới, train lại)
- Evaluation management (đánh giá, so sánh)
- History/Reports (lịch sử, báo cáo)
- Real-time monitoring
- Dataset management
Flask App for Bot-IoT Multi-Model Dashboard
Advanced Dashboard with:
- Model Evaluation
- Training Management
- Real-time Monitoring
- History & Reports
- Dataset Management
- Compare Results

Author: IoT Security Research Team
Date: 2026-01-03
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging
import sys
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path

# Add directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training'))

from backend.replay_detector import ReplayDetector
from backend.api_routes import api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DemoApp')

# Initialize Flask app
app = Flask(__name__, static_folder='public')
CORS(app, resources={r"/*": {"origins": "*"}})

# Register API Blueprint
app.register_blueprint(api)

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize components
detector = None
history_manager = get_history_manager("history")

# Training state
training_state = {
    "is_training": False,
    "current_model": None,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "logs": []
}

try:
    detector = ReplayDetector(models_dir='backend/models', data_dir='data')
    logger.info("✅ ReplayDetector initialized")
except Exception as e:
    logger.error(f"❌ ReplayDetector failed: {e}")
    detector = None


# =============================================================================
# WEBSOCKET CALLBACKS
# =============================================================================

def broadcast_traffic_update(data):
    """Broadcast traffic updates via SocketIO"""
    try:
        socketio.emit('traffic_update', data, namespace='/')
    except Exception as e:
        logger.error(f"Broadcast error: {e}")


def broadcast_training_update(data):
    """Broadcast training progress via SocketIO"""
    try:
        socketio.emit('training_update', data, namespace='/')
    except Exception as e:
        logger.error(f"Training broadcast error: {e}")


# =============================================================================
# STATIC FILES
# =============================================================================

@app.route('/')
def index():
    """Serve main dashboard page"""
    return send_from_directory('public', 'dashboard.html')


@app.route('/old')
def old_index():
    """Serve old demo page"""
    return send_from_directory('public', 'index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('public/static', path)


@app.route('/api/status')
def get_status():
    """Lấy trạng thái hệ thống"""
    return jsonify({
        'status': 'ok',
        'detector_ready': detector is not None,
        'running': detector.running if detector else False,
        'device': str(detector.device) if detector else 'N/A',
        'models_loaded': list(detector.models.keys()) if detector else [],
        'training_state': training_state,
        'timestamp': datetime.now().isoformat()
    })


# =============================================================================
# API: MODELS INFORMATION
# =============================================================================

@app.route('/api/models')
def get_models():
    """Lấy thông tin các models"""
    if detector is None:
        return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500

    models_info = {}
    for name in detector.models.keys():
        model = detector.models[name]
        params = sum(p.numel() for p in model.parameters())
        models_info[name] = {
            'name': name,
            'parameters': params,
            'loaded': True
        }

    return jsonify({
        'status': 'ok',
        'models': models_info,
        'device': str(detector.device),
        'features': detector.feature_names,
        'time_steps': detector.TIME_STEPS
    })


@app.route('/api/models/metrics')
def get_models_metrics():
    """Lấy metrics đánh giá của các models (từ file logs)"""
    metrics = {}

    # Đọc từ evaluation_results.json
    eval_files = [
        'training/logs/evaluation_results.json',
        'training/logs/evaluation_results_processed.json'
    ]

    for eval_file in eval_files:
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                data = json.load(f)
                for model_name, model_metrics in data.items():
                    if model_name not in metrics:
                        metrics[model_name] = model_metrics

    # Tính ranking
    if metrics:
        # Ranking theo F1-Score
        sorted_models = sorted(
            metrics.items(),
            key=lambda x: x[1].get('f1_score', 0),
            reverse=True
        )
        for rank, (name, m) in enumerate(sorted_models, 1):
            metrics[name]['ranking'] = rank

    return jsonify({
        'status': 'ok',
        'metrics': metrics
    })


# =============================================================================
# API: REPLAY (Real-time Demo)
# =============================================================================

@app.route('/api/replay/start', methods=['POST'])
def start_replay():
    """Bắt đầu replay traffic"""
    if detector is None:
        return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500

    if detector.running:
        return jsonify({'status': 'error', 'message': 'Replay already running'}), 400

    try:
        data = request.get_json() or {}
        csv_file = data.get('csv_file', 'demo_test.csv')
        speed = data.get('speed', 0.1)

        detector.start_replay(
            csv_filename=csv_file,
            speed=speed,
            callback=broadcast_traffic_update
        )

        return jsonify({
            'status': 'success',
            'message': f'Replay started: {csv_file}'
        })

    except Exception as e:
        logger.error(f"Start replay error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/replay/stop', methods=['POST'])
def stop_replay():
    """Dừng replay"""
    if detector is None:
        return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500

    try:
        detector.stop_replay()
        return jsonify({'status': 'success', 'message': 'Replay stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/replay/stats')
def get_replay_stats():
    """Lấy thống kê replay hiện tại"""
    if detector is None:
        return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500

    return jsonify({
        'status': 'ok',
        'running': detector.running,
        'stats': detector.get_stats()
    })


# =============================================================================
# API: TRAINING
# =============================================================================

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Bắt đầu training model"""
    global training_state

    if training_state["is_training"]:
        return jsonify({'status': 'error', 'message': 'Training already in progress'}), 400

    try:
        data = request.get_json() or {}
        models = data.get('models', ['CNN', 'LSTM', 'Hybrid'])
        epochs = data.get('epochs', 30)
        use_class_weights = data.get('use_class_weights', True)
        use_focal_loss = data.get('use_focal_loss', False)  # New option
        focal_alpha = data.get('focal_alpha', 0.25)
        focal_gamma = data.get('focal_gamma', 2.0)
        data_dir = data.get('data_dir', 'processed_data')

        # Start training in background thread
        thread = threading.Thread(
            target=run_training,
            args=(models, epochs, use_class_weights, data_dir, use_focal_loss, focal_alpha, focal_gamma)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'status': 'success',
            'message': f'Training started for models: {models}'
        })

    except Exception as e:
        logger.error(f"Start training error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def run_training(models, epochs, use_class_weights, data_dir,
                 use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0):
    """Background training function"""
    global training_state

    training_state["is_training"] = True
    training_state["total_epochs"] = epochs
    training_state["logs"] = []

    session_id = history_manager.generate_session_id()

    try:
        # Import training modules
        sys.path.insert(0, 'training')
        from training.config import DEVICE
        from training.data_loader import load_from_processed_data
        from training.models import get_model, count_parameters
        from training.trainer import Trainer

        # Load data
        add_training_log(f"Loading data from {data_dir}...")
        data = load_from_processed_data(data_dir)
        train_loader = data['train_loader']
        val_loader = data['val_loader']
        class_weights = data.get('class_weights') if use_class_weights else None

        loss_type = "Focal Loss" if use_focal_loss else "BCEWithLogitsLoss"
        add_training_log(f"Data loaded. Training on {DEVICE} with {loss_type}")

        for model_name in models:
            training_state["current_model"] = model_name
            training_state["current_epoch"] = 0
            add_training_log(f"\n{'='*50}")
            add_training_log(f"Training {model_name}...")

            # Create model
            model = get_model(model_name)
            n_params = count_parameters(model)
            add_training_log(f"Parameters: {n_params:,}")

            # Create trainer with callback
            trainer = Trainer(
                model, model_name,
                class_weights=class_weights,
                use_focal_loss=use_focal_loss,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma
            )

            # Custom training loop with progress updates
            for epoch in range(1, epochs + 1):
                training_state["current_epoch"] = epoch
                training_state["progress"] = int((epoch / epochs) * 100)

                # Train one epoch
                train_loss, train_acc = trainer.train_epoch(train_loader)
                val_loss, val_acc = trainer.validate(val_loader)

                # Update scheduler
                trainer.scheduler.step(val_loss)

                # Log
                log_msg = f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val: {val_loss:.4f}"
                add_training_log(log_msg)

                # Broadcast progress
                broadcast_training_update({
                    'model': model_name,
                    'epoch': epoch,
                    'total_epochs': epochs,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'progress': training_state["progress"]
                })

                # Check early stopping
                if trainer.early_stopping(val_loss):
                    add_training_log(f"Early stopping at epoch {epoch}")
                    break

                # Save best model
                if val_loss < trainer.best_val_loss:
                    trainer.best_val_loss = val_loss
                    trainer.best_val_acc = val_acc
                    trainer.best_epoch = epoch
                    trainer.save_model(f"{model_name}_best.pt")

            # Save to history
            history_manager.save_training_session(
                session_id=session_id,
                model_name=model_name,
                config={
                    'epochs': epochs,
                    'use_class_weights': use_class_weights,
                    'use_focal_loss': use_focal_loss,
                    'focal_alpha': focal_alpha if use_focal_loss else None,
                    'focal_gamma': focal_gamma if use_focal_loss else None,
                    'data_dir': data_dir
                },
                history=trainer.history,
                metrics={
                    'best_val_loss': trainer.best_val_loss,
                    'best_val_acc': trainer.best_val_acc,
                    'best_epoch': trainer.best_epoch
                },
                model_path=f"training/outputs/{model_name}_best.pt"
            )

            add_training_log(f"{model_name} completed. Best acc: {trainer.best_val_acc:.4f}")

        add_training_log("\n✅ Training completed!")
        broadcast_training_update({'status': 'completed', 'session_id': session_id})

    except Exception as e:
        add_training_log(f"❌ Error: {str(e)}")
        broadcast_training_update({'status': 'error', 'message': str(e)})

    finally:
        training_state["is_training"] = False
        training_state["current_model"] = None
        training_state["progress"] = 0


def add_training_log(message):
    """Thêm log và broadcast"""
    training_state["logs"].append({
        'timestamp': datetime.now().isoformat(),
        'message': message
    })
    broadcast_training_update({
        'type': 'log',
        'message': message
    })
    logger.info(f"[Training] {message}")


@app.route('/api/training/status')
def get_training_status():
    """Lấy trạng thái training hiện tại"""
    return jsonify({
        'status': 'ok',
        'training_state': training_state
    })


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Dừng training (graceful)"""
    global training_state
    training_state["is_training"] = False
    return jsonify({'status': 'success', 'message': 'Training stop requested'})


# =============================================================================
# API: EVALUATION
# =============================================================================

@app.route('/api/evaluation/run', methods=['POST'])
def run_evaluation():
    """Chạy đánh giá models"""
    try:
        data = request.get_json() or {}
        models = data.get('models', ['CNN', 'LSTM', 'Hybrid'])
        data_dir = data.get('data_dir', 'processed_data')
        model_dir = data.get('model_dir', 'training/outputs')

        # Import modules
        sys.path.insert(0, 'training')
        from training.config import DEVICE
        from training.data_loader import load_from_processed_data
        from training.models import get_model

        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

        # Load data
        test_data = load_from_processed_data(data_dir)
        test_loader = test_data['test_loader']

        results = {}

        for model_name in models:
            # Find model file
            model_path = None
            for pattern in [f"{model_name}_best.pt", f"{model_name}_CNN_LSTM_best.pt"]:
                path = Path(model_dir) / pattern
                if path.exists():
                    model_path = path
                    break

            if model_path is None:
                continue

            # Load model
            model = get_model(model_name)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.to(DEVICE)
            model.eval()

            # Predict
            all_probs, all_labels = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(DEVICE)
                    outputs = model(X_batch)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(y_batch.numpy())

            y_true = np.array(all_labels)
            y_prob = np.array(all_probs)
            y_pred = (y_prob > 0.5).astype(int)

            # Metrics
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, len(y_true)

            try:
                roc_auc = roc_auc_score(y_true, y_prob)
            except:
                roc_auc = 0.5

            results[model_name] = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc),
                'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
                'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
                'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
                'total_samples': len(y_true)
            }

        # Save to history
        session_id = history_manager.generate_session_id()
        history_manager.save_evaluation_session(
            session_id=session_id,
            models_results=results,
            dataset_info={'data_dir': data_dir, 'model_dir': model_dir}
        )

        # Calculate rankings
        sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        for rank, (name, _) in enumerate(sorted_models, 1):
            results[name]['ranking'] = rank

        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'results': results
        })

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/evaluation/latest')
def get_latest_evaluation():
    """Lấy kết quả đánh giá mới nhất"""
    result = history_manager.get_latest_evaluation()
    if result:
        return jsonify({'status': 'ok', 'data': result})
    return jsonify({'status': 'error', 'message': 'No evaluation found'}), 404


# =============================================================================
# API: HISTORY
# =============================================================================

@app.route('/api/history/training')
def get_training_history():
    """Lấy lịch sử training"""
    limit = request.args.get('limit', 20, type=int)
    model_name = request.args.get('model', None)

    sessions = history_manager.get_training_sessions(model_name=model_name, limit=limit)
    return jsonify({
        'status': 'ok',
        'sessions': sessions
    })


@app.route('/api/history/evaluation')
def get_evaluation_history():
    """Lấy lịch sử đánh giá"""
    limit = request.args.get('limit', 20, type=int)
    sessions = history_manager.get_evaluation_sessions(limit=limit)
    return jsonify({
        'status': 'ok',
        'sessions': sessions
    })


@app.route('/api/history/training/<session_id>/<model_name>')
def get_training_detail(session_id, model_name):
    """Lấy chi tiết một phiên training"""
    data = history_manager.get_training_history(session_id, model_name)
    if data:
        return jsonify({'status': 'ok', 'data': data})
    return jsonify({'status': 'error', 'message': 'Not found'}), 404


@app.route('/api/history/evaluation/<session_id>')
def get_evaluation_detail(session_id):
    """Lấy chi tiết một phiên đánh giá"""
    data = history_manager.get_evaluation_results(session_id)
    if data:
        return jsonify({'status': 'ok', 'data': data})
    return jsonify({'status': 'error', 'message': 'Not found'}), 404


@app.route('/api/history/compare', methods=['POST'])
def compare_sessions():
    """So sánh nhiều phiên đánh giá"""
    data = request.get_json() or {}
    session_ids = data.get('session_ids', [])

    if len(session_ids) < 2:
        return jsonify({'status': 'error', 'message': 'Need at least 2 sessions'}), 400

    comparison = history_manager.compare_sessions(session_ids)
    return jsonify({
        'status': 'ok',
        'comparison': comparison
    })


@app.route('/api/history/statistics')
def get_history_statistics():
    """Lấy thống kê tổng quan"""
    stats = history_manager.get_statistics()
    return jsonify({
        'status': 'ok',
        'statistics': stats
    })


# =============================================================================
# API: DATASETS
# =============================================================================

@app.route('/api/datasets')
def get_datasets():
    """Liệt kê các datasets có sẵn"""
    datasets = []

    # Check processed_data
    processed_dir = Path('processed_data')
    if processed_dir.exists():
        datasets.append({
            'name': 'processed_data',
            'type': 'preprocessed',
            'path': str(processed_dir),
            'files': [f.name for f in processed_dir.glob('*.npy')]
        })

    # Check data folder
    data_dir = Path('data')
    if data_dir.exists():
        for csv_file in data_dir.glob('*.csv'):
            datasets.append({
                'name': csv_file.stem,
                'type': 'csv',
                'path': str(csv_file),
                'size': csv_file.stat().st_size
            })

    return jsonify({
        'status': 'ok',
        'datasets': datasets
    })


@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    """Upload dataset mới (chỉ nhận đường dẫn)"""
    data = request.get_json() or {}
    path = data.get('path')

    if not path:
        return jsonify({'status': 'error', 'message': 'Path required'}), 400

    if not os.path.exists(path):
        return jsonify({'status': 'error', 'message': f'Path not found: {path}'}), 404

    return jsonify({
        'status': 'success',
        'message': f'Dataset registered: {path}',
        'path': path
    })


# =============================================================================
# WEBSOCKET EVENTS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    logger.info(f"🔌 Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"🔌 Client disconnected: {request.sid}")


@socketio.on('request_status')
def handle_status_request():
    if detector:
        emit('status_update', {
            'running': detector.running,
            'stats': detector.get_stats(),
            'training_state': training_state
        })


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    logger.info("="*70)
    logger.info("🚀 Bot-IoT Advanced Dashboard Server")
    logger.info("="*70)
    logger.info(f"Models: {list(detector.models.keys()) if detector else 'Not loaded'}")
    logger.info(f"Device: {detector.device if detector else 'N/A'}")
    logger.info("="*70)
    logger.info("📊 Dashboard: http://localhost:5000")
    logger.info("📊 Old Demo:  http://localhost:5000/old")
    logger.info("="*70)
    
    # Run with SocketIO
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )
