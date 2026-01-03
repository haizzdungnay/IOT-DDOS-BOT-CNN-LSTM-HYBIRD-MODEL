"""
Flask App for Bot-IoT Multi-Model Demo
=======================================
Real-time Traffic Replay Dashboard for 3 Deep Learning Models

Author: IoT Security Research Team
Date: 2026-01-02
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.replay_detector import ReplayDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DemoApp')

# Initialize Flask app
app = Flask(__name__, static_folder='public')
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Replay Detector
detector = None

try:
    detector = ReplayDetector(
        models_dir='backend/models',
        data_dir='data'
    )
    logger.info("✅ ReplayDetector initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize ReplayDetector: {e}")
    detector = None


# =============================================================================
# WEBSOCKET CALLBACKS
# =============================================================================

def broadcast_traffic_update(data):
    """
    Callback function to broadcast traffic updates via SocketIO
    
    Args:
        data: Dictionary containing:
            - packet_id: int
            - timestamp: float
            - true_label: 0/1
            - predictions: {CNN: {...}, LSTM: {...}, Hybrid: {...}}
            - stats: {...}
            - metadata: {...}
    """
    try:
        socketio.emit('traffic_update', data, namespace='/')
    except Exception as e:
        logger.error(f"❌ Error broadcasting update: {e}")


# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.route('/')
def index():
    """Serve main HTML page"""
    return send_from_directory('public', 'index.html')


@app.route('/api/status')
def get_status():
    """Get system status"""
    if detector is None:
        return jsonify({
            'status': 'error',
            'message': 'Detector not initialized'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'running': detector.running,
        'device': str(detector.device),
        'models_loaded': list(detector.models.keys()),
        'buffer_size': len(detector.buffer),
        'stats': detector.get_stats()
    })


@app.route('/api/start_replay', methods=['POST'])
def start_replay():
    """
    Start traffic replay
    
    Request JSON:
    {
        "csv_file": "demo_test.csv",
        "speed": 0.1  // seconds per packet
    }
    """
    if detector is None:
        return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500
    
    if detector.running:
        return jsonify({'status': 'error', 'message': 'Replay already running'}), 400
    
    try:
        data = request.get_json()
        csv_file = data.get('csv_file', 'demo_test.csv')
        speed = data.get('speed', 0.1)
        
        # Start replay with callback
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
        logger.error(f"❌ Error starting replay: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/stop_replay', methods=['POST'])
def stop_replay():
    """Stop traffic replay"""
    if detector is None:
        return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500
    
    try:
        detector.stop_replay()
        return jsonify({'status': 'success', 'message': 'Replay stopped'})
    except Exception as e:
        logger.error(f"❌ Error stopping replay: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get current statistics"""
    if detector is None:
        return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500
    
    return jsonify({
        'status': 'ok',
        'stats': detector.get_stats()
    })


@app.route('/api/models')
def get_models():
    """Get information about loaded models"""
    if detector is None:
        return jsonify({'status': 'error', 'message': 'Detector not initialized'}), 500
    
    return jsonify({
        'status': 'ok',
        'models': list(detector.models.keys()),
        'device': str(detector.device),
        'features': detector.feature_names,
        'time_steps': detector.TIME_STEPS
    })


# =============================================================================
# SOCKETIO EVENTS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    logger.info(f"🔌 Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    logger.info(f"🔌 Client disconnected: {request.sid}")


@socketio.on('request_status')
def handle_status_request():
    """Client requested status update"""
    if detector:
        emit('status_update', {
            'running': detector.running,
            'stats': detector.get_stats()
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
    logger.info("🚀 Bot-IoT Multi-Model Demo Server")
    logger.info("="*70)
    logger.info(f"Models: {list(detector.models.keys()) if detector else 'Not loaded'}")
    logger.info(f"Device: {detector.device if detector else 'N/A'}")
    logger.info("="*70)
    
    # Run with SocketIO
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to avoid loading models twice
    )
