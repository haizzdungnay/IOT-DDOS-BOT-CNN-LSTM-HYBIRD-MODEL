# 🛡️ Bot-IoT Multi-Model DDoS Detection Demo

Real-time Traffic Replay Dashboard for comparing 3 Deep Learning Models:
- **CNN 1D** - Spatial feature extraction
- **LSTM** - Temporal sequence modeling  
- **Hybrid CNN-LSTM** - Combined approach (IEEE standard)

## 🎯 Features

✅ **Real-time Replay**: Simulates traffic from Bot-IoT test dataset  
✅ **Multi-Model Comparison**: Side-by-side evaluation of 3 models  
✅ **Live Dashboard**: Beautiful web interface with charts  
✅ **WebSocket Updates**: Instant model predictions  
✅ **Consensus Detection**: 2-out-of-3 voting system  
✅ **Ground Truth Validation**: Compare predictions with actual labels

## 📁 Project Structure

```
DemoWeb_3Models/
├── app.py                          # Flask backend with SocketIO
├── backend/
│   ├── replay_detector.py          # Multi-model detector logic
│   └── models/                     # PyTorch model weights
│       ├── CNN_best.pt
│       ├── LSTM_best.pt
│       ├── Hybrid_CNN_LSTM_best.pt
│       └── scaler_standard.pkl
├── data/
│   └── demo_test.csv              # Test data for replay
├── public/
│   └── index.html                 # Frontend dashboard
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Models

Copy trained model weights to `backend/models/`:
- `CNN_best.pt`
- `LSTM_best.pt`  
- `Hybrid_CNN_LSTM_best.pt`
- `scaler_standard.pkl`

### 3. Prepare Test Data

Create `data/demo_test.csv` with Bot-IoT test data. Must include:
- 15 features: `pkts, bytes, dur, mean, stddev, sum, min, max, spkts, dpkts, sbytes, dbytes, rate, srate, drate`
- `attack` column (0=Normal, 1=Attack)
- Optional: `stime, srcip, dstip` for metadata

**Example CSV structure:**
```csv
pkts,bytes,dur,mean,stddev,sum,min,max,spkts,dpkts,sbytes,dbytes,rate,srate,drate,attack,stime,srcip,dstip
10,500,1.2,50,10,500,40,60,5,5,250,250,8.33,4.17,4.17,0,1234567890,192.168.1.1,192.168.1.2
...
```

### 4. Run Server

```bash
python app.py
```

Server starts at: **http://localhost:5000**

### 5. Open Dashboard

Navigate to `http://localhost:5000` in your browser.

## 🎮 Usage

### Control Panel:
- **Start Replay**: Begin traffic simulation
- **Stop Replay**: Pause simulation
- **Speed Control**: Adjust replay speed (0.01s - 0.5s per packet)

### Dashboard Elements:

#### Model Cards (3 cards):
Each shows:
- Real-time prediction (Normal/Attack)
- Confidence percentage
- Attack count
- Progress bar (attack probability)

#### Live Chart:
- Line chart comparing attack probabilities across all 3 models
- Updates in real-time (last 50 packets)

#### Statistics:
- **Ground Truth**: Total packets, true normal/attack counts
- **Consensus**: When 2+ models agree on attack

#### Traffic Log:
- Recent packet predictions
- Color-coded: 🟢 Normal | 🔴 Attack
- Shows: Packet ID, True label, All 3 predictions

## 📊 Evaluation Metrics

The system tracks:
- **Per-model detection counts**: How many attacks each model detected
- **Consensus detection**: When majority (2/3) agree
- **Ground truth comparison**: Accuracy against actual labels

## 🎓 For Academic Demo

### Talking Points:

> *"This system demonstrates real-world applicability of our trained models. By replaying actual Bot-IoT traffic, we can observe:*
> 
> *1. **Real-time Performance**: How fast each model responds*
> *2. **Consistency**: Which model maintains stable predictions*
> *3. **Consensus Validation**: When models agree, confidence increases*
> *4. **False Positive Analysis**: LSTM shows lowest FPR (0.7%) vs Hybrid (12.8%)"*

### Demo Scenarios:

#### Scenario 1: Normal Traffic
- All 3 models should predict 🟢 Normal
- Low attack probabilities (<0.5)

#### Scenario 2: DDoS Attack
- Models should detect 🔴 Attack
- High attack probabilities (>0.5)
- Observe which model responds first

#### Scenario 3: Mixed Traffic
- Observe how models handle transitions
- Check for false positives/negatives

## 🔧 Customization

### Change Replay Speed:
Edit in Control Panel dropdown (0.01s - 0.5s)

### Use Different Test Data:
Replace `data/demo_test.csv` with your own test set

### Adjust Consensus Threshold:
In `replay_detector.py`, line ~280:
```python
attack_votes = sum([p['pred'] for p in predictions.values()])
if attack_votes >= 2:  # Change to 3 for unanimous
    self.stats['consensus_attacks'] += 1
```

## 📝 Notes

- **No Real Network Traffic**: System uses CSV replay (safe for demo)
- **Feature Alignment**: CSV must match Bot-IoT preprocessing
- **Model Compatibility**: Requires PyTorch 2.0+
- **Browser Support**: Chrome/Firefox recommended for Chart.js

## 🐛 Troubleshooting

### Models not loading:
```bash
# Check model files exist
ls backend/models/

# Verify PyTorch version
python -c "import torch; print(torch.__version__)"
```

### CSV format error:
Ensure CSV has exactly 15 feature columns + `attack` column

### WebSocket connection failed:
- Check firewall allows port 5000
- Try `http://127.0.0.1:5000` instead of `localhost`

## 📚 References

- Bot-IoT Dataset: UNSW Canberra Cyber (2018)
- IEEE Papers: CNN→LSTM for DDoS Detection
- PyTorch: Deep Learning Framework

## 👥 Authors

IoT Security Research Team - 2026

## 📄 License

For academic use only.
