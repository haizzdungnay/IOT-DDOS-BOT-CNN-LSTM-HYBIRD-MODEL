# Hệ Thống Phát Hiện Tấn Công DDoS IoT với Deep Learning

## 🎯 Giới Thiệu Dự Án

Hệ thống phát hiện tấn công DDoS trong mạng IoT sử dụng 3 mô hình Deep Learning với khả năng tự động load models:

- **CNN 1D** (Convolutional Neural Network): Trích xuất đặc trưng không gian từ traffic patterns
- **LSTM** (Long Short-Term Memory): Mô hình hóa chuỗi thời gian và dependencies
- **Hybrid** (Parallel CNN-LSTM): CNN và LSTM song song, concatenate features

### ✨ Tính Năng Nổi Bật

- **Dynamic Model Loading**: Tự động phát hiện và load models mới từ folder
- **Advanced Dashboard**: Giao diện web hiện đại với 6 tabs chức năng
- **GPU Acceleration**: Tối ưu cho CUDA với RTX/GTX series
- **Real-time Monitoring**: Theo dõi predictions của 3 models đồng thời
- **Comprehensive Metrics**: Accuracy, FPR, FNR, ROC-AUC, Confusion Matrix
- **Training Management**: Train và đánh giá models qua web interface
- **Dataset Manager**: Upload và quản lý datasets qua UI
- **History & Reports**: Lưu trữ và so sánh kết quả training

---

## 📁 Cấu Trúc Dự Án

```
IOT-DDOS-BOT-CNN-LSTM-HYBIRD-MODEL/
│
├── backend/                        # Backend server
│   ├── replay_detector.py          # Multi-model inference engine
│   ├── api_routes.py               # REST APIs cho dashboard
│   └── models/                     # Model weights (auto-loaded)
│       ├── CNN_best.pt             # CNN model
│       ├── LSTM_best.pt            # LSTM model
│       ├── Hybrid_best.pt          # Parallel Hybrid model
│       └── scaler_standard.pkl     # Data scaler
│
├── training/                       # Training & Evaluation
│   ├── config.py                   # Global config (GPU, hyperparameters)
│   ├── models.py                   # Model architectures
│   ├── data_loader.py              # Data loading utilities
│   ├── trainer.py                  # Training class with early stopping
│   ├── train_processed.py          # Train với processed data
│   ├── evaluate_processed.py       # Evaluate models
│   ├── outputs/                    # Saved model weights
│   └── logs/                       # Training history & metrics
│
├── processed_data/                 # Pre-processed sequences
│   ├── X_train_seq.npy             # Training sequences (2.1M samples)
│   ├── X_test_seq.npy              # Test sequences (450K samples)
│   ├── config.pkl                  # Dataset config
│   └── class_weights.pkl           # Class balancing weights
│
├── public/                         # Frontend Dashboard
│   ├── dashboard.html              # Main UI (6 tabs)
│   ├── index.html                  # Legacy demo
│   └── static/js/
│       └── dashboard.js            # Dashboard logic
│
├── data/                           # Demo & training data
│   └── demo_test.csv               # Sample data for demo
│
├── app.py                          # Flask server entrypoint
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Hướng Dẫn Khởi Chạy Nhanh

### Bước 1: Clone Repository

```bash
git clone https://github.com/haizzdungnay/IOT-DDOS-BOT-CNN-LSTM-HYBIRD-MODEL.git
cd IOT-DDOS-BOT-CNN-LSTM-HYBIRD-MODEL
```

### Bước 2: Cài Đặt Dependencies

**Với GPU (khuyến nghị):**
```bash
# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Cài PyTorch với CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Cài các dependencies khác
pip install -r requirements.txt
```

**Với CPU only:**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt

```

### Bước 3: Khởi Động Dashboard Server

```bash
python app.py
```

Server sẽ chạy tại: **http://localhost:5000**

### Bước 4: Truy Cập Dashboard

Mở trình duyệt và truy cập:
- **Dashboard chính**: http://localhost:5000
- **Demo cũ**: http://localhost:5000/old

---

## 📊 Sử Dụng Dashboard

Dashboard có **6 tabs** chính:

### 1️⃣ Dashboard (Trang chủ)
- Tổng quan hệ thống: Số models, accuracy, FPR
- Model ranking table (xếp hạng theo FPR)
- Biểu đồ so sánh accuracy và error rates
- Thống kê GPU/CPU

### 2️⃣ Model Evaluation
- Hiển thị metrics chi tiết của từng model
- Confusion Matrix và Classification Report
- Refresh để cập nhật kết quả mới sau evaluation

### 3️⃣ Real-time Monitor
- Demo phát hiện real-time với 3 models
- Live chart hiển thị confidence scores
- Traffic log với predictions
- Statistics: Correct/Wrong predictions

**Cách sử dụng:**
1. Chọn tốc độ replay (Fast/Medium/Slow)
2. Click **Start Replay**
3. Quan sát predictions của 3 models đồng thời

### 4️⃣ Training
- Train models trực tiếp qua web interface
- Chọn models muốn train (CNN, LSTM, Hybrid)
- Cấu hình hyperparameters (epochs, batch size, learning rate)
- Xem progress bar và logs real-time

### 5️⃣ Dataset Manager
- Upload dataset (.csv format)
- Xem thông tin dataset hiện tại
- Quản lý data path

### 6️⃣ History & Reports
- Xem lịch sử training và evaluation
- So sánh kết quả giữa các lần chạy
- Export reports

---

## 🎓 Training & Evaluation

### Training với Processed Data

```bash
cd training

# Train tất cả 3 models
python train_processed.py

# Train model cụ thể
python train_processed.py --models CNN
python train_processed.py --models LSTM  
python train_processed.py --models Hybrid

# Custom hyperparameters
python train_processed.py --models LSTM --epochs 30 --batch-size 128 --lr 0.0001
```

### Evaluation Models

```bash
cd training

# Evaluate all models from backend/models
python evaluate_processed.py --model-dir ../backend/models

# Evaluate specific models
python evaluate_processed.py --models CNN LSTM --model-dir ../backend/models
```

### Output Files

**Sau training:**
```
training/outputs/
├── CNN_best.pt          # Model weights
├── LSTM_best.pt
├── Hybrid_best.pt
└── scaler_standard.pkl  # Data scaler
```

**Sau evaluation:**
```
training/logs/
├── evaluation_results_processed.json  # Metrics của all models
├── CNN_classification_report_processed.txt
├── LSTM_classification_report_processed.txt
└── Hybrid_classification_report_processed.txt
```

---

## 🔧 Dynamic Model Loading

Hệ thống tự động phát hiện và load models từ `backend/models/`:

**Supported filename patterns:**
- `*CNN*.pt` → CNN1D architecture
- `*LSTM*.pt` → LSTM architecture  
- `*Hybrid*.pt` → ParallelHybridCNNLSTM architecture
- `*Parallel*.pt` → ParallelHybridCNNLSTM architecture

**Để thêm model mới:**
1. Đặt file `.pt` vào `backend/models/`
2. Đảm bảo tên file chứa keyword: CNN, LSTM, Hybrid, hoặc Parallel
3. Restart server
4. Model sẽ tự động xuất hiện trong dashboard

---

## 📈 Model Performance

### Current Results (Bot-IoT Dataset)

| Model | Accuracy | Precision | Recall | F1-Score | FPR | FNR | ROC-AUC |
|-------|----------|-----------|--------|----------|-----|-----|---------|
| **LSTM** | 99.99% | 100.00% | 99.99% | 100.00% | 26.32% | 0.01% | 0.9423 |
| **Hybrid** | 99.99% | 99.99% | 99.99% | 99.97% | 36.84% | 0.004% | 0.9327 |
| **CNN** | 99.98% | 100.00% | 99.98% | 99.99% | 42.11% | 0.02% | 0.9103 |

**Test set:** 449,998 samples (99.99% attack traffic)

---

## 🖥️ GPU Support

Hệ thống tự động phát hiện và sử dụng GPU nếu có:

```bash
# Kiểm tra GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Tested GPUs:**
- NVIDIA GeForce RTX 3050 Laptop (4GB VRAM) ✅
- NVIDIA GeForce GTX 1060/1070/1080 ✅  
- NVIDIA RTX 2060/2070/2080 ✅
- NVIDIA RTX 3060/3070/3080/3090 ✅

---

## 🐛 Troubleshooting

### 1. Server không khởi động được

```bash
# Kiểm tra port 5000 có bị chiếm không
netstat -ano | findstr :5000  # Windows
lsof -i :5000  # Linux/Mac

# Hoặc đổi port trong app.py
socketio.run(app, host='0.0.0.0', port=5001, debug=True)
```

### 2. GPU không được sử dụng

```bash
# Cài đặt lại PyTorch với CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Out of Memory (OOM)

Giảm batch size trong `training/config.py`:
```python
BATCH_SIZE = 32  # Thay vì 64
```

### 4. Models không load được

Kiểm tra:
- File `.pt` có trong `backend/models/`?
- Tên file có chứa keyword: CNN, LSTM, Hybrid, Parallel?
- Check server logs để xem lỗi cụ thể

---

## 📝 Citation

Nếu sử dụng code này trong nghiên cứu, vui lòng cite:

```bibtex
@software{iot_ddos_detection_2026,
  title={IoT DDoS Detection System using Deep Learning},
  author={Your Name},
  year={2026},
  url={https://github.com/haizzdungnay/IOT-DDOS-BOT-CNN-LSTM-HYBIRD-MODEL}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 👥 Contributors

- **Author**: IoT Security Research Team
- **Contact**: [GitHub](https://github.com/haizzdungnay)

---

## 🙏 Acknowledgments

- Bot-IoT Dataset: [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- PyTorch Framework
- Flask & Socket.IO for real-time communication

---

## Phân Công Công Việc Nhóm

### Đồng Bộ Test Set (QUAN TRỌNG!)

**CẢNH BÁO**: Tất cả thành viên PHẢI dùng chung test set để đánh giá khách quan!

```bash
# Thành viên 1: Train và tạo test set
python train_all.py --data botiot.csv

# Chia sẻ file cho các thành viên khác:
# - training/outputs/X_test.npy
# - training/outputs/y_test.npy
# - training/outputs/scaler_standard.pkl
```

### Phân Công Chi Tiết

| Thành Viên | Nhiệm Vụ | Model |
|------------|----------|-------|
| **Dương** | Train LSTM, tổng hợp kết quả vào Excel | LSTM |
| **Thiện** | Train Hybrid, phân tích Confusion Matrix | Hybrid |
| **Nguyên** | Train CNN, vẽ biểu đồ so sánh | CNN |

### Checklist Sau Training

- [ ] Lưu model weights (*.pt)
- [ ] Lưu training history (*.json)
- [ ] Ghi lại thời gian training
- [ ] Chia sẻ test set cho các thành viên
- [ ] Chạy evaluate.py trên cùng test set
- [ ] Vẽ biểu đồ so sánh

---

## So Sánh Kiến Trúc Các Mô Hình

### 1. CNN 1D

```
Input: (batch, 20, 15)
├── Conv1d: 15 → 64 → 128 → 256 channels
├── MaxPooling + BatchNorm + Dropout
├── Global AdaptiveMaxPool
└── FC: 256 → 128 → 64 → 1

Đặc điểm: Nhanh, trích xuất features cục bộ
```

### 2. LSTM

```
Input: (batch, 20, 15)
├── LSTM Layer 1: 15 → 128 hidden
├── LSTM Layer 2: 128 → 64 hidden
├── Lấy output timestep cuối
└── FC: 64 → 64 → 32 → 1

Đặc điểm: FPR thấp nhất, học temporal patterns
```

### 3. Hybrid CNN-LSTM (KHÔNG Pooling)

```
Input: (batch, 20, 15)
├── CNN Block (KHÔNG Pooling):
│   └── Conv1d: 15 → 64 → 128 channels
├── LSTM Block:
│   └── LSTM: 128 → 64 hidden, 2 layers
└── FC: 64 → 32 → 1

Đặc điểm: Kết hợp CNN + LSTM, giữ thông tin temporal
```

### Tại Sao Hybrid Không Dùng Pooling?

| Hybrid với Pooling | Hybrid không Pooling |
|--------------------|----------------------|
| FPR ~12.8% | FPR ~2-3% |
| Mất thông tin temporal | Giữ nguyên thông tin |
| LSTM khó học patterns | LSTM học tốt hơn |

---

## Xử Lý Các Trường Hợp Đặc Biệt

### Trường Hợp 1: Hybrid Kém Hơn CNN/LSTM

**Nguyên nhân có thể**:
- Overfitting do model phức tạp
- Bot-IoT dataset quá "dễ" (CNN đã đạt 99.9%)

**Giải pháp**:
- Tăng Dropout
- So sánh độ ổn định (Loss curve mượt hơn)
- Nhấn mạnh training time ngắn hơn

### Trường Hợp 2: Tất Cả Model Đạt 99.99%

**Đây là đặc điểm của Bot-IoT** (dữ liệu rõ ràng)

**Giải pháp**:
- So sánh ở hàng phần nghìn (99.99% vs 99.95%)
- So sánh FPR và FNR
- So sánh thời gian training và inference

### Trường Hợp 3: Training Quá Chậm

**Giải pháp**:
- Đảm bảo đang dùng GPU
- Giảm batch size nếu hết VRAM
- Sử dụng Mixed Precision (đã bật sẵn)

---

## Chạy Web Demo

### Chuẩn Bị Models Cho Demo

```bash
# Copy models từ training sang backend
cp training/outputs/*_best.pt backend/models/
cp training/outputs/scaler_standard.pkl backend/models/
```

### Chạy Server

```bash
python app.py
```

### Mở Dashboard

- **Dashboard mới**: http://localhost:5000 (Advanced Dashboard)
- **Demo cũ**: http://localhost:5000/old (Replay only)

### ⭐ Tính Năng Dashboard Mới

| Tính năng | Mô tả |
|-----------|-------|
| **📊 Dashboard** | Tổng quan metrics, ranking models theo FPR |
| **🧠 Model Evaluation** | So sánh Accuracy, FPR, FNR, ROC-AUC, Confusion Matrix |
| **📡 Real-time Monitor** | Replay traffic, theo dõi predictions thời gian thực |
| **⚙️ Training** | Train models mới với epochs, batch size, learning rate tùy chỉnh |
| **💾 Dataset Manager** | Xem thông tin dataset, chọn custom dataset path |
| **⚖️ Compare Results** | So sánh kết quả cũ vs mới, tính improvement |
| **📜 History & Reports** | Lịch sử training/evaluation, classification reports |

### API Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/api/models/list` | GET | Danh sách models và metrics |
| `/api/models/evaluate` | POST | Chạy evaluation |
| `/api/training/start` | POST | Bắt đầu training |
| `/api/training/stop` | POST | Dừng training |
| `/api/training/status` | GET | Trạng thái training |
| `/api/dataset/info` | GET | Thông tin dataset |
| `/api/history` | GET | Lịch sử training/evaluation |
| `/api/compare` | GET | So sánh kết quả cũ/mới |
| `/api/system/info` | GET | Thông tin hệ thống |

---

## Talking Points Cho Hội Đồng

### 1. Giới Thiệu

> "Hệ thống so sánh 3 mô hình Deep Learning cho phát hiện DDoS trong IoT, với đánh giá khách quan trên cùng test set."

### 2. Kết Quả

> "LSTM có FPR thấp nhất (0.7%), phù hợp cho production. Hybrid kết hợp ưu điểm của cả CNN và LSTM."

### 3. Điểm Nhấn Kỹ Thuật

> "Hybrid không dùng Pooling để giữ thông tin temporal, giảm FPR từ 12.8% xuống 2-3%."

### 4. Hướng Phát Triển

> "Parallel Hybrid, Attention Mechanism, Ensemble Methods."

---

## Bộ Dữ Liệu Bot-IoT

### Nguồn Gốc

- **Tên**: Bot-IoT Dataset
- **Tác giả**: UNSW Canberra Cyber Security
- **Năm**: 2018

### 15 Đặc Trưng Sử Dụng

| Feature | Mô Tả |
|---------|-------|
| pkts | Số lượng packets |
| bytes | Tổng bytes |
| dur | Thời gian flow |
| mean, stddev, sum, min, max | Thống kê packet size |
| spkts, dpkts | Packets nguồn/đích |
| sbytes, dbytes | Bytes nguồn/đích |
| rate, srate, drate | Tốc độ packets |

---

## Tham Khảo

1. Bot-IoT Dataset - UNSW Canberra (2018)
2. IEEE Access: CNN-LSTM for DDoS Detection
3. Scientific Reports (2025): "LSTM uses CNN's output as input"
4. PyTorch Documentation

---

## Tác Giả

**Nhóm Nghiên Cứu An Ninh IoT - 2026**

- Dương: LSTM Model
- Thiện: Hybrid Model
- Nguyên: CNN Model

---

## Giấy Phép

Chỉ sử dụng cho mục đích học thuật và nghiên cứu.
