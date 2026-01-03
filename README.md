# Hệ Thống Phát Hiện Tấn Công DDoS IoT Sử Dụng Mô Hình Lai CNN-LSTM

## Giới Thiệu Dự Án

Đây là hệ thống phát hiện tấn công DDoS trong mạng IoT (Internet of Things) sử dụng 4 mô hình Deep Learning:

- **CNN 1D (Convolutional Neural Network)**: Trích xuất đặc trưng không gian
- **LSTM (Long Short-Term Memory)**: Mô hình hóa chuỗi thời gian
- **Hybrid CNN-LSTM**: Mô hình lai tuần tự CNN → LSTM (theo chuẩn IEEE)
- **Parallel Hybrid**: CNN và LSTM song song, concatenate features

### Tính Năng Chính

- Training thống nhất cho cả 4 models với cùng cách đánh giá
- Hỗ trợ dữ liệu đã tiền xử lý (processed_data/) hoặc CSV gốc
- Class weights cho dữ liệu mất cân bằng
- Tối ưu cho GPU (CUDA) với Automatic Mixed Precision
- Đánh giá khách quan với Confusion Matrix, Classification Report
- Dashboard demo real-time so sánh các models
- Hệ thống voting 2/3 tăng độ tin cậy

---

## Cấu Trúc Dự Án

```
IOT-DDOS-BOT-CNN-LSTM-HYBIRD-MODEL/
│
├── processed_data/                 # ⭐ Dữ liệu đã tiền xử lý (sequences, scaled)
│   ├── X_train_seq.npy             # Train features (2.1M samples)
│   ├── y_train_seq.npy             # Train labels
│   ├── X_val_seq.npy               # Validation features
│   ├── y_val_seq.npy               # Validation labels
│   ├── X_test_seq.npy              # Test features (450K samples)
│   ├── y_test_seq.npy              # Test labels
│   ├── config.pkl                  # Cấu hình (time_steps, features)
│   ├── class_weights.pkl           # Weights cho imbalanced data
│   └── scaler_standard.pkl         # StandardScaler
│
├── training/                       # Module training & evaluation
│   ├── config.py                   # Cấu hình chung (GPU, features, hyperparameters)
│   ├── data_loader.py              # Load dữ liệu (CSV hoặc .npy)
│   ├── models.py                   # Định nghĩa 4 models (CNN, LSTM, Hybrid, Parallel)
│   ├── trainer.py                  # Training class với Early Stopping + Class Weights
│   ├── train_processed.py          # ⭐ Train với dữ liệu đã xử lý
│   ├── train_all.py                # Train từ CSV gốc
│   ├── evaluate_processed.py       # ⭐ Đánh giá với processed data
│   ├── evaluate.py                 # Đánh giá cũ
│   ├── visualize.py                # Vẽ biểu đồ so sánh
│   ├── outputs/                    # Model weights và test set
│   └── logs/                       # Training history và reports
│
├── backend/                        # Web demo backend
│   ├── replay_detector.py          # Logic phát hiện đa mô hình
│   ├── api_routes.py               # ⭐ APIs cho dashboard nâng cao
│   └── models/                     # Model weights cho demo
│       ├── CNN_best.pt
│       ├── LSTM_best.pt
│       ├── Hybrid_CNN_LSTM_best.pt
│       ├── Parallel_Hybrid_best.pt
│       └── scaler_standard.pkl
│
├── data/                           # Dữ liệu demo
│   ├── demo_test.csv               # 1000 samples cho web demo
│   └── training_history.json       # Lịch sử training/evaluation
│
├── public/                         # Frontend dashboard
│   ├── dashboard.html              # ⭐ Advanced Dashboard mới
│   ├── index.html                  # Demo cũ (replay only)
│   └── static/js/
│       └── dashboard.js            # JavaScript cho dashboard
│
├── app.py                          # Flask server
├── requirements.txt                # Dependencies
├── RUN_GUIDE.md                    # Hướng dẫn chạy nhanh
└── README.md

---

## Hướng Dẫn Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8+
- PyTorch 2.0+ (khuyến nghị GPU)
- RAM: 8GB+
- GPU: NVIDIA với CUDA 11.8+ (khuyến nghị)

### Bước 1: Clone Repository

```bash
git clone https://github.com/your-repo/IOT-DDOS-BOT-CNN-LSTM-HYBIRD-MODEL.git
cd IOT-DDOS-BOT-CNN-LSTM-HYBIRD-MODEL
```

### Bước 2: Tạo Virtual Environment

```bash
python -m venv .venv

# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Bước 3: Cài Đặt PyTorch (Chọn GPU hoặc CPU)

```bash
# GPU với CUDA 11.8 (khuyến nghị)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU với CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (chậm hơn nhiều)
pip install torch torchvision torchaudio
```

### Bước 4: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 5: Kiểm Tra GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## Hướng Dẫn Training

### ⭐ Training với Dữ Liệu Đã Xử Lý (Khuyến Nghị)

Sử dụng dữ liệu từ `processed_data/` (đã có sẵn sequences, scaled, class weights):

```bash
cd training

# Train tất cả models với class weights
python train_processed.py

# Train model cụ thể
python train_processed.py --models CNN
python train_processed.py --models LSTM
python train_processed.py --models Hybrid
python train_processed.py --models Parallel

# Tùy chỉnh epochs
python train_processed.py --epochs 30 --models LSTM

# Không dùng class weights (không khuyến nghị cho dữ liệu mất cân bằng)
python train_processed.py --no-weights
```

### Training từ CSV Gốc

```bash
cd training
python train_all.py --data /path/to/botiot.csv --epochs 50
```

### Training Một Model Cụ Thể (CSV)

```bash
python train_all.py --data /path/to/botiot.csv --models CNN --epochs 50
python train_all.py --data /path/to/botiot.csv --models LSTM --epochs 50
python train_all.py --data /path/to/botiot.csv --models Hybrid --epochs 50
```

### Kết Quả Training

Sau khi training, các file sau sẽ được tạo:

```
training/
├── outputs/
│   ├── CNN_best.pt                 # Trọng số CNN tốt nhất
│   ├── LSTM_best.pt                # Trọng số LSTM tốt nhất
│   ├── Hybrid_best.pt              # Trọng số Hybrid tốt nhất
│   ├── scaler_standard.pkl         # Scaler để chuẩn hóa dữ liệu
│   ├── X_test.npy                  # Test set features (DÙNG CHUNG!)
│   ├── y_test.npy                  # Test set labels (DÙNG CHUNG!)
│   └── data_metadata.json          # Thông tin dữ liệu
│
└── logs/
    ├── CNN_history.json            # Lịch sử training CNN
    ├── LSTM_history.json           # Lịch sử training LSTM
    ├── Hybrid_history.json         # Lịch sử training Hybrid
    └── training_summary.json       # Tóm tắt training
```

---

## Hướng Dẫn Đánh Giá Models

### ⭐ Đánh Giá với Processed Data (Khuyến Nghị)

```bash
cd training

# Đánh giá tất cả models
python evaluate_processed.py

# Đánh giá model cụ thể
python evaluate_processed.py --models CNN LSTM

# Dùng models từ backend/models
python evaluate_processed.py --model-dir ../backend/models
```

### Đánh Giá Cũ (với test set từ training/outputs)

```bash
cd training
python evaluate.py
```

### Các Metrics Được Đánh Giá

| Metric | Ý Nghĩa |
|--------|---------|
| **Accuracy** | Tỷ lệ dự đoán đúng tổng thể |
| **Precision** | Tỷ lệ dự đoán Attack đúng |
| **Recall** | Tỷ lệ phát hiện được Attack (quan trọng nhất!) |
| **F1-Score** | Trung bình hài hòa của Precision và Recall |
| **FPR** | False Positive Rate - Tỷ lệ báo động giả |
| **FNR** | False Negative Rate - Tỷ lệ bỏ sót tấn công |

---

## Vẽ Biểu Đồ So Sánh

```bash
cd training
python visualize.py
```

### Các Biểu Đồ Được Tạo

1. **training_curves.png**: Loss và Accuracy qua các epochs
2. **confusion_matrices.png**: Ma trận nhầm lẫn của 3 models
3. **metrics_comparison.png**: So sánh Accuracy, Precision, Recall, F1
4. **fpr_fnr_comparison.png**: So sánh FPR và FNR
5. **training_time_comparison.png**: Thời gian training
6. **summary_table.txt**: Bảng tóm tắt chi tiết

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
