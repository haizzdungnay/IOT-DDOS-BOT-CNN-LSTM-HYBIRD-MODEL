# Hệ Thống Phát Hiện Tấn Công DDoS IoT Sử Dụng Mô Hình Lai CNN-LSTM

## Giới Thiệu Dự Án

Đây là hệ thống demo phát hiện tấn công DDoS trong mạng IoT (Internet of Things) sử dụng 3 mô hình Deep Learning:

- **CNN 1D (Convolutional Neural Network)**: Trích xuất đặc trưng không gian
- **LSTM (Long Short-Term Memory)**: Mô hình hóa chuỗi thời gian
- **Hybrid CNN-LSTM**: Mô hình lai kết hợp cả hai (theo chuẩn IEEE)

Hệ thống phát lại (replay) traffic thực tế từ bộ dữ liệu Bot-IoT để so sánh hiệu năng của từng mô hình một cách trực quan.

---

## Đánh Giá Dự Án

### Điểm Mạnh

| Tiêu chí | Đánh giá |
|----------|----------|
| **Kiến trúc hệ thống** | Thiết kế module rõ ràng, tách biệt backend/frontend |
| **Giao diện người dùng** | Dashboard trực quan với biểu đồ real-time |
| **So sánh đa mô hình** | Cho phép đánh giá song song 3 mô hình cùng lúc |
| **Cơ chế đồng thuận** | Voting 2/3 tăng độ tin cậy của dự đoán |
| **WebSocket** | Cập nhật real-time không cần refresh trang |
| **Tái tạo kết quả** | Replay từ CSV đảm bảo reproducibility |

### Điểm Cần Cải Thiện

- Chưa có đánh giá metrics chi tiết (Precision, Recall, F1-Score)
- Chưa hỗ trợ capture traffic thực tế
- Chưa có tính năng export báo cáo tự động

---

## So Sánh Kiến Trúc Các Mô Hình

### 1. Mô Hình CNN 1D

```
Input: (batch_size, 20, 15) - 20 timesteps, 15 features

Kiến trúc:
├── Conv1d: 15 → 64 channels, kernel=3
├── BatchNorm1d + ReLU + MaxPool1d(2) + Dropout(0.4)
├── Conv1d: 64 → 128 channels, kernel=3
├── BatchNorm1d + ReLU + MaxPool1d(2) + Dropout(0.4)
├── Conv1d: 128 → 256 channels, kernel=3
├── BatchNorm1d + ReLU + AdaptiveMaxPool1d(1)
├── FC: 256 → 128 → 64
└── Output: Sigmoid (0 = Bình thường, 1 = Tấn công)

Đặc điểm:
- Sử dụng Pooling để giảm chiều dữ liệu
- Tốt cho trích xuất đặc trưng cục bộ
- Không học được dependencies dài hạn
```

### 2. Mô Hình LSTM

```
Input: (batch_size, 20, 15) - 20 timesteps, 15 features

Kiến trúc:
├── LSTM Layer 1: 15 → 128 hidden units
├── Dropout(0.3)
├── LSTM Layer 2: 128 → 64 hidden units
├── Dropout(0.4)
├── FC: 64 → 64 + BatchNorm1d + ReLU
├── FC: 64 → 32 + ReLU
└── Output: Sigmoid (0 = Bình thường, 1 = Tấn công)

Đặc điểm:
- Học được patterns tuần tự trong traffic
- Nhớ được thông tin dài hạn (Long-term dependencies)
- FPR thấp nhất (~0.7%) trên Bot-IoT
```

### 3. Mô Hình Hybrid CNN-LSTM (Không Pooling)

```
Input: (batch_size, 20, 15) - 20 timesteps, 15 features

Kiến trúc:
├── CNN Block (KHÔNG có Pooling):
│   ├── Conv1d: 15 → 64 channels, kernel=3, padding=1
│   ├── BatchNorm1d + ReLU + Dropout(0.3)
│   ├── Conv1d: 64 → 128 channels, kernel=3, padding=1
│   └── BatchNorm1d + ReLU + Dropout(0.3)
│
├── LSTM Block:
│   ├── LSTM: 128 → 64 hidden units, 2 layers
│   └── Dropout(0.3)
│
└── Dense Block:
    ├── FC: 64 → 32 + ReLU + Dropout(0.4)
    └── Output: Sigmoid

Đặc điểm:
- CNN trích xuất features → LSTM học temporal patterns
- KHÔNG có Pooling để giữ nguyên độ phân giải thời gian
- Kết hợp ưu điểm của cả CNN và LSTM
```

### Tại Sao Không Dùng Pooling Trong Hybrid?

| Hybrid với Pooling | Hybrid không Pooling |
|--------------------|----------------------|
| FPR cao (~12.8%) | FPR thấp (~2-3%) |
| Mất thông tin temporal | Giữ nguyên thông tin |
| LSTM khó học patterns | LSTM học tốt hơn |

---

## So Sánh Phương Pháp Đánh Giá

### Các Metrics Được Theo Dõi

| Metric | Mô tả | Cách tính |
|--------|-------|-----------|
| **True Attacks** | Số mẫu tấn công thực tế | Đếm label = 1 trong ground truth |
| **True Normal** | Số mẫu bình thường thực tế | Đếm label = 0 trong ground truth |
| **Attacks Detected** | Số dự đoán tấn công của mỗi model | Đếm prediction = 1 |
| **Correct/Wrong** | Số dự đoán đúng/sai | So sánh với ground truth |
| **Consensus Attacks** | Khi ≥2/3 model đồng ý Attack | Voting system |

### So Sánh Hiệu Năng Thực Tế

| Model | FPR (False Positive Rate) | Đặc điểm | Đánh giá |
|-------|---------------------------|----------|----------|
| **LSTM** | ~0.7% | Ít báo động giả nhất | Tốt nhất cho Bot-IoT |
| **CNN 1D** | ~1-2% | Ổn định | Phù hợp real-time |
| **Hybrid (no pooling)** | ~2-3% | Cân bằng | Khuyến nghị cho research |
| **Hybrid (có pooling)** | ~12.8% | Nhiều báo động giả | Không khuyến nghị |

### Cơ Chế Đồng Thuận (Consensus)

```python
# Logic voting 2/3
attack_votes = sum([model1_pred, model2_pred, model3_pred])
if attack_votes >= 2:
    consensus = "ATTACK"  # Độ tin cậy cao
else:
    consensus = "NORMAL"
```

**Ý nghĩa**: Khi ≥2 mô hình cùng dự đoán là Attack, hệ thống có độ tin cậy cao hơn so với dựa vào một mô hình đơn lẻ.

---

## Cấu Trúc Dự Án

```
IOT-DDOS-BOT-CNN-LSTM-HYBIRD-MODEL/
├── app.py                          # Flask backend với WebSocket
├── prepare_demo_data.py            # Script chuẩn bị dữ liệu
├── requirements.txt                # Dependencies
├── README.md                       # Tài liệu này
├── SETUP_GUIDE.md                  # Hướng dẫn setup chi tiết
│
├── backend/
│   ├── replay_detector.py          # Logic phát hiện đa mô hình
│   └── models/
│       ├── CNN_best.pt             # Trọng số CNN
│       ├── LSTM_best.pt            # Trọng số LSTM
│       ├── Hybrid_CNN_LSTM_best.pt # Trọng số Hybrid
│       └── scaler_standard.pkl     # StandardScaler
│
├── data/
│   └── demo_test.csv               # Dữ liệu test (500 Normal + 500 Attack)
│
└── public/
    └── index.html                  # Dashboard web
```

---

## Bộ Dữ Liệu Bot-IoT

### Nguồn Gốc
- **Tên**: Bot-IoT Dataset
- **Tác giả**: UNSW Canberra Cyber Security
- **Năm**: 2018
- **Mô tả**: Traffic giả lập từ botnet trong mạng IoT

### Đặc Trưng Sử Dụng (15 Features)

| # | Feature | Mô tả |
|---|---------|-------|
| 1 | `pkts` | Số lượng packets |
| 2 | `bytes` | Tổng số bytes |
| 3 | `dur` | Thời gian flow (giây) |
| 4 | `mean` | Kích thước packet trung bình |
| 5 | `stddev` | Độ lệch chuẩn |
| 6 | `sum` | Tổng kích thước |
| 7 | `min` | Kích thước nhỏ nhất |
| 8 | `max` | Kích thước lớn nhất |
| 9 | `spkts` | Packets từ nguồn |
| 10 | `dpkts` | Packets từ đích |
| 11 | `sbytes` | Bytes từ nguồn |
| 12 | `dbytes` | Bytes từ đích |
| 13 | `rate` | Tốc độ packet |
| 14 | `srate` | Tốc độ nguồn |
| 15 | `drate` | Tốc độ đích |

### Phân Bố Dữ Liệu Demo

- **Tổng**: 1,000 mẫu
- **Normal (attack=0)**: 500 mẫu (50%)
- **Attack (attack=1)**: 500 mẫu (50%)

---

## Hướng Dẫn Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.8+
- PyTorch 2.0+
- RAM: 4GB+
- GPU: Tùy chọn (hỗ trợ CUDA)

### Bước 1: Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Chuẩn Bị Models

Copy các file model vào `backend/models/`:
- `CNN_best.pt`
- `LSTM_best.pt`
- `Hybrid_CNN_LSTM_best.pt`
- `scaler_standard.pkl`

### Bước 3: Chuẩn Bị Dữ Liệu

```bash
python prepare_demo_data.py
```

### Bước 4: Chạy Server

```bash
python app.py
```

Server khởi động tại: **http://localhost:5000**

---

## Hướng Dẫn Sử Dụng Dashboard

### Bảng Điều Khiển

| Nút | Chức năng |
|-----|-----------|
| **Start Replay** | Bắt đầu phát lại traffic |
| **Stop Replay** | Dừng phát lại |
| **Speed** | Chọn tốc độ (0.01s - 0.5s/packet) |

### Thành Phần Dashboard

1. **Model Cards**: Hiển thị dự đoán real-time của 3 mô hình
   - Confidence (%)
   - Prediction (Normal/Attack)
   - Progress bar xác suất tấn công

2. **Live Chart**: Biểu đồ đường theo dõi Attack Probability
   - Xanh dương: CNN
   - Xanh lá: LSTM
   - Tím: Hybrid

3. **Statistics**: Thống kê Ground Truth và Consensus

4. **Traffic Log**: Nhật ký dự đoán từng packet

---

## Kịch Bản Demo

### Kịch Bản 1: Kiểm Tra Độ Chính Xác

1. Bắt đầu replay với tốc độ 0.1s
2. Theo dõi Traffic Log
3. So sánh predictions với True Label
4. Kết luận: LSTM có độ chính xác cao nhất

### Kịch Bản 2: Phân Tích False Positive

1. Quan sát các mẫu Normal (True = Normal)
2. Đếm số lần model báo nhầm là Attack
3. Kết luận: LSTM có FPR thấp nhất

### Kịch Bản 3: Đánh Giá Consensus

1. Theo dõi counter "Consensus Attacks"
2. So sánh với từng model riêng lẻ
3. Kết luận: Consensus tăng độ tin cậy

---

## Kết Luận Và Khuyến Nghị

### Kết Quả Nghiên Cứu

| Mô hình | Ưu điểm | Nhược điểm | Khuyến nghị |
|---------|---------|------------|-------------|
| **LSTM** | FPR thấp nhất, ổn định | Chậm hơn CNN | Production |
| **CNN 1D** | Nhanh, đơn giản | Không học temporal | Real-time |
| **Hybrid** | Kết hợp ưu điểm | Phức tạp hơn | Research |

### Đề Xuất Cải Tiến

1. **Parallel Hybrid**: Chạy CNN và LSTM song song thay vì tuần tự
2. **Attention Mechanism**: Thêm cơ chế attention để focus vào features quan trọng
3. **Ensemble Methods**: Kết hợp nhiều mô hình với voting có trọng số
4. **Online Learning**: Cập nhật model theo thời gian thực

---

## Tham Khảo

1. Bot-IoT Dataset - UNSW Canberra Cyber Security (2018)
2. IEEE Papers: CNN-LSTM for DDoS Detection
3. Scientific Reports (2025): "LSTM uses CNN's output as input"
4. PyTorch Documentation

---

## Tác Giả

**Nhóm Nghiên Cứu An Ninh IoT - 2026**

---

## Giấy Phép

Chỉ sử dụng cho mục đích học thuật và nghiên cứu.
