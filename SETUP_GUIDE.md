# 🚀 HƯỚNG DẪN SETUP VÀ DEMO - ĐẦY ĐỦ

## 📋 CHECKLIST TRƯỚC KHI BẮT ĐẦU

- [ ] Python 3.8+ đã cài đặt
- [ ] PyTorch đã cài đặt
- [ ] Đã train xong 3 models (CNN, LSTM, Hybrid)
- [ ] Có file scaler_standard.pkl
- [ ] Có dataset Bot-IoT (botiot.csv)

---

## 🎯 BƯỚC 1: COPY MODEL WEIGHTS

Copy các file sau từ `D:\Project\IoT\Trainning\outputs_standard\` vào `DemoWeb_3Models\backend\models\`:

```
backend/models/
├── CNN_best.pt              ← từ outputs_standard/CNN_best.pt
├── LSTM_best.pt             ← từ outputs_standard/LSTM_best.pt
├── Hybrid_CNN_LSTM_best.pt  ← từ outputs_standard/Hybrid_CNN_LSTM_best.pt (hoặc new_hybrid_cnn_lstm_best.pt)
└── scaler_standard.pkl      ← từ Trainning/processed_data/scaler_standard.pkl
```

### PowerShell Commands:

```powershell
# Tạo thư mục models
New-Item -ItemType Directory -Path "backend\models" -Force

# Copy models
Copy-Item "D:\Project\IoT\Trainning\outputs_standard\CNN_best.pt" -Destination "backend\models\"
Copy-Item "D:\Project\IoT\Trainning\outputs_standard\LSTM_best.pt" -Destination "backend\models\"
Copy-Item "D:\Project\IoT\Trainning\outputs_standard\Hybrid_CNN_LSTM_best.pt" -Destination "backend\models\"

# Copy scaler
Copy-Item "D:\Project\IoT\Trainning\processed_data\scaler_standard.pkl" -Destination "backend\models\"
```

---

## 🎯 BƯỚC 2: INSTALL DEPENDENCIES

```powershell
cd D:\Project\IoT\DemoWeb_3Models
pip install -r requirements.txt
```

**Lưu ý:** Nếu PyTorch chưa cài, install riêng:
```powershell
# CPU only
pip install torch torchvision torchaudio

# CUDA (GPU) - Nếu có GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎯 BƯỚC 3: CHUẨN BỊ DEMO DATA

Chạy script tự động tạo demo data:

```powershell
python prepare_demo_data.py
```

**Script này sẽ:**
- Load Bot-IoT dataset
- Skip 2M rows đầu (training data)
- Lấy 500 Normal + 500 Attack
- Tạo file `data/demo_test.csv`

**Kiểm tra:**
```powershell
# Kiểm tra file đã tạo
dir data\demo_test.csv

# Xem vài dòng đầu
Get-Content data\demo_test.csv -Head 5
```

---

## 🎯 BƯỚC 4: RUN SERVER

```powershell
python app.py
```

**Bạn sẽ thấy:**
```
======================================================================
🚀 Bot-IoT Multi-Model Demo Server
======================================================================
✅ Scaler loaded from backend/models/scaler_standard.pkl
✅ CNN model loaded from backend/models/CNN_best.pt
✅ LSTM model loaded from backend/models/LSTM_best.pt
✅ Hybrid model loaded from backend/models/Hybrid_CNN_LSTM_best.pt
✅ All 3 models loaded successfully!
✅ ReplayDetector initialized successfully
Models: ['CNN', 'LSTM', 'Hybrid']
Device: cuda (hoặc cpu)
======================================================================
 * Running on http://0.0.0.0:5000
```

---

## 🎯 BƯỚC 5: MỞ DASHBOARD

1. Mở browser: **http://localhost:5000**

2. Bạn sẽ thấy dashboard với:
   - 3 Model Cards (CNN, LSTM, Hybrid)
   - Control Panel (Start/Stop buttons)
   - Live Chart
   - Statistics
   - Traffic Log

3. Nhấn **"Start Replay"** để bắt đầu demo

---

## 🎮 SỬ DỤNG DASHBOARD

### Control Panel:
- **▶️ Start Replay**: Bắt đầu phát lại traffic
- **⏹️ Stop Replay**: Dừng lại
- **Speed**: Chọn tốc độ (Fast/Medium/Normal/Slow)

### Model Cards:
Mỗi card hiển thị real-time:
- **Confidence**: Độ tin cậy (%)
- **Prediction**: Normal 🟢 hoặc Attack 🔴  
- **Attacks Detected**: Số lần phát hiện attack
- **Progress Bar**: Xác suất attack (0-100%)

### Live Chart:
- 3 đường màu (Blue=CNN, Green=LSTM, Purple=Hybrid)
- Trục Y: Attack Probability (0-1)
- Trục X: Packet Number
- Tự động update

### Statistics:
- **Ground Truth**: Số lượng thực tế (Normal/Attack)
- **Consensus**: Khi 2/3 models đồng ý

### Traffic Log:
- Real-time packet predictions
- Màu xanh 🟢 = Normal
- Màu đỏ 🔴 = Attack

---

## 📊 DEMO SCENARIOS

### Scenario 1: Kiểm tra Accuracy

**Mục tiêu:** So sánh predictions với ground truth

**Cách làm:**
1. Start replay với speed Normal (0.1s)
2. Quan sát Traffic Log: So sánh "True" vs "Predictions"
3. Sau 100 packets, check Statistics:
   - True Normal: X
   - True Attacks: Y
   - Attacks Detected (mỗi model): Z

**Kỳ vọng:**
- **LSTM**: Phát hiện gần đúng số attacks thực tế (high accuracy)
- **CNN**: Có thể bỏ sót một vài attacks
- **Hybrid**: Tùy thuộc architecture (no pooling = tốt, có pooling = kém)

---

### Scenario 2: Kiểm tra False Positive Rate

**Mục tiêu:** Xem model nào báo động giả nhiều nhất

**Cách làm:**
1. Chỉ xem những packet có True Label = 🟢 Normal
2. Đếm xem model nào predict nhầm thành 🔴 Attack

**Kỳ vọng:**
- **LSTM**: FPR thấp nhất (~0.7%) → Ít báo động giả
- **Hybrid (có pooling)**: FPR cao (~12%) → Nhiều báo động giả
- **Hybrid (no pooling)**: FPR thấp (~2-3%)

---

### Scenario 3: Real-time Performance

**Mục tiêu:** Model nào phản ứng nhanh nhất

**Cách làm:**
1. Start replay với speed Fast (0.01s)
2. Quan sát Animation và Chart update

**Kỳ vọng:**
- Tất cả 3 models update đồng thời (vì chạy song song)
- Chart mượt mà không lag

---

### Scenario 4: Consensus Detection

**Mục tiêu:** Khi nào 2/3 models đồng ý?

**Cách làm:**
1. Theo dõi "Model Consensus" counter
2. So sánh với từng model riêng lẻ

**Kỳ vọng:**
- Consensus count < mỗi model riêng lẻ
- Khi consensus tăng = High confidence attack!

---

## 🎓 TALKING POINTS CHO HỘI ĐỒNG

### 1. **Giới thiệu Demo:**

> *"Em xin phép demo hệ thống Real-time Detection với 3 mô hình Deep Learning. Hệ thống này replay traffic từ Bot-IoT test set, cho phép quan sát trực quan hiệu năng của từng mô hình."*

### 2. **Giải thích Dashboard:**

> *"Dashboard hiển thị real-time predictions của 3 models:*
> - *CNN: Trích xuất features không gian*
> - *LSTM: Học temporal patterns*
> - *Hybrid: Kết hợp cả hai*
>
> *Biểu đồ dưới cho thấy Attack Probability theo thời gian của cả 3 models."*

### 3. **Phân tích Kết quả:**

> *"Qua demo, em quan sát thấy:*
> - *LSTM có FPR thấp nhất (0.7%), phù hợp cho production*
> - *Hybrid với pooling có FPR cao (12.8%), do mất thông tin temporal*
> - *Khi 2/3 models đồng ý, confidence tăng lên đáng kể"*

### 4. **So sánh với Papers:**

> *"Kết quả này phù hợp với nghiên cứu của [cite paper] khi chỉ ra LSTM thuần túy có thể vượt trội hơn Hybrid nếu Hybrid architecture chưa optimize."*

### 5. **Future Work:**

> *"Hướng phát triển tiếp theo: Implement Hybrid architecture song song (Parallel Hybrid) như đề xuất trong paper [XYZ] để cải thiện FPR."*

---

## 🐛 TROUBLESHOOTING

### Lỗi: "Model not found"
```powershell
# Kiểm tra models đã copy chưa
dir backend\models\

# Phải có 4 files:
# - CNN_best.pt
# - LSTM_best.pt
# - Hybrid_CNN_LSTM_best.pt
# - scaler_standard.pkl
```

### Lỗi: "CSV file not found"
```powershell
# Chạy lại script chuẩn bị data
python prepare_demo_data.py
```

### Lỗi: "CUDA out of memory"
Sửa trong `replay_detector.py`:
```python
# Line ~8: Force CPU
self.device = torch.device('cpu')
```

### Lỗi: "WebSocket connection failed"
- Tắt firewall tạm thời
- Thử `http://127.0.0.1:5000` thay vì `localhost`

---

## 📸 SCREENSHOT CHO BÁO CÁO

Chụp màn hình các phần sau:

1. **Dashboard Overview**: Toàn bộ giao diện với 3 model cards
2. **Live Chart**: Biểu đồ 3 đường màu đang chạy
3. **Statistics**: So sánh số liệu ground truth vs detections
4. **Traffic Log**: Hiển thị predictions chi tiết
5. **Consensus Example**: Khi cả 3 models đều báo 🔴 Attack

---

## 🎬 VIDEO DEMO SCRIPT

**Timeline (2-3 phút):**

0:00 - 0:15: Giới thiệu dashboard  
0:15 - 0:30: Giải thích 3 models  
0:30 - 0:45: Click Start Replay  
0:45 - 1:30: Theo dõi predictions real-time  
1:30 - 2:00: Phân tích chart (LSTM stable, Hybrid spike nhiều)  
2:00 - 2:30: So sánh statistics  
2:30 - 3:00: Kết luận: LSTM tốt nhất cho Bot-IoT  

---

## ✅ CHECKLIST TRƯỚC BUỔI DEMO

- [ ] Đã test chạy ít nhất 1 lần thành công
- [ ] Models load nhanh (<5s)
- [ ] Dashboard hiển thị đúng trên Chrome/Firefox  
- [ ] Replay chạy mượt không lag
- [ ] Chart update real-time
- [ ] Đã chụp screenshot backup (phòng demo bị lỗi)
- [ ] Đã record video demo backup
- [ ] Đã chuẩn bị script talking points
- [ ] Internet ổn định (nếu demo online)

---

## 🎯 TẠI SAO CÁCH NÀY TỐT HƠN BẮT GÓI TIN THẬT?

1. **Reproducible**: Chạy lại nhiều lần, kết quả giống nhau
2. **Controllable**: Chọn đúng data có Normal + Attack
3. **Feature Accuracy**: Đảm bảo features đúng 100% với lúc train
4. **No Setup**: Không cần config network, firewall, tấn công thật
5. **Safe**: Không rủi ro làm hỏng mạng thật
6. **Fast**: Setup trong 10 phút vs setup thật 1-2 ngày

---

## 📚 REFERENCES

1. Bot-IoT Dataset: UNSW Canberra (2018)
2. Scientific Reports (2025): "LSTM uses CNN's output as input"
3. IEEE Access: CNN→LSTM for DDoS Detection

---

**Chúc bạn demo thành công và đạt điểm cao! 🎉**
