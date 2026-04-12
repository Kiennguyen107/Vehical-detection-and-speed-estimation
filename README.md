# 🚦 Traffic Violation Detection System

Hệ thống phát hiện vi phạm giao thông theo thời gian thực sử dụng Computer Vision và Deep Learning. Tự động phát hiện vi phạm tốc độ, vi phạm khoảng cách an toàn, nhận dạng biển số và màu xe từ camera hoặc file video.

---

## 📸 Demo

> Camera nhìn từ trên xuống → phát hiện xe → tính tốc độ → ghi nhận vi phạm → OCR biển số

---

## ✨ Tính năng chính

- **Phát hiện làn đường** — YOLO segmentation nhận diện vạch kẻ đường (solid/dotted), tính vanishing point, gom cụm DBSCAN, chỉ chạy 1 lần trên frame đầu tiên
- **Theo dõi xe** — YOLO + BotSORT tracking, mỗi xe có ID ổn định xuyên suốt video
- **Ước tính tốc độ** — Perspective transform ROI → mét thực, tính tốc độ từ lịch sử tọa độ
- **Kiểm tra khoảng cách an toàn** — Phát hiện xe đi quá gần theo từng làn, áp dụng ngưỡng theo tốc độ (40–60 km/h → 35m, 60–80 km/h → 55m, >80 km/h → 120m)
- **Nhận dạng biển số** — YOLO detect vùng biển số + PaddleOCR đọc text, chuẩn hóa tự động
- **Phát hiện màu xe** — KMeans clustering trên không gian HSV, 16 clusters
- **Giao diện desktop** — PyQt6, hiển thị video realtime, bảng vi phạm, biểu đồ thống kê
- **Tra cứu web** — Flask app tra cứu lịch sử vi phạm theo biển số
- **Lưu trữ** — PostgreSQL, tự động export Excel khi kết thúc

---

## 🏗️ Kiến trúc hệ thống

```
main.py (PyQt6 UI)
├── lane_detection.py       # Phát hiện làn — chạy 1 lần
├── speed_estimation.py     # Tốc độ + khoảng cách — chạy mỗi frame
├── car_color_detection.py  # Màu xe — background thread
├── database.py             # PostgreSQL
└── app.py                  # Flask web
```

**Luồng xử lý:**

```
Load video
    → Detect làn (YOLO + DBSCAN + polyfit)
    → Chọn ROI (4 điểm hình thang)
    → Vòng lặp video:
        ├── Blend lane overlay
        ├── YOLO track xe
        ├── Perspective transform → mét thực
        ├── Tính tốc độ + kiểm tra khoảng cách
        └── Vi phạm → OCR biển số + detect màu (background)
    → Lưu DB + Export Excel
```

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Object Detection & Tracking | YOLOv8 + BotSORT |
| Lane Detection | YOLOv8 Segmentation + DBSCAN |
| OCR biển số | PaddleOCR |
| Nhận dạng màu | KMeans (scikit-learn) |
| Giao diện | PyQt6 + Matplotlib |
| Web tra cứu | Flask |
| Cơ sở dữ liệu | PostgreSQL (psycopg2) |
| Xử lý ảnh | OpenCV |
| Tính toán | NumPy |

---

## 📋 Yêu cầu hệ thống

- Python 3.9+
- CUDA GPU (khuyến nghị)
- PostgreSQL 14+
- LibreOffice (export Excel)

---

## 🚀 Cài đặt

**1. Clone repo**
```bash
git clone https://github.com/.../traffic-violation-detection.git
cd traffic-violation-detection
```

**2. Cài dependencies**
```bash
pip install -r requirements.txt
```

**3. Tạo database**
```bash
psql -U postgres -c "CREATE DATABASE traffic_violations;"
```

**4. Cấu hình**

Mở `main.py` và chỉnh các đường dẫn:
```python
VIDEO_PATH       = "path/to/video.mp4"
LANE_MODEL_PATH  = "path/to/lane_model.pt"
CAR_MODEL_PATH   = "path/to/car_model.pt"
PLATE_MODEL_PATH = "path/to/plate_model.pt"
```

Cấu hình database trong `database.py`:
```python
DB_CONFIG = {
    'host':     'localhost',
    'port':     5432,
    'database': 'traffic_violations',
    'user':     'postgres',
    'password': 'your_password',
}
```

**5. Chạy ứng dụng desktop**
```bash
python main.py
```

**6. Chạy web tra cứu**
```bash
python app.py
# Truy cập http://localhost:5000
```

---

## 📖 Hướng dẫn sử dụng

**Bước 1 — Setup tab:**
1. Nhấn **Load Video** chọn file hoặc **Connect to Camera**
2. Nhấn **Detect Lanes** — hệ thống tự động phát hiện làn đường
3. Click 4 điểm trên preview để vẽ **ROI** (vùng hình thang mặt đường)
4. Nhấn **Complete**

**Bước 2 — Detection tab:**
1. Nhấn **Start** để bắt đầu xử lý
2. Theo dõi video realtime với bbox và tốc độ từng xe
3. Nhấn **Stop** để dừng — Excel tự động được xuất

**Bước 3 — Violations tab:**
- Xem danh sách vi phạm tốc độ và khoảng cách
- Click vào hàng để xem ảnh frame vi phạm
- Export CSV thủ công nếu cần

---

## 📊 Cấu hình giới hạn tốc độ

```python
LANE_SPEED_LIMITS = {
    1: {"min": 0, "max": 61},
    2: {"min": 0, "max": 71},
    3: {"min": 0, "max": 81},
    4: {"min": 0, "max": 81},
}
```

---

## 🗃️ Cấu trúc database

**Bảng `speed_violations`**

| Cột | Kiểu | Mô tả |
|---|---|---|
| vehicle_id | INTEGER | Track ID của xe |
| speed | INTEGER | Tốc độ (km/h) |
| lane | TEXT | Làn đường |
| plate | TEXT | Biển số xe |
| color | TEXT | Màu xe |
| status | TEXT | TOO_FAST / OK |
| frame_path | TEXT | Đường dẫn ảnh |
| dist_violation | BOOLEAN | Có vi phạm khoảng cách |

**Bảng `distance_violations`**

| Cột | Kiểu | Mô tả |
|---|---|---|
| id_behind | INTEGER | ID xe phía sau |
| distance | REAL | Khoảng cách thực tế (m) |
| safe_dist | REAL | Khoảng cách an toàn (m) |
| speed | INTEGER | Tốc độ xe sau (km/h) |

---

## 📁 Cấu trúc thư mục

```
traffic-violation-detection/
├── main.py                     # Entry point, PyQt6 UI
├── speed_estimation.py         # Tính tốc độ & khoảng cách
├── lane_detection.py           # Phát hiện làn đường
├── car_color_detection.py      # Nhận dạng màu xe
├── database.py                 # Tương tác PostgreSQL
├── app.py                      # Flask web tra cứu
├── templates/
│   └── index.html              # Giao diện web
├── roi_vehicle_crops_violations/   # Ảnh crop xe vi phạm
├── roi_vehicle_crops_frames/       # Full frame vi phạm
├── violations_excel/               # File Excel export
└── requirements.txt
```

---

## 🤝 Đóng góp

Pull requests và issues đều được chào đón. Vui lòng mở issue trước khi tạo PR lớn.

---

## 📄 License

MIT License — xem file [LICENSE](LICENSE) để biết thêm chi tiết.
