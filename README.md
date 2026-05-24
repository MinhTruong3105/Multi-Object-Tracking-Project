# Multi-Object-Tracking-Project: Pedestrian Tracking Pipeline: YOLOv11m + BoT-SORT

Dự án này triển khai một hệ thống theo dõi đa đối tượng (Multi-Object Tracking - MOT) theo kiến trúc Tracking-by-Detection, sử dụng mô hình **YOLOv11m** cho bài toán phát hiện đối tượng và thuật toán **BoT-SORT** cho bài toán liên kết dữ liệu. Hệ thống được đánh giá bằng công cụ **TrackEval** chuẩn của MOT Challenge.

## 🎥 Demo Kết quả theo dõi (Tracking Output)

[![Tracking Demo](https://img.youtube.com/vi/GXN3nJD7hdg/maxresdefault.jpg)](https://youtu.be/GXN3nJD7hdg)

*(Click vào ảnh để xem video demo trên YouTube)*

## 🚀 Tính năng nổi bật
* Sử dụng phiên bản YOLOv11m mới nhất từ Ultralytics, kết hợp cùng BoT-SORT.
* Tối ưu hóa siêu tham số cho môi trường đám đông (Confidence: 0.15, NMS IoU: 0.60, Image Size: 1280).
* Tùy chỉnh `track_buffer=60` trong `custom_botsort.yaml` để giảm thiểu ID Switch khi đối tượng bị che khuất (Dynamic Occlusion).
* Tích hợp sẵn module đánh giá chuẩn học thuật từ thư viện **TrackEval** (tính toán tự động HOTA, CLEAR MOT, Identity metrics).
* Tự động xuất biểu đồ báo cáo trực quan.

## 📁 Cấu trúc thư mục

```text
.
├── main.py                     # Script thực thi toàn bộ pipeline (Tracking + Evaluation)
├── custom_botsort.yaml         # Cấu hình tùy chỉnh cho thuật toán BoT-SORT
├── README.md                   # Tài liệu hướng dẫn
└── assets/                     # Thư mục chứa ảnh biểu đồ cho báo cáo
    ├── chart_error_counts.png
    └── chart_percentage_metrics.png
```
