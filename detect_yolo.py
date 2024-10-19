from ultralytics import YOLO
import cv2

# Đường dẫn tới mô hình đã fine-tune
MODEL_PATH = "/Users/user/PycharmProjects/trainyolo/yolo11n.pt"  # Cập nhật đường dẫn nếu cần

# Load mô hình đã lưu
model = YOLO(MODEL_PATH)

# Đường dẫn tới ảnh cần dự đoán
IMAGE_PATH = "/data/test/images/R0011089_MP4-0039_jpg.rf.4be31ecd4d1d65b140ca61845764641a.jpg"  # Cập nhật đường dẫn ảnh

# Thực hiện phát hiện đối tượng
results = model(IMAGE_PATH)

# Hiển thị kết quả dự đoán
results[0].show()

# (Tùy chọn) Lưu ảnh với kết quả phát hiện
# results.save("output/")
print(f"Kết quả đã được lưu tại: output/")