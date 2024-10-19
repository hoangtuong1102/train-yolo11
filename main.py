from ultralytics import YOLO

# Đường dẫn tới mô hình pre-trained
PRETRAINED_MODEL = "yolo11n.pt"  # Bạn có thể thay bằng yolo11s.pt, yolo11m.pt nếu cần

# Đường dẫn tới file cấu hình dataset của bạn
DATASET_CONFIG = "/Users/user/PycharmProjects/trainyolo/data/data.yaml"  # File YAML mô tả dataset

# Thiết lập tham số huấn luyện
EPOCHS = 2  # Số epoch huấn luyện
IMG_SIZE = 640  # Kích thước ảnh đầu vào
BATCH_SIZE = 16  # Kích thước batch
DEVICE = 0  # Chạy trên GPU (hoặc 'cpu' nếu không có GPU)

def train_yolo():
    print("Đang tải mô hình pre-trained...")
    model = YOLO(PRETRAINED_MODEL)  # Tải mô hình

    print("Bắt đầu huấn luyện...")
    model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        # device=DEVICE
    )

    print("Đánh giá mô hình...")
    metrics = model.val()  # Đánh giá trên tập validation
    print(f"Kết quả đánh giá: {metrics}")

    # Thử dự đoán trên ảnh mẫu
    sample_image = "/Users/user/PycharmProjects/trainyolo/data/R0010716.JPG"  # Thay bằng đường dẫn ảnh của bạn
    results = model.predict(sample_image)
    results[0].show()  # Hiển thị kết quả dự đoán

    # Xuất mô hình sang định dạng ONNX
    print("Xuất mô hình sang định dạng ONNX...")
    onnx_path = model.export(format="onnx")
    print(f"Mô hình đã được xuất tại: {onnx_path}")

if __name__ == "__main__":
    train_yolo()