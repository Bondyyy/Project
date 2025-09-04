# utils.py

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN
from PIL import Image

# Định nghĩa class để map index sang tên class
CLASS_NAMES = {0: 'def_front', 1: 'ok_front'}

# Thiết bị sử dụng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes=2):
    """
    Tải và cấu hình mô hình EfficientNetV2-S.
    """
    model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model.to(device)

def get_transforms():
    """
    Trả về các phép biến đổi ảnh cho training và testing.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_val_transform

def predict_image(model, image_pil):
    """
    Dự đoán lớp của một ảnh PIL.
    """
    _, test_transform = get_transforms()
    image_tensor = test_transform(image_pil).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)

    class_idx = predicted_class_idx.item()
    class_name = CLASS_NAMES.get(class_idx, "Unknown")
    
    return class_name, confidence.item()

def segment_defect(image_pil):
    """
    Phân đoạn vùng lỗi trên ảnh bằng phương pháp phân ngưỡng và tìm đường viền.
    Đây là cách tiếp cận hiệu quả hơn cho việc tìm các vết nứt, xước.
    """
    # Chuyển đổi PIL Image sang OpenCV format (BGR) và giữ nguyên kích thước gốc
    original_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # 1. Chuyển sang ảnh xám để tập trung vào cường độ sáng
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Làm mờ ảnh để giảm nhiễu và làm nổi bật các vùng lớn
    blurred_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    
    # 3. Áp dụng phân ngưỡng Otsu để tạo ảnh nhị phân (đen-trắng)
    # THRESH_BINARY_INV: Vùng tối (lỗi) sẽ trở thành màu trắng, nền sáng thành màu đen.
    # Otsu tự động tìm giá trị ngưỡng tối ưu.
    _, thresh = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Tìm các đường viền (contours) của các vùng trắng (vùng nghi ngờ là lỗi)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Vẽ các đường viền tìm được lên ảnh gốc
    # Tạo một bản sao của ảnh gốc để vẽ lên
    segmented_img = original_img.copy() 
    # Vẽ các đường viền màu đỏ với độ dày là 2
    cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 2) # (0, 0, 255) là màu đỏ trong BGR

    # Chuyển đổi lại sang PIL Image (RGB) để Streamlit hiển thị
    return Image.fromarray(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))