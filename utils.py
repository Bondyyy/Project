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
    original_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = original_img.shape[:2]

    # 1. Gray + blur
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    # 2. Threshold (THRESH_BINARY) với Otsu
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Morphological: đóng để kết nối vùng, mở để loại nhiễu nhỏ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Tìm contours trên mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((h,w), dtype=np.uint8)
    if len(contours) == 0:
        # fallback: dùng Canny nếu Otsu không tìm được gì
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vẽ các contour lên mask (fill)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Optionally: lọc theo diện tích nhỏ -> loại bỏ các noise nhỏ
    min_area = 100  # điều chỉnh tuỳ kích thước ảnh
    large_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(large_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    mask = large_mask

    # 5. Tạo overlay đỏ bán trong suốt trên ảnh gốc
    red_overlay = np.zeros_like(original_img)
    red_overlay[:, :] = (0, 0, 255)  # BGR red
    mask_3ch = cv2.merge([mask, mask, mask]) // 255  # 0 or 1

    # alpha blending: chỉ overlay nơi mask==1
    alpha = 0.5
    overlayed = original_img.copy()
    overlayed = np.where(mask_3ch==1, (original_img * (1-alpha) + red_overlay * alpha).astype(np.uint8), original_img)

    # Nếu muốn vẽ viền đỏ đậm nữa
    cv2.drawContours(overlayed, contours, -1, (0,0,255), thickness=2)

    # Trả về PIL RGB
    return Image.fromarray(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
