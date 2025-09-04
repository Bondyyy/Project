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
    Phân đoạn vùng lỗi trên ảnh bằng DBSCAN và Fuzzy C-Means.
    """
    # Chuyển đổi PIL Image sang OpenCV format (BGR)
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (128, 128))

    pixels = img_resized.reshape((-1, 3))
    pixels_norm = pixels / 255.0

    # DBSCAN để tìm outlier (lỗi)
    dbscan = DBSCAN(eps=0.05, min_samples=5)
    db_labels = dbscan.fit_predict(pixels_norm)
    outlier_mask = (db_labels == -1)

    # Fuzzy C-Means cho các pixel không lỗi
    pixels_in = pixels_norm[~outlier_mask].T
    if pixels_in.shape[1] < 2:  # Không đủ điểm để phân cụm
        # Nếu gần như toàn bộ là outlier, coi toàn bộ là lỗi
        segmented_img = np.full(img_resized.shape, (255, 0, 0), dtype=np.uint8)
        return Image.fromarray(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))

    n_clusters = 2
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        pixels_in, n_clusters, 2, error=0.005, maxiter=1000, init=None
    )
    
    labels_soft = np.zeros(len(pixels))
    labels_soft[~outlier_mask] = np.argmax(u, axis=0)
    labels_soft[outlier_mask] = n_clusters # Outlier là cụm riêng

    # Tạo ảnh segmentation
    segmented_img = np.zeros_like(pixels)
    unique_labels = np.unique(labels_soft)
    
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3))
    colors[n_clusters] = [255, 0, 0] # Màu đỏ cho vùng lỗi chính

    for i, lbl in enumerate(unique_labels):
        segmented_img[labels_soft == lbl] = colors[i]

    segmented_img = segmented_img.reshape(img_resized.shape)
    
    # Chuyển đổi lại sang PIL Image (RGB) để hiển thị
    return Image.fromarray(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))