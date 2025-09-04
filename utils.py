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

# Bỏ hàm segment_defect cũ và thay bằng hàm này trong utils.py

def analyze_and_draw_defects(image_pil):
    
    original_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image_pil, [], 0

    defect_types = []
    filtered_contours = []
 
    for contour in contours:
        
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        filtered_contours.append(contour)
        perimeter = cv2.arcLength(contour, True)
        
        
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        if circularity > 0.7 and area < 2000:
            defect_type = "Lỗ khí"
        elif aspect_ratio > 3 and circularity < 0.3:
            defect_type = "Xước"
        elif aspect_ratio > 2 and circularity < 0.5:
            defect_type = "Nứt"
        elif area > 5000 and circularity < 0.6:
            defect_type = "Mẻ"
        else:
            defect_type = "Lỗi khác"
            
        defect_types.append(defect_type)

    img_with_labels = original_img.copy()
    for i, contour in enumerate(filtered_contours):
        
        cv2.drawContours(img_with_labels, [contour], -1, (0, 0, 255), 2)

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            label = defect_types[i]
            # Removed/commented out the line that adds text to the image
            # cv2.putText(img_with_labels, label, (cx - 20, cy - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    final_image_pil = Image.fromarray(cv2.cvtColor(img_with_labels, cv2.COLOR_BGR2RGB))
    
    return final_image_pil, defect_types, len(defect_types)
