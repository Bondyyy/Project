# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import time
import pandas as pd

# Import các hàm từ utils.py
from utils import get_model, get_transforms, device

# --- CÁC THAM SỐ CẤU HÌNH ---
DATA_DIR = "data/data_casting_Khoa"
SAVE_DIR = "saved_models"
EPOCHS = 15
BATCH_SIZE = 32
N_SPLITS = 5
LEARNING_RATE = 1e-4

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

def main():
    print(f"Sử dụng thiết bị: {device}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Thư mục lưu model: {SAVE_DIR}")

    train_transform, test_val_transform = get_transforms()

    train_dir = os.path.join(DATA_DIR, "train")
    
    full_train_dataset = datasets.ImageFolder(train_dir, transform=None)
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss()
    
    print("\n=== Bắt đầu Cross-Validation ===")
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_train_dataset)))):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        
        train_subset = Subset(full_train_dataset, train_idx)
        val_subset = Subset(full_train_dataset, val_idx)
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = test_val_transform

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = get_model()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        best_val_loss = float('inf')
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = evaluate(model, val_loader, criterion)
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{SAVE_DIR}/best_model_fold{fold+1}.pth")

            scheduler.step(val_loss)

    print("\n=== Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu Train ===")
    full_train_dataset.transform = train_transform
    full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    final_model = get_model()
    optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(final_model, full_train_loader, optimizer, criterion)
        print(f"Final Model Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    final_model_path = f"{SAVE_DIR}/final_model.pth"
    torch.save(final_model.state_dict(), final_model_path)
    print(f"Đã lưu mô hình cuối cùng tại: {final_model_path}")
    print("\nQUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT!")

if __name__ == "__main__":
    main()