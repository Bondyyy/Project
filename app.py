# app.py

import streamlit as st
from PIL import Image
import torch
import os

# Import các hàm từ utils.py
from utils import get_model, predict_image, segment_defect, device

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Phát hiện lỗi vật đúc",
    page_icon="🤖",
    layout="wide"
)

# --- TẢI MODEL (CACHE ĐỂ TĂNG TỐC) ---
@st.cache_resource
def load_trained_model(model_path):
    """
    Tải model đã được huấn luyện.
    Hàm này được cache để không phải load lại model mỗi khi có tương tác.
    """
    model = get_model()
    # Load state_dict với map_location để đảm bảo chạy được trên cả CPU và GPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# --- GIAO DIỆN CHÍNH ---
st.title("Ứng dụng phát hiện lỗi trên vật đúc kim loại 🔩")
st.write(
    "Chào mừng bạn đến với ứng dụng demo! "
    "Hãy tải lên một ảnh của vật đúc để kiểm tra xem nó có bị lỗi ('def_front') hay không ('ok_front'). "
    "Nếu có lỗi, hệ thống sẽ thử khoanh vùng vị trí của lỗi đó."
)
st.markdown("---")

# Đường dẫn đến model đã huấn luyện
MODEL_PATH = "saved_models/final_model.pth"

# Kiểm tra xem file model có tồn tại không
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Không tìm thấy file model tại '{MODEL_PATH}'. "
        "Vui lòng chạy file `train.py` để huấn luyện và tạo ra model trước."
    )
else:
    # Tải model
    model = load_trained_model(MODEL_PATH)
    
    # --- Sidebar để upload ảnh ---
    st.sidebar.header("Tải ảnh lên")
    uploaded_file = st.sidebar.file_uploader(
        "Chọn một ảnh vật đúc...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Đọc ảnh từ file upload
        image = Image.open(uploaded_file).convert("RGB")

        st.sidebar.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)
        st.sidebar.markdown("---")

        if st.sidebar.button("Bắt đầu phân tích"):
            with st.spinner("🧠 Mô hình đang phân tích, vui lòng chờ..."):
                # --- Cột hiển thị kết quả ---
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🖼️ Ảnh gốc")
                    st.image(image, use_container_width=True)

                # Dự đoán bằng model
                class_name, confidence = predict_image(model, image)
                
                with col2:
                    st.subheader("🎯 Kết quả phân loại")
                    
                    if class_name == 'ok_front':
                        st.success(f"✅ Kết quả: OK (Không có lỗi)")
                    else:
                        st.error(f"❌ Kết quả: CÓ LỖI (Defect)")
                    
                    st.metric(label="Độ tin cậy", value=f"{confidence * 100:.2f}%")
                    st.write("---")
                    
                    # Nếu phát hiện lỗi, tiến hành phân đoạn
                    if class_name == 'def_front':
                        st.subheader("🗺️ Phân đoạn vùng lỗi")
                        segmented_image = segment_defect(image)
                        st.image(segmented_image, caption="Vùng lỗi được đánh dấu màu đỏ", use_container_width=True)
                        
    else:
        st.info("Vui lòng tải ảnh lên từ thanh công cụ bên trái để bắt đầu.")