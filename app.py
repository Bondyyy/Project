import streamlit as st
from PIL import Image
import torch
import os
from utils import get_model, predict_image, analyze_and_draw_defects, device

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Phát hiện lỗi vật đúc",
    page_icon="🤖",
    layout="wide"
)

# --- TẢI MODEL  ---
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
        image = Image.open(uploaded_file).convert("RGB")

        st.sidebar.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)
        st.sidebar.markdown("---")

        if st.sidebar.button("Bắt đầu phân tích"):
            with st.spinner("🧠 Mô hình đang phân tích, vui lòng chờ..."):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🖼️ Ảnh gốc")
                    st.image(image, use_container_width=True)

                class_name, confidence = predict_image(model, image)
                
                with col2:
                    st.subheader("🎯 Kết quả phân loại")
                    
                    if class_name == 'ok_front':
                        st.success(f"✅ Kết quả: OK (Không có lỗi)")
                    else:
                        st.error(f"❌ Kết quả: CÓ LỖI (Defect)")
                    
                    st.metric(label="Độ tin cậy", value=f"{confidence * 100:.2f}%")
                    st.write("---")
                    
                    if class_name == 'def_front':
                        st.subheader("🗺️ Phân tích chi tiết vùng lỗi")
                        segmented_image, defect_types, defect_count = analyze_and_draw_defects(image)
                        st.image(segmented_image, caption="Các vùng lỗi đã được đánh dấu (không có chữ).", use_container_width=True)
                        st.markdown("---")
                        st.subheader("📝 Chi tiết các loại lỗi")

                        if defect_count > 0:
                            st.write(f"**Tổng số vùng lỗi phát hiện:** {defect_count}")
                            st.write("**Phân loại chi tiết:**")
                            for i, dtype in enumerate(defect_types, 1):
                                st.markdown(f"- Vùng lỗi {i}: **{dtype}**")
                        else:
                            st.info("Mô hình phát hiện có khả năng bị lỗi, nhưng không tìm thấy vùng lỗi rõ ràng bằng phân tích hình ảnh.")
    else:
        st.info("Vui lòng tải ảnh lên từ thanh công cụ bên trái để bắt đầu.")
