# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# ========== KONFIGURASI ==========
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "EfficientNetB0_pepaya.keras") # Pastikan file model ada di direktori yang sama
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

st.set_page_config(
    page_title="Klasifikasi Pepaya", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== CSS KUSTOM ==========
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');
    
    /* Sembunyikan elemen bawaan Streamlit */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Reset dan Style Dasar */
    .stApp {
        background: linear-gradient(135deg, #f9f9f9, #d2fbd2);
        font-family: "Segoe UI", Arial, sans-serif;
    }
    
    /* Container Utama */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 450px;
    }
    
    /* Container Kustom */
    .pepaya-container {
        background: #fff;
        padding: 30px 25px;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        text-align: center;
        animation: fadeIn 0.8s ease;
        margin: 0 auto;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Style Judul */
    .pepaya-title {
        font-size: 1.8rem;
        margin-bottom: 20px;
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
    }
    
    /* Style Tombol */
    .stButton > button {
        width: 100% !important;
        margin: 8px 0 !important;
        padding: 12px 20px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 30px !important;
        cursor: pointer !important;
        background: linear-gradient(135deg, #28a745, #4cd964) !important;
        color: white !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4) !important;
    }
    
    .stButton > button:focus {
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4) !important;
    }
    
    /* Pratinjau Gambar */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
        margin: 15px 0;
    }
    
    .stImage > img {
        border-radius: 12px;
    }
    
    /* Style Hasil Prediksi */
    .result-success {
        margin-top: 20px;
        font-size: 1.1rem;
        font-weight: bold;
        color: #2c3e50;
        padding: 15px;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-radius: 12px;
        border: 1px solid #c3e6cb;
    }
    
    .result-error {
        margin-top: 20px;
        font-size: 1.1rem;
        font-weight: bold;
        color: #721c24;
        padding: 15px;
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-radius: 12px;
        border: 1px solid #f5c6cb;
    }
    
    .result-info {
        margin-top: 20px;
        font-size: 1rem;
        color: #0c5460;
        padding: 15px;
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border-radius: 12px;
        border: 1px solid #bee5eb;
    }
    
    /* Sembunyikan label bawaan file uploader dan kamera */
    .stFileUploader label, .stCameraInput label {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== MEMUAT MODEL ==========
@st.cache_resource
def load_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model(MODEL_PATH)

if model is None:
    st.stop()

# Daftar kelas label
class_labels = ["matang", "mentah", "setengah"]

# ========== FUNGSI UTILITAS ==========
def predict_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, 0).astype(np.float32)
        
        predictions = model.predict(img_batch, verbose=0)
        
        score = tf.nn.softmax(predictions[0])
        label_index = np.argmax(score)
        label = class_labels[label_index]
        confidence = float(np.max(score))
        
        return label, confidence, None
    except Exception as e:
        return None, None, str(e)

# ========== TAMPILAN UTAMA (UI) ==========
st.markdown('<div class="pepaya-container">', unsafe_allow_html=True)
st.markdown('<h1 class="pepaya-title">🍈 Klasifikasi Kematangan Pepaya</h1>', unsafe_allow_html=True)

# Inisialisasi session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'current_image_bytes' not in st.session_state:
    st.session_state.current_image_bytes = None

# Input Kamera & File Uploader
camera_file = st.camera_input("📷 Ambil Foto")
uploaded_file = st.file_uploader("🖼️ Pilih dari Galeri", type=list(ALLOWED_EXTENSIONS))

# Tentukan file mana yang akan digunakan
active_file = camera_file if camera_file is not None else uploaded_file

# Proses dan tampilkan gambar jika ada file yang aktif
if active_file is not None:
    try:
        # Simpan byte gambar ke session state
        st.session_state.current_image_bytes = active_file.getvalue()
        # Tampilkan gambar
        # INI BAGIAN YANG DIPERBAIKI: use_container_width=True
        st.image(st.session_state.current_image_bytes, caption="Gambar Pratinjau", use_container_width=True)
    except Exception as e:
        st.markdown(f'<div class="result-error">❌ Error saat membaca file: {e}</div>', unsafe_allow_html=True)
        st.session_state.current_image_bytes = None

# Tombol Prediksi
if st.button("🔍 Prediksi Kematangan"):
    if st.session_state.current_image_bytes is not None:
        with st.spinner("⏳ Menganalisis gambar..."):
            label, confidence, error = predict_image(st.session_state.current_image_bytes)
            
            if error:
                st.session_state.prediction_result = f'<div class="result-error">❌ Error prediksi: {error}</div>'
            else:
                confidence_percent = confidence * 100
                result_text = f"✅ Hasil: **{label.upper()}** ({confidence_percent:.2f}%)"
                st.session_state.prediction_result = f'<div class="result-success">{result_text}</div>'
    else:
        st.session_state.prediction_result = '<div class="result-info">⚠️ Silakan pilih atau ambil foto terlebih dahulu!</div>'

# Tampilkan hasil prediksi
if st.session_state.prediction_result:
    st.markdown(st.session_state.prediction_result, unsafe_allow_html=True)
elif st.session_state.current_image_bytes is None:
    st.markdown('<div class="result-info">📱 Ambil foto atau unggah gambar pepaya untuk memulai.</div>', unsafe_allow_html=True)

# Tutup container
st.markdown('</div>', unsafe_allow_html=True)