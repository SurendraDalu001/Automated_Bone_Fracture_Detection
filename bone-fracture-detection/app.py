import os
import io
import streamlit as st
import numpy as np
from PIL import Image, ImageFile

# Handle truncated or slightly damaged X-ray images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "BFD5.h5"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0d3b6e 0%, #1a6bb5 100%);
        color: white;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.3); }

    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #0d3b6e 0%, #1a6bb5 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(13,59,110,0.25);
    }
    .header-card h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
    .header-card p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

    /* Upload zone */
    .upload-card {
        background: white;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1.2rem;
    }

    /* Result cards */
    .result-fractured {
        background: #fff5f5;
        border: 2px solid #fc8181;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(252,129,129,0.2);
    }
    .result-normal {
        background: #f0fff4;
        border: 2px solid #68d391;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(104,211,145,0.2);
    }
    .result-label-fractured { color: #c53030; font-size: 2rem; font-weight: 800; }
    .result-label-normal    { color: #276749; font-size: 2rem; font-weight: 800; }
    .confidence-text { font-size: 1.1rem; color: #555; margin-top: 0.5rem; }

    /* Metric pill */
    .metric-pill {
        display: inline-block;
        background: #ebf4ff;
        color: #1a6bb5;
        border-radius: 999px;
        padding: 0.3rem 1rem;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }

    /* Info box */
    .info-box {
        background: #ebf8ff;
        border-left: 4px solid #1a6bb5;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        color: #2c5282;
        font-size: 0.92rem;
    }

    /* Predict button */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #0d3b6e, #1a6bb5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    target_path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_NAME
    if not os.path.exists(target_path):
        return None, f"Model file `{MODEL_NAME}` not found. Expected at `{MODEL_PATH}`."
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(target_path, compile=False)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # shape (1, 224, 224, 3)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦴 BFD System")
    st.markdown("---")
    st.markdown("### 📋 About")
    st.markdown(
        "This application uses a **MobileNetV2** deep-learning model to detect "
        "bone fractures from X-ray images."
    )
    st.markdown("---")
    st.markdown("### 👨‍🏫 Supervisor")
    st.markdown("**Dr. C. Ranjeeth Kumar**")
    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    st.markdown(
        "1. Select a sample X-ray or upload your own (JPG / PNG)\n"
        "2. Click **Predict**\n"
        "3. View the diagnosis result"
    )
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.caption(
        "This tool is for **research purposes only** and is not a substitute "
        "for professional medical diagnosis."
    )


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>🦴 Bone Fracture Detection</h1>
    <p>AI-powered X-ray analysis using MobileNetV2 deep learning</p>
</div>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
model, model_error = load_model()

if model_error:
    st.error(f"⚠️ {model_error}")


# ── Main layout ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("#### 📤 Input X-Ray Image")

    input_mode = st.radio(
        "Select input source:",
        ["Choose from Samples", "Upload Image"],
        horizontal=True,
    )

    pil_image = None
    image_name = ""
    image_valid = False

    if input_mode == "Choose from Samples":
        sample_options = {
            "🚨 Fractured Sample 1": "fractured_sample_1.png",
            "🚨 Fractured Sample 2": "fractured_sample_2.png",
            "✅ Normal Sample 1": "normal_sample_1.jpg",
            "✅ Normal Sample 2": "normal_sample_2.jpg",
        }
        selected_label = st.selectbox("Pick a sample X-ray:", list(sample_options.keys()))
        sample_filename = sample_options[selected_label]
        sample_path = os.path.join(SAMPLE_DIR, sample_filename)

        if os.path.exists(sample_path):
            try:
                pil_image = Image.open(sample_path)
                image_name = sample_filename
                image_valid = True
                st.image(pil_image, caption=f"📁 Sample: {selected_label}", use_container_width=True)
                st.markdown(
                    f'<div class="metric-pill">📐 {pil_image.width} × {pil_image.height} px &nbsp;|&nbsp; Sample Image</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"❌ Failed to load sample image: {e}")
        else:
            st.warning("Sample images directory is empty or file not found.")

    else:
        uploaded_file = st.file_uploader(
            "Supported formats: JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            try:
                image_bytes = uploaded_file.read()
                pil_image = Image.open(io.BytesIO(image_bytes))
                image_name = uploaded_file.name
                st.image(pil_image, caption=f"📁 {uploaded_file.name}", use_container_width=True)
                st.markdown(
                    f'<div class="metric-pill">📐 {pil_image.width} × {pil_image.height} px &nbsp;|&nbsp; {uploaded_file.type}</div>',
                    unsafe_allow_html=True,
                )
                image_valid = True
            except Exception:
                st.error("❌ Could not read the file. Please upload a valid JPG or PNG image.")
                image_valid = False
        else:
            st.markdown(
                '<div class="info-box">👆 Upload an X-ray image to get started.</div>',
                unsafe_allow_html=True,
            )
            image_valid = False

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button ─────────────────────────────────────────────────────────
    predict_clicked = st.button("🔍 Predict", disabled=(not image_valid or model is None))


with col_right:
    st.markdown("#### 🩺 Diagnosis Result")

    if predict_clicked and image_valid and model is not None and pil_image is not None:
        with st.spinner("Analyzing X-ray..."):
            try:
                input_arr = preprocess_image(pil_image)
                raw_output = model.predict(input_arr, verbose=0)

                # The model was trained with ImageDataGenerator:
                # Class 0: 'fractured'
                # Class 1: 'not fractured'
                # Sigmoid output represents P(class=1) = P(not fractured)
                if raw_output.shape[-1] == 1:
                    prob_normal = float(raw_output[0][0])
                    prob_fractured = 1.0 - prob_normal
                else:
                    prob_fractured = float(raw_output[0][0])
                    prob_normal = float(raw_output[0][1])

                is_fractured = prob_fractured >= 0.5
                confidence = prob_fractured if is_fractured else prob_normal

                if is_fractured:
                    st.markdown(f"""
                    <div class="result-fractured">
                        <div style="font-size:3rem;">🚨</div>
                        <div class="result-label-fractured">FRACTURED</div>
                        <div class="confidence-text">
                            Confidence: <strong>{confidence*100:.1f}%</strong>
                        </div>
                        <div style="margin-top:1rem; color:#c53030; font-size:0.9rem;">
                            ⚠️ Bone fracture detected. Please consult an orthopedic specialist.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-normal">
                        <div style="font-size:3rem;">✅</div>
                        <div class="result-label-normal">NORMAL</div>
                        <div class="confidence-text">
                            Confidence: <strong>{confidence*100:.1f}%</strong>
                        </div>
                        <div style="margin-top:1rem; color:#276749; font-size:0.9rem;">
                            No fracture detected in the X-ray.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Probability metrics ─────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Fracture Probability**")
                st.progress(float(np.clip(prob_fractured, 0.0, 1.0)))
                st.caption(
                    f"Fracture Risk: **{prob_fractured * 100:.2f}%** &nbsp;|&nbsp; "
                    f"Normal Bone Confidence: **{prob_normal * 100:.2f}%**"
                )

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

    elif predict_clicked and model is None:
        st.warning("⚠️ Model not loaded. Cannot run prediction.")

    else:
        st.markdown("""
        <div style="background:white; border-radius:14px; padding:3rem 2rem;
                    text-align:center; box-shadow:0 2px 12px rgba(0,0,0,0.07);
                    color:#a0aec0;">
            <div style="font-size:3.5rem;">🩻</div>
            <div style="font-size:1.1rem; margin-top:1rem;">
                Select a sample or upload an image and click <strong>Predict</strong><br>to see the diagnosis here.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#a0aec0; font-size:0.85rem; padding:1rem 0;">
    Bone Fracture Detection System &nbsp;|&nbsp; MobileNetV2 Architecture &nbsp;|&nbsp;
    Supervised by <strong>Dr. C. Ranjeeth Kumar</strong>
</div>
""", unsafe_allow_html=True)
