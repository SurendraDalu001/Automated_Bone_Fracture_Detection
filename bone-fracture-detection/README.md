# 🦴 Bone Fracture Detection System

An AI-powered web application to detect bone fractures from X-ray images using a transfer-learning **MobileNetV2** deep learning model.

---

## 📁 Project Structure

```
bone-fracture-detection/
├── app.py               # Main Streamlit web application
├── BFD5.h5              # Pre-trained MobileNetV2 model (GitHub-ready, <100MB)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/AutomatedBone.git
cd AutomatedBone/bone-fracture-detection
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the web application
```bash
streamlit run app.py
```

The app will open automatically in your browser at **http://localhost:8501**.

---

## 🚀 Usage

1. **Upload an X-ray image** (supported formats: JPG, JPEG, or PNG).
2. Click **🔍 Predict**.
3. View the AI diagnosis:
   - **FRACTURED 🚨** (in red) with confidence percentage.
   - **NORMAL ✅** (in green) with confidence percentage.
   - Detailed fracture risk probability bar.

---

## 🧠 Model & Dataset Details

| Property | Details |
|---|---|
| **Base Architecture** | MobileNetV2 (Pre-trained on ImageNet) |
| **Input Shape** | 224 × 224 × 3 |
| **Preprocessing** | Rescaled (1/255.0), RGB |
| **Classification Head** | GlobalAveragePooling2D + Dense(128, ReLU) + Dropout(0.5) + Dense(1, Sigmoid) |
| **Overall Test Accuracy** | **95.85%** |
| **Fracture Detection Sensitivity** | **99.16%** |
| **Model Size** | ~43.8 MB (Optimized for GitHub file limits) |

---

## 👨‍🏫 Project Supervision

- **Supervisor:** Dr. J Krishnaiah

---

## ⚠️ Medical Disclaimer

This tool is created for **research and educational purposes only**. It does not constitute medical advice or certified medical diagnosis. Always consult certified radiologists and healthcare professionals for clinical decision-making.
