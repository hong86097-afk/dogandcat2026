import streamlit as st
import torch
import numpy as np
import cv2 as cv
from PIL import Image
import random
import os

# Fix randomness
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@st.cache_resource
def load_weights():
    files = ["W1.npy", "b1.npy", "W2.npy", "b2.npy", "W3.npy", "b3.npy"]
    # Friendly error if any file is missing
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        return None, missing
    tensors = [torch.tensor(np.load(f), dtype=torch.float32) for f in files]
    return tensors, []


# Model
def my_ANN(X, W1, b1, W2, b2, W3, b3, training=False):
    Z1 = torch.relu(X @ W1 + b1)
    if training:
        Z1 = torch.nn.functional.dropout(Z1, p=0.5)
    Z2 = torch.relu(Z1 @ W2 + b2)
    if training:
        Z2 = torch.nn.functional.dropout(Z2, p=0.5)
    Z3 = Z2 @ W3 + b3
    return Z3   # raw logits


st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #0f3460 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: #12122a !important;
    border-right: 1px solid #2a2a4a !important;
}
[data-testid="stSidebar"] * { color: #ccccee !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #8888bb !important; font-size: 0.85rem !important; }

h1, h2, h3, p, label, span, div { color: #ffffff; }

[data-testid="stFileUploader"] {
    background: rgba(26,26,60,0.8) !important;
    border-radius: 16px !important;
    border: 2px dashed #4a4a8a !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"] * { color: #aaaacc !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: #7777aa !important; }

[data-testid="stImage"] img {
    border-radius: 20px !important;
    border: 2px solid #3a3a6a !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5) !important;
}

[data-testid="stMetric"] {
    background: rgba(26,26,60,0.8);
    border-radius: 14px;
    padding: 1rem 1.2rem !important;
    border: 1px solid #3a3a6a;
}
[data-testid="stMetricLabel"] { color: #8888bb !important; font-size: 0.8rem !important; letter-spacing: 1px; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; font-weight: 800 !important; }

[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #4FC3F7, #a78bfa) !important;
    border-radius: 10px !important;
}
[data-testid="stProgress"] {
    background: #2a2a4a !important;
    border-radius: 10px !important;
    height: 12px !important;
}

[data-testid="stAlert"] {
    background: rgba(26,26,60,0.9) !important;
    border-radius: 14px !important;
    border: 1px solid #3a3a6a !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 1rem 1.5rem !important;
}

.stMarkdown small, caption { color: #8888bb !important; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align:center; padding: 2rem 0 0.5rem 0;">
    <div style="font-size:3.5rem; font-weight:900;
                background:linear-gradient(90deg,#4FC3F7,#a78bfa,#f472b6);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Cat vs Dog Classifier
    </div>
    <div style="color:#6666aa; font-size:1rem; margin-top:0.4rem; letter-spacing:1px;">
        Upload an image and let the model decide!
    </div>
</div>
<div style="height:2px; background:linear-gradient(90deg,transparent,#4FC3F7,#a78bfa,transparent);
            margin: 1rem 2rem 2rem 2rem; border-radius:2px;"></div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div style="text-align:center; padding:1rem 0;">
    <div style="font-size:2rem;">🐾</div>
    <div style="font-size:1.1rem; font-weight:700; color:#ffffff;">Settings</div>
</div>
<hr style="border-color:#2a2a4a;">
""", unsafe_allow_html=True)
st.sidebar.markdown("**About This App**")
st.sidebar.markdown("""This is a simple image classifier that distinguishes between cats and dogs using a 3-layer ANN built with PyTorch. Upload an image of a cat or dog, and the model will predict which one it is along with confidence scores.""")
st.sidebar.markdown("<hr style='border-color:#2a2a4a;'>", unsafe_allow_html=True)
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("- Input: 128×128 grayscale")
st.sidebar.markdown("- Architecture: 3-layer ANN")
st.sidebar.markdown("- Framework: PyTorch")
st.sidebar.markdown("<hr style='border-color:#2a2a4a;'>", unsafe_allow_html=True)
st.sidebar.markdown("**Classes**")
st.sidebar.markdown("🐱 Class 0 — Cat")
st.sidebar.markdown("🐶 Class 1 — Dog")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# Load weights with friendly error
weights, missing = load_weights()
if weights is None:
    st.error(f"⚠️ Missing model files: {', '.join(missing)}. Please run the notebook training cells to generate them, then push to GitHub.")
    st.stop()
W1, b1, W2, b2, W3, b3 = weights

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("<div style='font-size:0.8rem; color:#6666aa; letter-spacing:1px; margin-bottom:0.5rem;'>INPUT IMAGE</div>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img_gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
    img_resized = cv.resize(img_gray, (128, 128)) / 255.0
    tensor = torch.tensor(img_resized.flatten(), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = my_ANN(tensor, W1, b1, W2, b2, W3, b3, training=False)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)

    cat_conf = confidence[0][0].item() * 100
    dog_conf = confidence[0][1].item() * 100
    top_conf = max(cat_conf, dog_conf)
    diff_conf = abs(cat_conf - dog_conf)

    threshold = 70
    margin = 15

    if top_conf < threshold or diff_conf < margin:
        label = "Other ❓"
        conf = top_conf
        color = "#a78bfa"
    elif cat_conf >= dog_conf:
        label = "Cat 🐱"
        conf = cat_conf
        color = "#4FC3F7"
    else:
        label = "Dog 🐶"
        conf = dog_conf
        color = "#EF9A9A"

    if label == "Dog 🐶" and conf >= threshold:
        st.balloons()
    elif label == "Cat 🐱" and conf >= threshold:
        st.snow()

    with col_res:
        st.markdown("<div style='font-size:0.8rem; color:#6666aa; letter-spacing:1px; margin-bottom:0.5rem;'>RESULTS</div>", unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Prediction", label)
        m2.metric("Confidence", f"{conf:.1f}%")

        st.markdown("<hr style='border-color:#2a2a4a; margin:1rem 0;'>", unsafe_allow_html=True)

        st.markdown("**Confidence Breakdown**")
        st.markdown(f"<div style='color:#8888bb; font-size:0.85rem; margin-bottom:2px;'>Cat — {cat_conf:.1f}%</div>", unsafe_allow_html=True)
        st.progress(int(cat_conf))
        st.markdown(f"<div style='color:#8888bb; font-size:0.85rem; margin-bottom:2px; margin-top:0.5rem;'>Dog — {dog_conf:.1f}%</div>", unsafe_allow_html=True)
        st.progress(int(dog_conf))

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if label == "Other ❓":
            st.success("⚠️ The model is not confident enough. Classified as Other.")
        elif label.startswith("Dog"):
            st.success("✅ The model predicts: Dog 🐶")
        else:
            st.success("✅ The model predicts: Cat 🐱")
else:
    st.markdown("""
    <div style="background:rgba(26,26,60,0.6); border:2px dashed #3a3a6a;
                border-radius:20px; padding:5rem 2rem; text-align:center; margin-top:1rem;">
        <div style="font-size:4rem;">🐾</div>
        <div style="color:#5555aa; font-size:1.1rem; margin-top:1rem; font-weight:600;">
            Upload a JPG or PNG to get started
        </div>
        <div style="color:#3a3a6a; font-size:0.85rem; margin-top:0.5rem;">
            The model will classify your image as Cat or Dog
        </div>
    </div>
    """, unsafe_allow_html=True)
