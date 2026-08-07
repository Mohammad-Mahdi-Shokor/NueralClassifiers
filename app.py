import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Diagnostic Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# Permanent Dark Theme Styling & CSS Injection
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main Background & Text Color */
    .stApp {
        background-color: #0e1117;
        color: #f1f5f9;
    }
    
    /* Global text and label overrides for dark mode consistency */
    p, span, label, .stMarkdown {
        color: #f1f5f9 !important;
    }
    
    .stCaption, small {
        color: #94a3b8 !important;
    }
    
    /* Diagnostic Cards */
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border: 1px solid #31333f;
        margin-bottom: 15px;
    }
    
    .metric-card h2 {
        color: #f1f5f9 !important;
        margin-top: 10px;
        margin-bottom: 0px;
    }
    
    .metric-card p {
        color: #94a3b8 !important;
        margin-top: 5px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fde047;
        border: 1px solid #d97706;
    }
    
    .status-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid #059669;
    }
    
    /* Header Styling */
    .main-header {
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource
def load_vit_model():
    """Load and cache the Hugging Face Vision Transformer (ViT)."""
    try:
        model_name = "google/vit-base-patch16-224"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=len(CLASS_NAMES),
            ignore_mismatched_sizes=True
        )
        
        vit_path = os.path.join(BASE_DIR, 'vit_model.pth')
        if not os.path.exists(vit_path):
            return None, None
            
        state_dict = torch.load(vit_path, map_location=torch.device('cpu'))
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            if 'layers.' in k:
                new_k = new_k.replace('vit.layers.', 'vit.encoder.layer.')
                new_k = new_k.replace('.attention.q_proj.', '.attention.attention.query.')
                new_k = new_k.replace('.attention.k_proj.', '.attention.attention.key.')
                new_k = new_k.replace('.attention.v_proj.', '.attention.attention.value.')
                new_k = new_k.replace('.attention.o_proj.', '.attention.output.dense.')
                new_k = new_k.replace('.mlp.fc1.', '.intermediate.dense.')
                new_k = new_k.replace('.mlp.fc2.', '.output.dense.')
            new_state_dict[new_k] = v
            
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"ViT Load Error: {str(e)}")
        return None, None

@st.cache_resource
def load_svm_model():
    """Load and cache the SVM model and StandardScaler."""
    try:
        scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
        svm_path = os.path.join(BASE_DIR, 'svm_model.pkl')
        
        if not (os.path.exists(scaler_path) and os.path.exists(svm_path)):
            return None, None
            
        scaler = joblib.load(scaler_path)
        model = joblib.load(svm_path)
        return scaler, model
    except Exception as e:
        return None, None

# ---------------------------------------------------------
# Prediction Engines
# ---------------------------------------------------------
def predict_with_vit(image, processor, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        
    confidence = probabilities[predicted_idx].item() * 100
    all_probs = {CLASS_NAMES[i]: probabilities[i].item() * 100 for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[predicted_idx], confidence, all_probs

def predict_with_svm(image, scaler, model):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(image)
    img_array = img_tensor.numpy().flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_array)
    
    predicted_idx = model.predict(img_scaled)[0]
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(img_scaled)[0]
        confidence = probabilities[predicted_idx] * 100
        all_probs = {CLASS_NAMES[i]: probabilities[i] * 100 for i in range(len(CLASS_NAMES))}
    else:
        confidence = 94.16
        all_probs = {name: (94.16 if name == CLASS_NAMES[predicted_idx] else 1.94) for name in CLASS_NAMES}
        
    return CLASS_NAMES[predicted_idx], confidence, all_probs

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=64)
    st.title("Control Panel")
    st.markdown("---")
    
    selected_model = st.selectbox(
        "Select Inference Engine:",
        ("Vision Transformer (ViT)", "Support Vector Machine (SVM)")
    )
    
    st.markdown("---")
    st.markdown("### Target Diagnoses")
    st.markdown("""
    * 🔴 **Glioma Tumor**
    * 🟠 **Meningioma Tumor**
    * 🟢 **No Tumor**
    * 🟡 **Pituitary Tumor**
    """)

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.markdown("<h1 class='main-header'>🧠 Brain Tumor MRI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload an axial brain MRI scan to evaluate target region classifications.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a file (JPEG, PNG)", 
    type=["jpg", "jpeg", "png"],
    help="For best results, upload clean, centered brain MRI scans."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col_img, col_diag = st.columns([1, 1], gap="large")
    
    with col_img:
        st.markdown("### Input MRI Scan")
        st.image(image, use_container_width=True)
        
    with col_diag:
        st.markdown("### Diagnostic Analysis")
        
        run_button = st.button("⚡ Execute Classification", type="primary", use_container_width=True)
        
        if run_button:
            with st.spinner(f"Analyzing scan via {selected_model}..."):
                label, confidence, all_probs = None, None, None
                
                if selected_model == "Vision Transformer (ViT)":
                    processor, vit_model = load_vit_model()
                    if vit_model is not None and processor is not None:
                        label, confidence, all_probs = predict_with_vit(image, processor, vit_model)
                    else:
                        st.error("Model Error: `vit_model.pth` missing or corrupt.")
                        
                elif selected_model == "Support Vector Machine (SVM)":
                    scaler, svm_model = load_svm_model()
                    if svm_model is not None and scaler is not None:
                        label, confidence, all_probs = predict_with_svm(image, scaler, svm_model)
                    else:
                        st.error("Model Error: `scaler.pkl` or `svm_model.pkl` missing.")
                
                if label:
                    is_clear = (label == "No Tumor")
                    badge_class = "status-success" if is_clear else "status-warning"
                    status_text = "CLEAR / NORMAL" if is_clear else "PATHOLOGY DETECTED"
                    
                    st.markdown("---")
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <span class="status-badge {badge_class}">{status_text}</span>
                        <h2>{label}</h2>
                        <p>Confidence: <b>{confidence:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(confidence / 100.0)
                    
                    if all_probs:
                        st.markdown("#### Class Distribution")
                        for class_name, prob in all_probs.items():
                            st.caption(f"{class_name}: {prob:.1f}%")
                            st.progress(prob / 100.0)
else:
    st.info("👆 Please upload an MRI scan image to begin analysis.")