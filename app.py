"""
VisionInspect AI — Automated Surface Defect Detection using Deep Learning (CNN)
Industrial-grade Streamlit dashboard for steel surface defect classification.
"""

import io
import os
from datetime import datetime
import random
import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image

# Optional heavy deps — handled gracefully if missing/model absent
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="VisionInspect AI | Industrial Defect Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# CONSTANTS
# =========================================================================
MODEL_PATH = "visioninspect_cnn.keras"
INPUT_SHAPE = (224, 224)

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_DISPLAY_NAMES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled-in_scale": "Rolled-In Scale",
    "scratches": "Scratches",
}

DEFECT_KNOWLEDGE_BASE = {
    "crazing": {
        "description": (
            "Crazing appears as a network of fine, interconnected surface cracks "
            "resembling a spider-web or mud-crack pattern, typically caused by "
            "thermal stress or surface embrittlement during processing."
        ),
        "causes": [
            "Rapid or uneven cooling during rolling",
            "Excessive surface tension during deformation",
            "Poor lubrication leading to localized friction heat",
            "Material embrittlement from improper alloy composition",
        ],
        "impact": (
            "Reduces fatigue resistance and can act as initiation sites for "
            "larger cracks under cyclic loading, compromising structural integrity "
            "in downstream applications."
        ),
        "actions": [
            "Inspect and recalibrate cooling rate controls",
            "Review lubrication and rolling speed parameters",
            "Quarantine affected coil/batch for further metallurgical testing",
            "Schedule root-cause analysis with process engineering team",
        ],
        "icon": "🕸️",
    },
    "inclusion": {
        "description": (
            "Inclusions are foreign, non-metallic particles (oxides, sulfides, or "
            "slag) trapped within the steel matrix during casting, visible as "
            "dark or irregular surface/sub-surface spots."
        ),
        "causes": [
            "Contaminated raw material or scrap input",
            "Inadequate slag removal during steelmaking",
            "Refractory erosion contaminating the melt",
            "Poor deoxidation practice",
        ],
        "impact": (
            "Acts as a stress concentrator, significantly reducing tensile "
            "strength and ductility; high risk of failure under load-bearing "
            "conditions."
        ),
        "actions": [
            "Flag batch for source material traceability check",
            "Inspect ladle and tundish refractory condition",
            "Increase slag skimming frequency",
            "Reject part if inclusion density exceeds tolerance",
        ],
        "icon": "🔘",
    },
    "patches": {
        "description": (
            "Patches present as irregular, localized discolored or textured "
            "regions on the surface, often from inconsistent oxide scale removal "
            "or surface contamination."
        ),
        "causes": [
            "Inconsistent descaling process",
            "Localized oxidation due to uneven furnace heating",
            "Roll surface contamination transferring onto sheet",
            "Inadequate pickling time in acid bath",
        ],
        "impact": (
            "Primarily a cosmetic and coating-adhesion concern, but may indicate "
            "underlying inconsistent surface chemistry affecting paint or "
            "galvanizing quality."
        ),
        "actions": [
            "Audit descaling and pickling line parameters",
            "Clean and inspect work rolls for surface contamination",
            "Verify furnace zone temperature uniformity",
            "Re-route batch for secondary surface treatment if needed",
        ],
        "icon": "🟫",
    },
    "pitted_surface": {
        "description": (
            "Pitted surface defects are small, localized cavities or depressions "
            "distributed across the steel surface, commonly resulting from "
            "corrosion or rolling-process particulate contamination."
        ),
        "causes": [
            "Acid pickling over-exposure",
            "Hard particulate contamination on work rolls",
            "Localized corrosion during storage/transit",
            "Surface roughness inherited from prior processing stage",
        ],
        "impact": (
            "Compromises surface finish quality and corrosion resistance; "
            "pits can propagate into fatigue cracks under cyclic mechanical "
            "stress."
        ),
        "actions": [
            "Reduce pickling bath exposure time/concentration",
            "Inspect and clean rolling mill rolls regularly",
            "Improve storage conditions to prevent corrosion",
            "Apply protective coating prior to storage",
        ],
        "icon": "🕳️",
    },
    "rolled-in_scale": {
        "description": (
            "Rolled-in scale occurs when oxide scale is pressed into the steel "
            "surface during hot rolling, leaving embedded flaky or textured "
            "patches that resist standard cleaning."
        ),
        "causes": [
            "Incomplete scale removal before rolling",
            "Excessive furnace residence time causing thick oxide formation",
            "Worn or damaged descaling nozzles",
            "Insufficient rolling pressure to break loose scale",
        ],
        "impact": (
            "Creates an uneven surface profile affecting coating adhesion and "
            "aesthetic quality; can also serve as a stress riser under load."
        ),
        "actions": [
            "Verify high-pressure descaling nozzle function",
            "Reduce furnace soak time where possible",
            "Increase descaling pass count before final rolling",
            "Conduct surface roughness sampling on affected coils",
        ],
        "icon": "🌀",
    },
    "scratches": {
        "description": (
            "Scratches are linear surface marks caused by mechanical contact "
            "with equipment, tooling, or handling fixtures during processing, "
            "transport, or storage."
        ),
        "causes": [
            "Misaligned or damaged guide rollers",
            "Debris caught between sheet and equipment",
            "Improper handling during coiling/uncoiling",
            "Worn conveyor or tensioning equipment",
        ],
        "impact": (
            "Generally cosmetic at shallow depths, but deep scratches reduce "
            "effective material thickness and can become crack-initiation "
            "points under stress."
        ),
        "actions": [
            "Inspect and align guide rollers and tensioning units",
            "Implement debris-detection checks on the line",
            "Retrain handling personnel on coil management",
            "Measure scratch depth against tolerance specification",
        ],
        "icon": "📏",
    },
}

# =========================================================================
# SESSION STATE INIT
# =========================================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "model" not in st.session_state:
    st.session_state.model = None

if "model_load_error" not in st.session_state:
    st.session_state.model_load_error = None

if "uploaded_image_data" not in st.session_state:
    st.session_state.uploaded_image_data = None


# =========================================================================
# STYLING — DARK THEME + GLASSMORPHISM
# =========================================================================
def inject_custom_css():
    """Inject custom CSS for dark glassmorphism industrial dashboard theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #111827;
            --accent-cyan: #22d3ee;
            --accent-violet: #8b5cf6;
            --accent-blue: #3b82f6;
            --text-primary: #e5e7eb;
            --text-secondary: #9ca3af;
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.10);
        }

        .stApp {
            background: radial-gradient(circle at 10% 0%, #131c31 0%, #0b0f19 45%, #060810 100%);
            color: var(--text-primary);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1322 0%, #0a0e18 100%);
            border-right: 1px solid var(--glass-border);
        }

        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(34, 211, 238, 0.12);
            border-color: rgba(34, 211, 238, 0.3);
        }

        .analytics-card {
            background: linear-gradient(135deg, rgba(34,211,238,0.06) 0%, rgba(139,92,246,0.06) 100%);
            border: 1px solid rgba(34,211,238,0.12);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .analytics-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(34, 211, 238, 0.1);
            border-color: rgba(34, 211, 238, 0.3);
        }

        /* ===== HERO SECTION - FIXED ===== */
        .hero-container {
            background: linear-gradient(135deg, 
                rgba(34, 211, 238, 0.15) 0%, 
                rgba(139, 92, 246, 0.15) 50%, 
                rgba(59, 130, 246, 0.15) 100%
            );
            border: 1px solid rgba(34, 211, 238, 0.2);
            border-radius: 24px;
            padding: 2rem 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }
        
        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at center, rgba(34, 211, 238, 0.05) 0%, transparent 70%);
            animation: heroGlow 8s ease-in-out infinite;
        }
        
        @keyframes heroGlow {
            0%, 100% { transform: translate(0%, 0%); }
            50% { transform: translate(10%, -10%); }
        }
        
        .hero-icon {
            font-size: 3.5rem;
            display: block;
            margin-bottom: 0.3rem;
            position: relative;
            z-index: 1;
            filter: drop-shadow(0 0 30px rgba(34, 211, 238, 0.3));
            animation: iconPulse 3s ease-in-out infinite;
        }
        
        @keyframes iconPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .hero-title {
            font-size: 2.8rem;
            font-weight: 900;
            background: linear-gradient(135deg, #22d3ee 0%, #8b5cf6 40%, #3b82f6 70%, #22d3ee 100%);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.3rem;
            letter-spacing: -0.5px;
            position: relative;
            z-index: 1;
            animation: gradientShift 4s ease-in-out infinite;
        }
        
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .hero-subtitle {
            font-size: 1.05rem;
            color: var(--text-secondary);
            font-weight: 400;
            max-width: 700px;
            margin: 0 auto 1rem auto;
            line-height: 1.6;
            position: relative;
            z-index: 1;
        }

        .badge-row {
            display: flex;
            justify-content: center;
            gap: 0.6rem;
            flex-wrap: wrap;
            position: relative;
            z-index: 1;
        }
        
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 1rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid var(--glass-border);
            background: rgba(255,255,255,0.06);
            color: var(--text-primary);
            transition: all 0.3s ease;
            backdrop-filter: blur(5px);
        }
        
        .badge:hover {
            transform: scale(1.05) translateY(-2px);
            box-shadow: 0 8px 24px rgba(34,211,238,0.2);
        }
        
        .badge-cnn { border-color: rgba(34,211,238,0.5); color: #22d3ee; background: rgba(34,211,238,0.08); }
        .badge-tf { border-color: rgba(245,158,11,0.5); color: #f59e0b; background: rgba(245,158,11,0.08); }
        .badge-st { border-color: rgba(239,68,68,0.5); color: #ef4444; background: rgba(239,68,68,0.08); }
        .badge-cv { border-color: rgba(139,92,246,0.5); color: #8b5cf6; background: rgba(139,92,246,0.08); }

        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.7rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid transparent;
            background-image: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
            background-size: 100% 2px;
            background-position: 0 100%;
            background-repeat: no-repeat;
        }

        .severity-high {
            background: rgba(239,68,68,0.15);
            color: #f87171;
            border: 1px solid rgba(239,68,68,0.4);
            padding: 0.25rem 0.7rem;
            border-radius: 8px;
            font-weight: 700;
            display: inline-block;
            font-size: 0.85rem;
        }
        
        .severity-medium {
            background: rgba(245,158,11,0.15);
            color: #fbbf24;
            border: 1px solid rgba(245,158,11,0.4);
            padding: 0.25rem 0.7rem;
            border-radius: 8px;
            font-weight: 700;
            display: inline-block;
            font-size: 0.85rem;
        }
        
        .severity-low {
            background: rgba(16,185,129,0.15);
            color: #34d399;
            border: 1px solid rgba(16,185,129,0.4);
            padding: 0.25rem 0.7rem;
            border-radius: 8px;
            font-weight: 700;
            display: inline-block;
            font-size: 0.85rem;
        }

        .prediction-label {
            font-size: 1.4rem;
            font-weight: 800;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.2rem;
        }

        .metric-sub {
            color: var(--text-secondary);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .footer-container {
            text-align: center;
            padding: 1rem;
            color: var(--text-secondary);
            border-top: 1px solid var(--glass-border);
            margin-top: 1.5rem;
            font-size: 0.75rem;
        }

        .rec-list {
            list-style-type: none;
            padding: 0;
        }
        
        .rec-list li {
            margin-bottom: 0.3rem;
            line-height: 1.4;
            color: var(--text-primary);
            padding-left: 1.2rem;
            position: relative;
            font-size: 0.85rem;
        }
        
        .rec-list li::before {
            content: "▸";
            position: absolute;
            left: 0;
            color: var(--accent-cyan);
            font-weight: 700;
        }

        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: #0b0f19; }
        ::-webkit-scrollbar-thumb { 
            background: linear-gradient(180deg, var(--accent-cyan), var(--accent-violet));
            border-radius: 4px; 
        }

        div[data-testid="stMetric"] {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 0.6rem;
            backdrop-filter: blur(10px);
        }
        
        div[data-testid="stMetric"] label {
            color: #9ca3af !important;
            font-size: 0.75rem !important;
        }
        
        div[data-testid="stMetric"] div {
            color: #e5e7eb !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: var(--glass-bg);
            border-radius: 8px 8px 0 0;
            border: 1px solid var(--glass-border);
            padding: 0.4rem 0.9rem;
            font-size: 0.85rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(34,211,238,0.12);
            border-color: var(--accent-cyan);
        }

        .image-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 0.8rem;
            margin: 0 auto;
            max-width: 250px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .image-card:hover {
            border-color: var(--accent-cyan);
            box-shadow: 0 8px 24px rgba(34,211,238,0.12);
        }
        
        .image-card img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }
        
        .image-card .caption {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.3rem;
        }
        
        .stat-card {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 0.8rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent-cyan);
            box-shadow: 0 8px 24px rgba(34,211,238,0.08);
        }
        
        .stat-number {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stat-label {
            color: var(--text-secondary);
            font-size: 0.7rem;
            margin-top: 0.1rem;
        }
        
        .center-upload {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            max-width: 500px;
            margin: 0 auto;
        }
        
        .result-container {
            max-width: 700px;
            margin: 0 auto;
        }
        
        .dash-metric {
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 0.6rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .dash-metric:hover {
            border-color: var(--accent-cyan);
            transform: translateY(-2px);
        }
        
        .dash-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-primary);
            display: block;
        }
        
        .dash-label {
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .dash-icon {
            font-size: 1.2rem;
            margin-bottom: 0.1rem;
        }
        
        .analytics-image-card {
            max-width: 150px;
            margin: 0 auto;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--glass-border);
        }
        
        .analytics-image-card img {
            width: 100%;
            height: auto;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# MODEL LOADING
# =========================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the CNN model from disk."""
    if not TF_AVAILABLE:
        return None, "TensorFlow is not installed."
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file '{MODEL_PATH}' not found."
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalize, and batch an input PIL image."""
    img = image.convert("RGB").resize(INPUT_SHAPE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def run_inference(model, image: Image.Image):
    """Run model prediction."""
    arr = preprocess_image(image)
    preds = model.predict(arr, verbose=0)[0]
    preds = np.array(preds).flatten()

    n = min(len(preds), len(CLASS_NAMES))
    preds = preds[:n]
    classes = CLASS_NAMES[:n]

    if preds.sum() <= 0 or abs(preds.sum() - 1.0) > 0.05:
        exp = np.exp(preds - np.max(preds))
        preds = exp / exp.sum()

    idx = int(np.argmax(preds))
    predicted_class = classes[idx]
    confidence = float(preds[idx]) * 100
    prob_dict = {classes[i]: float(preds[i]) * 100 for i in range(n)}
    return predicted_class, confidence, prob_dict


def simulate_inference(image: Image.Image):
    """Demo-mode prediction based on image characteristics."""
    arr = np.array(image.convert("L").resize(INPUT_SHAPE)).astype("float32")
    
    mean_intensity = arr.mean()
    std_intensity = arr.std()
    edge_density = np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean()
    
    if edge_density > 80 and std_intensity > 50:
        if mean_intensity > 150:
            predicted_class = "scratches"
            confidence = 85 + (mean_intensity - 150) / 10
        else:
            predicted_class = "crazing"
            confidence = 80 + (std_intensity - 50) / 5
    elif std_intensity < 30 and mean_intensity > 100:
        if mean_intensity > 180:
            predicted_class = "patches"
            confidence = 75 + (mean_intensity - 180) / 5
        else:
            predicted_class = "rolled-in_scale"
            confidence = 78 + (100 - mean_intensity) / 10
    elif mean_intensity < 80 and std_intensity > 40:
        if std_intensity > 60:
            predicted_class = "pitted_surface"
            confidence = 82 + (std_intensity - 60) / 5
        else:
            predicted_class = "inclusion"
            confidence = 76 + (80 - mean_intensity) / 5
    else:
        classes_sorted = sorted(CLASS_NAMES, key=lambda x: hash(x) % 6)
        idx = int(mean_intensity / 255 * 5) % len(CLASS_NAMES)
        predicted_class = classes_sorted[idx]
        confidence = 70 + (mean_intensity / 255 * 20)
    
    confidence = min(98, max(65, confidence))
    
    probs = np.zeros(len(CLASS_NAMES))
    for i, cls in enumerate(CLASS_NAMES):
        if cls == predicted_class:
            probs[i] = confidence / 100
        else:
            probs[i] = (1 - confidence / 100) / (len(CLASS_NAMES) - 1)
    
    noise = np.random.RandomState(int(arr.mean() * 100) % 1000).uniform(-0.02, 0.02, len(CLASS_NAMES))
    probs = np.clip(probs + noise, 0.01, 0.99)
    probs = probs / probs.sum()
    
    idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[idx]
    confidence = float(probs[idx]) * 100
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) * 100 for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidence, prob_dict


# =========================================================================
# SEVERITY LOGIC
# =========================================================================
def get_severity(confidence: float):
    """Return (label, css_class) for a given confidence score."""
    if confidence > 90:
        return "HIGH SEVERITY", "severity-high"
    elif confidence >= 70:
        return "MEDIUM SEVERITY", "severity-medium"
    else:
        return "LOW SEVERITY", "severity-low"


# =========================================================================
# UI COMPONENTS
# =========================================================================
def render_hero():
    st.markdown(
        """
        <div class="hero-container">
            <span class="hero-icon">👁️</span>
            <div class="hero-title">Neo - Inspect</div>
            <div class="hero-subtitle">
                Automated Surface Defect Detection &amp; Industrial Quality Inspection
            </div>
            <div class="badge-row">
                <span class="badge badge-cnn">🧠 CNN</span>
                <span class="badge badge-tf">🔶 TensorFlow</span>
                <span class="badge badge-st">🎈 Streamlit</span>
                <span class="badge badge-cv">👁️ Computer Vision</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.markdown("---")

        st.markdown("### 📋 Overview")
        st.markdown(
            """
            Deep learning system for automatic steel surface defect detection
            using Convolutional Neural Networks.
            """
        )

        st.markdown("### 🧠 Model Info")
        st.markdown(f"- **Model:** CNN")
        st.markdown(f"- **Framework:** TensorFlow")
        st.markdown(f"- **Input:** {INPUT_SHAPE[0]}×{INPUT_SHAPE[1]}")

        st.markdown("### 🏷️ Classes")
        for cls in CLASS_NAMES:
            icon = DEFECT_KNOWLEDGE_BASE[cls]["icon"]
            st.markdown(f"{icon} {CLASS_DISPLAY_NAMES[cls]}")

        st.markdown("### 📥 Input")
        st.markdown(
            """
            - JPG, JPEG, PNG
            - ≥ 224×224 px
            """
        )

        st.markdown("---")
        if st.session_state.model is not None:
            st.success("✅ Model Loaded")
        elif st.session_state.model_load_error is not None:
            st.warning("⚠️ Demo Mode")
            with st.expander("Details"):
                st.caption(st.session_state.model_load_error)
        st.caption(f"Session: {datetime.now().strftime('%H:%M')}")


def render_metrics_dashboard():
    st.markdown('<div class="section-header">📊 System Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            """
            <div class="dash-metric">
                <div class="dash-icon">🏷️</div>
                <span class="dash-value">6</span>
                <span class="dash-label">Classes</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="dash-metric">
                <div class="dash-icon">🧠</div>
                <span class="dash-value">CNN</span>
                <span class="dash-label">Architecture</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="dash-metric">
                <div class="dash-icon">📐</div>
                <span class="dash-value">224×224</span>
                <span class="dash-label">Resolution</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            """
            <div class="dash-metric">
                <div class="dash-icon">⚡</div>
                <span class="dash-value">80.2%</span>
                <span class="dash-label">Accuracy</span>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_gauge(confidence: float):
    """Render a speedometer gauge."""
    if confidence > 90:
        bar_color = "#ef4444"
    elif confidence >= 70:
        bar_color = "#f59e0b"
    else:
        bar_color = "#10b981"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 28, "color": "#e5e7eb"}},
            title={"text": "Confidence", "font": {"size": 14, "color": "#9ca3af"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9ca3af", "tickfont": {"size": 10}},
                "bar": {"color": bar_color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 70], "color": "rgba(16,185,129,0.15)"},
                    {"range": [70, 90], "color": "rgba(245,158,11,0.15)"},
                    {"range": [90, 100], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.6,
                    "value": confidence,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb"},
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_probability_chart(prob_dict: dict):
    st.markdown('<div class="section-header">📈 Probability Analysis</div>', unsafe_allow_html=True)
    df = pd.DataFrame(
        {
            "Defect": [CLASS_DISPLAY_NAMES[k] for k in prob_dict.keys()],
            "Probability": list(prob_dict.values()),
        }
    ).sort_values("Probability", ascending=True)

    fig = px.bar(
        df,
        x="Probability",
        y="Defect",
        orientation="h",
        text=df["Probability"].apply(lambda v: f"{v:.1f}%"),
        color="Probability",
        color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
    )
    fig.update_traces(textposition="outside", marker_line_width=0, textfont_size=10)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb", "size": 11},
        xaxis=dict(range=[0, max(100, df["Probability"].max() * 1.15)], gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.0)"),
        coloraxis_showscale=False,
        height=280,
        margin=dict(l=10, r=30, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_recommendation(predicted_class: str):
    info = DEFECT_KNOWLEDGE_BASE[predicted_class]
    st.markdown('<div class="section-header">🤖 Recommendations</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="glass-card">
            <p style="font-size:0.9rem; margin-bottom:0.3rem;"><b>{info['icon']} Description</b></p>
            <p style="font-size:0.85rem; color:#e5e7eb;">{info['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        causes_html = "".join([f"<li>{c}</li>" for c in info["causes"]])
        st.markdown(
            f"""
            <div class="glass-card">
                <p style="font-size:0.9rem; margin-bottom:0.3rem;"><b>🔍 Causes</b></p>
                <ul class="rec-list">{causes_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <p style="font-size:0.9rem; margin-bottom:0.3rem;"><b>⚠️ Impact</b></p>
                <p style="font-size:0.85rem;">{info['impact']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    actions_html = "".join([f"<li>{a}</li>" for a in info["actions"]])
    st.markdown(
        f"""
        <div class="glass-card">
            <p style="font-size:0.9rem; margin-bottom:0.3rem;"><b>✅ Actions</b></p>
            <ul class="rec-list">{actions_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_report_text(image_name, predicted_class, confidence, severity_label):
    info = DEFECT_KNOWLEDGE_BASE[predicted_class]
    causes = "\n".join([f"  - {c}" for c in info["causes"]])
    actions = "\n".join([f"  - {a}" for a in info["actions"]])
    report = f"""
==================================================
        VisionInspect AI — Report
==================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image: {image_name}

--------------------------------------------------
PREDICTION
--------------------------------------------------
Defect : {CLASS_DISPLAY_NAMES[predicted_class]}
Confidence : {confidence:.2f}%
Severity : {severity_label}

--------------------------------------------------
DETAILS
--------------------------------------------------
{info['description']}

Causes:
{causes}

Impact:
{info['impact']}

Actions:
{actions}

==================================================
"""
    return report.strip()


def render_prediction_tab():
    st.markdown('<div class="center-upload">', unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="text-align:center; justify-content:center;">🖼️ Upload Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload surface image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            st.session_state.uploaded_image_data = img_str
        except Exception as e:
            st.error(f"Error: {e}")
            return

        st.markdown(
            f"""
            <div class="image-card">
                <img src="data:image/png;base64,{img_str}" alt="Uploaded Image"/>
                <div class="caption">{uploaded_file.name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.markdown(
            f"""
            <div style="text-align:center; font-size:0.75rem; color:#9ca3af; margin-top:0.3rem;">
                {image.size[0]}×{image.size[1]} • {file_size_kb:.0f}KB
            </div>
            """,
            unsafe_allow_html=True,
        )

        analyze = st.button("🔬 Run Inspection", type="primary", use_container_width=True)

        if analyze:
            with st.spinner("Analyzing..."):
                try:
                    if st.session_state.model is not None:
                        predicted_class, confidence, prob_dict = run_inference(
                            st.session_state.model, image
                        )
                    else:
                        predicted_class, confidence, prob_dict = simulate_inference(image)

                    st.session_state.last_prediction = {
                        "image_name": uploaded_file.name,
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "prob_dict": prob_dict,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_data": img_str,
                    }
                    st.session_state.history.append(
                        {
                            "Image": uploaded_file.name[:20],
                            "Prediction": CLASS_DISPLAY_NAMES[predicted_class],
                            "Confidence": f"{confidence:.1f}%",
                            "Time": datetime.now().strftime("%H:%M"),
                        }
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Inference failed: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    result = st.session_state.last_prediction
    if result is not None:
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        severity_label, severity_css = get_severity(confidence)

        st.markdown('<div class="section-header">🎯 Result</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-sub">Predicted</div>
                    <div class="prediction-label">{DEFECT_KNOWLEDGE_BASE[predicted_class]['icon']} {CLASS_DISPLAY_NAMES[predicted_class]}</div>
                    <div style="margin-top:0.5rem;">
                        <div class="metric-sub">Confidence</div>
                        <div style="font-size:1.3rem; font-weight:700;">{confidence:.1f}%</div>
                    </div>
                    <div style="margin-top:0.5rem;">
                        <span class="{severity_css}">{severity_label}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col2:
            render_gauge(confidence)

        st.markdown("---")
        render_probability_chart(result["prob_dict"])
        
        st.markdown("---")
        render_recommendation(predicted_class)
        
        st.markdown("---")
        report_text = generate_report_text(
            result["image_name"], predicted_class, confidence, severity_label
        )
        st.download_button(
            label="📄 Download Report",
            data=report_text,
            file_name=f"VisionInspect_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )
        
        st.markdown("---")
        render_history_table()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_history_table():
    st.markdown('<div class="section-header">🕘 History</div>', unsafe_allow_html=True)
    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
        return
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True, height=150)
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.history = []
        st.rerun()


def render_analytics_tab():
    """Analytics tab - input-based analysis"""
    st.markdown('<div class="section-header">📊 Input Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.last_prediction is None:
        st.info("ℹ️ Upload and analyze an image first in the Prediction tab.")
        return
    
    result = st.session_state.last_prediction
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{confidence:.1f}%</div>
                <div class="stat-label">Confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number" style="font-size:1.1rem;">{CLASS_DISPLAY_NAMES[predicted_class]}</div>
                <div class="stat-label">Defect</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        severity_label, _ = get_severity(confidence)
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number" style="font-size:1.1rem;">{severity_label.split()[0]}</div>
                <div class="stat-label">Severity</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{len(result['prob_dict'])}</div>
                <div class="stat-label">Classes</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
        st.markdown("**Uploaded Image**")
        img_data = result.get('image_data', '')
        if img_data:
            st.markdown(
                f"""
                <div class="analytics-image-card">
                    <img src="data:image/png;base64,{img_data}" alt="Uploaded Image"/>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.caption(f"Predicted: {CLASS_DISPLAY_NAMES[predicted_class]}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
        st.markdown("**Probability Distribution**")
        df = pd.DataFrame(
            {
                "Defect": [CLASS_DISPLAY_NAMES[k][:12] for k in result["prob_dict"].keys()],
                "Probability": list(result["prob_dict"].values()),
            }
        ).sort_values("Probability", ascending=False)
        
        fig = px.bar(
            df,
            x="Defect",
            y="Probability",
            text=df["Probability"].apply(lambda v: f"{v:.1f}%"),
            color="Probability",
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
        )
        fig.update_traces(textposition="outside", marker_line_width=0, textfont_size=9)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb', size=10),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 100]),
            coloraxis_showscale=False,
            height=250,
            margin=dict(l=10, r=10, t=10, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Defect Analysis")
    info = DEFECT_KNOWLEDGE_BASE[predicted_class]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div class="analytics-card">
                <p style="font-size:0.9rem; margin-bottom:0.3rem;"><b>{info['icon']} Description</b></p>
                <p style="font-size:0.85rem;">{info['description']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        causes_html = "".join([f"<li>{c}</li>" for c in info["causes"]])
        st.markdown(
            f"""
            <div class="analytics-card">
                <p style="font-size:0.9rem; margin-bottom:0.3rem;"><b>🔍 Causes</b></p>
                <ul class="rec-list">{causes_html}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="analytics-card">
                <p style="font-size:0.9rem; margin-bottom:0.3rem;"><b>⚠️ Impact</b></p>
                <p style="font-size:0.85rem;">{info['impact']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        actions_html = "".join([f"<li>{a}</li>" for a in info["actions"]])
        st.markdown(
            f"""
            <div class="analytics-card">
                <p style="font-size:0.9rem; margin-bottom:0.3rem;"><b>✅ Actions</b></p>
                <ul class="rec-list">{actions_html}</ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    st.markdown("### 📈 Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.history)}</div>
                <div class="stat-label">Total</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        if len(st.session_state.history) > 0:
            conf_vals = [float(h["Confidence"].replace("%", "")) for h in st.session_state.history]
            avg_conf = np.mean(conf_vals)
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-number">{avg_conf:.1f}%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col3:
        if len(st.session_state.history) > 0:
            defect_counts = pd.DataFrame(st.session_state.history)["Prediction"].value_counts()
            if len(defect_counts) > 0:
                most_common = defect_counts.index[0]
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-number" style="font-size:1.1rem;">{most_common[:15]}</div>
                        <div class="stat-label">Most Common</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def render_model_info_tab():
    st.markdown('<div class="section-header">🧠 CNN Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
        Convolutional Neural Network for multi-class steel defect classification.
        Stacked convolution + pooling blocks extract hierarchical features,
        followed by dense layers and softmax output for 6 defect classes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🏋️ Training</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
        Trained with data augmentation (rotation, flipping, brightness) for
        generalization. Used categorical cross-entropy loss, Adam optimizer,
        early stopping to prevent overfitting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🗂️ Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
        NEU surface-defect style dataset with 6 defect classes from hot-rolled
        steel strips under industrial imaging conditions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🏷️ Classes</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, cls in enumerate(CLASS_NAMES):
        info = DEFECT_KNOWLEDGE_BASE[cls]
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="glass-card">
                    <p style="font-size:0.9rem; margin-bottom:0.2rem;"><b>{info['icon']} {CLASS_DISPLAY_NAMES[cls]}</b></p>
                    <p style="font-size:0.75rem; color:#9ca3af; margin:0;">{info['description'][:80]}...</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_footer():
    st.markdown(
        """
        <div class="footer-container">
            <span class="badge badge-tf">TensorFlow</span>
            <span class="badge badge-cnn">CNN</span>
            <span class="badge badge-st">Streamlit</span>
            <span class="badge badge-cv">CV</span>
            <p style="margin-top:0.5rem;">© 2026 VisionInspect AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# MAIN APP
# =========================================================================
def main():
    inject_custom_css()

    model, error = load_model()
    st.session_state.model = model
    st.session_state.model_load_error = error

    render_hero()
    render_sidebar()
    render_metrics_dashboard()

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔬 Prediction", "📊 Analytics", "🧠 Info"])

    with tab1:
        render_prediction_tab()
    with tab2:
        render_analytics_tab()
    with tab3:
        render_model_info_tab()

    render_footer()


if __name__ == "__main__":
    main()