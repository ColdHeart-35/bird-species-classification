"""
BirdVision — AI-Powered Bird Species Identification Web Application
Built with Streamlit & TensorFlow (EfficientNetB0 Transfer Learning)

Usage:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# Import existing inference pipeline helpers
from src.predict_cnn import MODEL_CONFIGS, preprocess_image_bytes, run_inference
from src.data_loader import load_class_labels, LABEL_MAP_PATH

# ──────────────────────────────────────────────
# Page Configuration & Styling
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="BirdVision — AI Bird Species Classifier",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern, clean CSS styling
CUSTOM_CSS = """
<style>
    /* Global Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    /* Main Header Card */
    .header-card {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }
    .header-card h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #00f2fe;
    }
    .header-card p {
        font-size: 1.15rem;
        color: #e0e0e0;
        margin-bottom: 0;
    }
    /* Prediction Result Banner */
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.8rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(56, 239, 125, 0.3);
    }
    .prediction-card .species-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .prediction-card .confidence-badge {
        font-size: 1.3rem;
        font-weight: 600;
        background: rgba(0, 0, 0, 0.25);
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    /* Top predictions container */
    .top-pred-box {
        background: #1e293b;
        padding: 1.2rem;
        border-radius: 12px;
        margin-top: 0.8rem;
        border-left: 5px solid #38ef7d;
    }
    /* Metric Cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38ef7d;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #94a3b8;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Cached Model & Label Loader (Loaded ONCE)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNetB0 Model...")
def get_model_and_labels():
    """Load and cache the trained Keras model and class labels map."""
    cfg = MODEL_CONFIGS["efficientnet"]
    model_path = cfg["path"]
    
    if not model_path.exists():
        return None, None, f"Model file not found at '{model_path}'. Please verify the file exists."
    if not LABEL_MAP_PATH.exists():
        return None, None, f"Class labels file not found at '{LABEL_MAP_PATH}'. Please verify the file exists."

    try:
        model = keras.models.load_model(str(model_path))
        labels = load_class_labels()
        return model, labels, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


# ──────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────
def main():
    # Header Banner
    st.markdown(
        """
        <div class="header-card">
            <h1>🦜 BirdVision</h1>
            <p>AI-Powered Bird Species Identification System — 25 Common Indian Bird Species</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/000000/bird.png", width=70)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Feature:",
        ["🔍 Species Identification", "📊 Model Performance Dashboard", "ℹ️ About the Model"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Quick Info")
    st.sidebar.info(
        "**Backbone**: EfficientNetB0\n\n"
        "**Strategy**: Two-Stage Transfer Learning & Fine-Tuning\n\n"
        "**Validation Acc**: 96.50%\n\n"
        "**Test Acc**: 96.20%"
    )

    # ── Page 1: Species Identification ──────────
    if page == "🔍 Species Identification":
        st.subheader("Upload an Image of a Bird")
        st.write("Supported formats: **JPG, JPEG, PNG**. Drag & drop or browse your local files.")

        model, class_labels, err_msg = get_model_and_labels()

        if err_msg:
            st.error(err_msg)
            return

        uploaded_file = st.file_uploader(
            "Choose a bird photo...",
            type=["jpg", "jpeg", "png"],
            help="Select a clear photo of an Indian bird species.",
        )

        col1, col2 = st.columns([1, 1], gap="large")

        if uploaded_file is not None:
            with col1:
                st.markdown("### Uploaded Image")
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True, caption=f"File: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Invalid image file: {e}")
                    return

            with col2:
                st.markdown("### Prediction Results")
                predict_btn = st.button("🚀 Identify Bird Species", type="primary", use_container_width=True)

                if predict_btn or "last_prediction" in st.session_state:
                    # Read image raw bytes
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()

                    with st.spinner("Running EfficientNetB0 inference..."):
                        try:
                            result = run_inference(
                                image_input=image_bytes,
                                model=model,
                                class_labels=class_labels,
                                img_size=MODEL_CONFIGS["efficientnet"]["img_size"],
                                top_k=5,
                            )
                            st.session_state["last_prediction"] = result
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                            return

                    res = st.session_state["last_prediction"]
                    species_name = res["predicted_species"]
                    confidence_pct = res["confidence"] * 100

                    # Result Card
                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div style="font-size:0.95rem; text-transform:uppercase; letter-spacing:1px;">Top Match</div>
                            <div class="species-title">{species_name}</div>
                            <div class="confidence-badge">Confidence: {confidence_pct:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Top-5 Predictions Probability Bars
                    st.markdown("#### Top Probabilities")
                    for item in res["top_predictions"]:
                        sp = item["species"]
                        prob = item["confidence"]
                        st.write(f"**{item['rank']}. {sp}** ({prob:.2%})")
                        st.progress(float(prob))

        else:
            with col1:
                st.info("👆 Please upload a bird image above to begin identification.")

    # ── Page 2: Model Performance Dashboard ────
    elif page == "📊 Model Performance Dashboard":
        st.subheader("Model Performance & Evaluation Metrics")
        st.write("Comprehensive metrics computed on the untouched **1,500-image test set**.")

        # Metric summary cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown('<div class="metric-card"><div class="metric-value">96.50%</div><div class="metric-label">Validation Accuracy</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-card"><div class="metric-value">96.20%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card"><div class="metric-value">96.34%</div><div class="metric-label">Macro Precision</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown('<div class="metric-card"><div class="metric-value">96.21%</div><div class="metric-label">Macro F1 Score</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Tabs for visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Confusion Matrix",
            "📈 Per-Class Accuracy",
            "📊 Model Comparison",
            "📉 Training Curves",
        ])

        results_dir = Path("results")
        cm_path = results_dir / "confusion_matrix" / "efficientnetb0_fine_tuned_confusion_matrix.png"
        per_class_path = results_dir / "plots" / "efficientnetb0_fine_tuned_per_class_accuracy.png"
        comp_path = results_dir / "plots" / "model_comparison.png"
        curves_path = results_dir / "plots" / "efficientnet_accuracy.png"

        with tab1:
            st.markdown("### 25×25 Confusion Matrix Heatmap")
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                st.warning("Confusion matrix plot not found in `results/confusion_matrix/`. Run `python -m src.evaluate --compare` to generate.")

        with tab2:
            st.markdown("### Accuracy breakdown across all 25 species")
            if per_class_path.exists():
                st.image(str(per_class_path), use_container_width=True)
            else:
                st.warning("Per-class accuracy chart not found in `results/plots/`.")

        with tab3:
            st.markdown("### Comparison: EfficientNetB0 vs Baseline CNN vs HOG+SVM")
            if comp_path.exists():
                st.image(str(comp_path), use_container_width=True)
            else:
                st.warning("Model comparison plot not found in `results/plots/`.")

        with tab4:
            st.markdown("### EfficientNetB0 Training & Validation Curves")
            if curves_path.exists():
                st.image(str(curves_path), use_container_width=True)
            else:
                st.warning("Training curves plot not found in `results/plots/`.")

    # ── Page 3: About the Model ───────────────
    elif page == "ℹ️ About the Model":
        st.subheader("About the Machine Learning Architecture")
        st.markdown(
            """
            ### 🏗️ Architecture Overview
            This project uses **Transfer Learning** with **EfficientNetB0**, pretrained on the ImageNet dataset (1.28M images, 1,000 classes), followed by a two-stage training strategy:

            1. **Stage 1 — Head-Only Training**:
               - Backbone layers are completely frozen.
               - Only the new custom classification head learns (`lr = 1e-3`).
               - Prevents disturbing the high-quality ImageNet pretrained visual filters.

            2. **Stage 2 — Fine-Tuning**:
               - The top 30 layers of the EfficientNetB0 backbone are unfrozen.
               - Trained with a very small learning rate (`lr = 1e-5`) to fine-tune bird-specific features (plumage patterns, beak shapes).
               - `BatchNormalization` layers remain locked to preserve ImageNet statistics.

            ### 🛡️ Overfitting Prevention
            - **Data Augmentation**: Random horizontal flips, rotation, zoom, brightness & contrast jitter.
            - **Regularization**: Dropout layers (`rate=0.3`) in the classification head.
            - **Early Stopping & ReduceLROnPlateau**: Automatically stops training when validation accuracy plateaus.

            ### 📊 Why CNN out-performed Classical ML (HOG + SVM)
            - **HOG + SVM**: Converts images to grayscale, losing crucial bird plumage color details → ~17.8% Accuracy.
            - **EfficientNetB0**: Learns hierarchical 3D spatial & color representations directly from raw pixels → **96.20% Test Accuracy**.
            """
        )


if __name__ == "__main__":
    main()
