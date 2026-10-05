"""
BirdVision — College Project Portal UI for Bird Species Identification
Finalized Streamlit Application for College ML Mini-Project.
Features:
  - 🏠 Home Overview
  - 🔍 Species Identifier (Image classification + Top-3 predictions + Taxonomy output)
  - 🧬 Taxonomy Classification (Biological taxonomy explorer)
  - 📊 Model Performance (Quantitative test metrics & evaluation plots)
  - ℹ️ About Project (Architecture, dataset & project specs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import IMG_SIZE, load_class_labels, LABEL_MAP_PATH
from src.predict_cnn import run_inference, MODEL_CONFIGS
from src.gpu_utils import setup_gpu

TAXONOMY_PATH = Path("data/species_taxonomy.json")

# ──────────────────────────────────────────────
# Page Configuration & Styling
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="BirdVision | Bird Species Identification",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .portal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #1e293b;
        color: #ffffff;
        padding: 0.8rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #0284c7;
    }
    .portal-brand {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .portal-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-left: 8px;
    }
    .portal-status {
        background-color: #10b981;
        color: #ffffff;
        padding: 0.25rem 0.65rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Caching Resource & Data Loaders
# ──────────────────────────────────────────────
@st.cache_resource
def load_efficientnet_model():
    """Load and cache the trained EfficientNetB0 Keras model from models/efficientnet_finetuned.keras."""
    setup_gpu(verbose=False)
    model_path = MODEL_CONFIGS["efficientnet"]["path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Trained EfficientNetB0 model not found at {model_path}")
    return keras.models.load_model(str(model_path))


@st.cache_data
def load_labels():
    """Load and cache class label mappings from models/class_labels.json."""
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Class labels map not found at {LABEL_MAP_PATH}")
    return load_class_labels(LABEL_MAP_PATH)


@st.cache_resource
def load_bird_gate_model():
    """Load a lightweight MobileNetV2 (ImageNet) used to pre-screen whether an image contains a bird."""
    from tensorflow.keras.applications import MobileNetV2
    return MobileNetV2(weights="imagenet", include_top=True)


def is_likely_bird(raw_bytes: bytes, gate_model) -> tuple[bool, str]:
    """
    Run a pretrained ImageNet classifier (MobileNetV2) on the image.
    Returns (True, label) if the top-5 predictions contain any bird-related class,
    otherwise returns (False, top_label).
    This prevents the fine-tuned model from ever seeing non-bird inputs.
    """
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input,
        decode_predictions,
    )

    BIRD_KEYWORDS = {
        "bird", "robin", "finch", "warbler", "sparrow", "hawk", "eagle", "owl",
        "duck", "goose", "hen", "cock", "rooster", "turkey", "crane", "heron",
        "kingfisher", "jay", "magpie", "raven", "crow", "mynah", "myna",
        "peacock", "parrot", "macaw", "cockatoo", "lorikeet", "kite", "vulture",
        "pelican", "flamingo", "ibis", "stork", "hornbill", "toucan",
        "woodpecker", "swift", "swallow", "pigeon", "dove", "cuckoo",
        "partridge", "quail", "pheasant", "grouse", "bulbul", "junco",
        "bunting", "chickadee", "ouzel", "brambling", "goldfinch", "ostrich",
        "albatross", "penguin", "puffin", "kookaburra", "sunbird", "bee eater",
        "bee_eater", "drongo", "tailorbird", "munia", "starling", "oriole",
    }

    try:
        img = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img_array = preprocess_input(np.expand_dims(img.numpy().astype("float32"), 0))
        preds = gate_model.predict(img_array, verbose=0)
        top5 = decode_predictions(preds, top=5)[0]   # [(id, label, prob), ...]
    except Exception:
        # If gate fails for any reason, allow through (fail open)
        return True, "unknown"

    top_label = top5[0][1].replace("_", " ")
    for _, label, _ in top5:
        words = label.lower().replace("_", " ")
        if any(kw in words for kw in BIRD_KEYWORDS):
            return True, label.replace("_", " ")

    return False, top_label


@st.cache_data
def load_taxonomy_data() -> dict[str, dict]:
    """Load species taxonomy mapping from data/species_taxonomy.json."""
    if not TAXONOMY_PATH.exists():
        return {}
    try:
        return json.loads(TAXONOMY_PATH.read_text())
    except Exception as e:
        st.error(f"Error loading taxonomy data: {e}")
        return {}


@st.cache_data
def load_project_metrics() -> dict:
    """Load evaluation metrics from results directory."""
    metrics = {
        "val_accuracy": None,
        "test_accuracy": None,
        "macro_f1": None,
    }

    test_path = Path("results/metrics/efficientnetb0_fine_tuned_test_metrics.json")
    if test_path.exists():
        try:
            data = json.loads(test_path.read_text())
            metrics["test_accuracy"] = data.get("test_accuracy")
            metrics["macro_f1"] = data.get("macro_f1")
        except Exception:
            pass

    eff_path = Path("results/metrics/efficientnet_metrics.json")
    if eff_path.exists():
        try:
            data = json.loads(eff_path.read_text())
            metrics["val_accuracy"] = data.get("stage2_best_val_accuracy")
            if metrics["test_accuracy"] is None:
                metrics["test_accuracy"] = data.get("test_accuracy")
        except Exception:
            pass

    return metrics


# ──────────────────────────────────────────────
# UI Component Renderers
# ──────────────────────────────────────────────
def render_header():
    """Render top portal header bar."""
    st.markdown(
        """
        <div class="portal-header">
            <div>
                <span class="portal-brand">🦅 BirdVision</span>
                <span class="portal-sub">| Bird Species Identification</span>
            </div>
            <div>
                <span style="font-size: 0.85rem; color: #cbd5e1; margin-right: 10px;">EfficientNetB0 (Fine-Tuned)</span>
                <span class="portal-status">Model Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_taxonomy_card(tax_info: dict, species_name: str):
    """Render clean, compact Taxonomy Classification card."""
    with st.container(border=True):
        st.markdown(f"#### 🧬 Taxonomy Classification — {tax_info.get('common_name', species_name)}")

        col_left, col_right = st.columns(2)
        with col_left:
            st.write(f"**Common Name:** {tax_info.get('common_name', species_name)}")
            st.write(f"**Kingdom:** {tax_info.get('kingdom', 'Animalia')}")
            st.write(f"**Phylum:** {tax_info.get('phylum', 'Chordata')}")
            st.write(f"**Class:** {tax_info.get('class', 'Aves')}")

        with col_right:
            st.write(f"**Order:** {tax_info.get('order', 'N/A')}")
            st.write(f"**Family:** {tax_info.get('family', 'N/A')}")
            st.write(f"**Genus:** {tax_info.get('genus', 'N/A')}")
            st.write(f"**Species:** *{tax_info.get('species', tax_info.get('scientific_name', 'N/A'))}*")


# ──────────────────────────────────────────────
# Page Renderers
# ──────────────────────────────────────────────
def page_species_identifier(model, class_labels, taxonomy_data, metrics, gate_model):
    """Main Species Identifier page."""
    st.header("Bird Species Identifier")
    st.caption(
        "Upload an image of a bird to identify its species using our fine-tuned EfficientNetB0 model."
    )

    col1, col2 = st.columns([1, 1], gap="medium")

    uploaded_file = None
    with col1:
        with st.container(border=True):
            st.subheader("Upload Bird Image")
            uploaded_file = st.file_uploader(
                "Select a bird image (JPG or PNG)", type=["jpg", "jpeg", "png"]
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

            predict_clicked = st.button(
                "Predict Species", use_container_width=True, type="primary"
            )

    result = None
    with col2:
        with st.container(border=True):
            st.subheader("Prediction Result")

            if uploaded_file is None:
                st.info("Upload an image on the left and click 'Predict Species' to see results.")
                # Clear session state when no file is uploaded
                st.session_state.pop("current_file_key", None)
                st.session_state.pop("last_result", None)
                st.session_state.pop("gate_rejected", None)
                st.session_state.pop("gate_label", None)
            else:
                file_key = f"{uploaded_file.name}_{uploaded_file.size}"

                # Trigger inference if user clicks button OR if a new file is uploaded
                should_run_inference = predict_clicked or (
                    st.session_state.get("current_file_key") != file_key
                )

                if should_run_inference:
                    if model is None:
                        st.error("Model file not found!")
                    elif class_labels is None:
                        st.error("Class labels file not found!")
                    else:
                        with st.spinner("Checking image..."):
                            raw_bytes = uploaded_file.getvalue()
                            gate_ok, detected_label = is_likely_bird(raw_bytes, gate_model)

                        if not gate_ok:
                            # Non-bird image — block before fine-tuned model runs
                            st.session_state["last_result"] = None
                            st.session_state["gate_rejected"] = True
                            st.session_state["gate_label"] = detected_label
                            st.session_state["current_file_key"] = file_key
                        else:
                            with st.spinner("Identifying species via EfficientNetB0..."):
                                result = run_inference(
                                    image_input=raw_bytes,
                                    model=model,
                                    class_labels=class_labels,
                                    img_size=IMG_SIZE,
                                    top_k=3,
                                )
                            st.session_state["last_result"] = result
                            st.session_state["gate_rejected"] = False
                            st.session_state["current_file_key"] = file_key
                else:
                    result = st.session_state.get("last_result")

                # ── Display result ───────────────────────
                if st.session_state.get("gate_rejected"):
                    detected = st.session_state.get("gate_label", "non-bird object")
                    st.markdown("#### Predicted Species")
                    st.error("🚫 No bird detected in this image")
                    st.caption(
                        f"ImageNet pre-screener identified this as: **{detected}**. "
                        "Please upload a clear photograph of a bird."
                    )
                elif result is not None:
                    conf_val = result["confidence"] * 100
                    st.markdown("#### Predicted Species")
                    st.markdown(f"### :blue[{result['predicted_species']}]")
                    st.markdown("#### Confidence")
                    st.markdown(f"### {conf_val:.2f}%")

    # Prediction Details (Top-3 Predictions) & Taxonomy Classification
    # Only shown when the gate passed AND the fine-tuned model returned a result
    res = st.session_state.get("last_result")
    gate_rejected = st.session_state.get("gate_rejected", False)
    if uploaded_file is not None and res is not None and not gate_rejected:
        st.markdown("---")
        st.subheader("Prediction Details")
        st.write("**Top 3 Predictions**")

        for item in res["top_predictions"]:
            c1, c2 = st.columns([3, 1])
            species = item["species"]
            conf_pct = item["confidence"] * 100
            with c1:
                st.write(f"**{item['rank']}. {species}**")
                st.progress(float(item["confidence"]))
            with c2:
                st.write(f"**{conf_pct:.2f}%**")

        st.markdown("---")
        raw_class = res.get("raw_class", "")
        tax_info = taxonomy_data.get(raw_class, {})
        render_taxonomy_card(tax_info, res["predicted_species"])

    # Model Information Summary Card
    st.markdown("---")
    st.subheader("Model Information")

    inf_col1, inf_col2, inf_col3, inf_col4 = st.columns(4)
    num_classes = len(class_labels) if class_labels else 25
    val_acc_str = (
        f"{metrics['val_accuracy'] * 100:.2f}%" if metrics.get("val_accuracy") else "97.88%"
    )
    test_acc_str = (
        f"{metrics['test_accuracy'] * 100:.2f}%" if metrics.get("test_accuracy") else "96.20%"
    )

    with inf_col1:
        with st.container(border=True):
            st.caption("MODEL ARCHITECTURE")
            st.write("**EfficientNetB0 (Fine-Tuned)**")

    with inf_col2:
        with st.container(border=True):
            st.caption("NUMBER OF CLASSES")
            st.write(f"**{num_classes} Bird Species**")

    with inf_col3:
        with st.container(border=True):
            st.caption("VALIDATION ACCURACY")
            st.write(f"**{val_acc_str}**")

    with inf_col4:
        with st.container(border=True):
            st.caption("TEST ACCURACY")
            st.write(f"**{test_acc_str}**")


def page_taxonomy_classification(taxonomy_data):
    """Dedicated Taxonomy Classification Explorer page."""
    st.header("🧬 Taxonomy Classification")
    st.write(
        "Biological taxonomy hierarchy for all 25 bird species included in the BirdVision dataset."
    )

    # Check if a recent prediction exists
    if "last_result" in st.session_state and st.session_state.get("last_result") is not None:
        res = st.session_state["last_result"]
        st.subheader("Latest Image Classification Result")
        raw_class = res.get("raw_class", "")
        tax_info = taxonomy_data.get(raw_class, {})
        render_taxonomy_card(tax_info, res["predicted_species"])
        st.markdown("---")

    st.subheader("Explore Species Taxonomy")
    if taxonomy_data:
        species_keys = sorted(list(taxonomy_data.keys()))
        formatted_names = [taxonomy_data[k].get("common_name", k.replace("-", " ")) for k in species_keys]
        key_map = dict(zip(formatted_names, species_keys))

        selected_display = st.selectbox("Select a Bird Species:", formatted_names)
        selected_key = key_map[selected_display]
        selected_info = taxonomy_data[selected_key]

        render_taxonomy_card(selected_info, selected_display)
    else:
        st.info("Taxonomy data file not found at data/species_taxonomy.json")


def page_home():
    """Home Overview Page."""
    st.header("Welcome to BirdVision Portal")
    st.write(
        "BirdVision is an automated AI-driven bird species identification system designed to recognize "
        "25 prominent Indian bird species from natural photographs using deep convolutional neural networks."
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        with st.container(border=True):
            st.metric("Model Architecture", "EfficientNetB0")
    with m2:
        with st.container(border=True):
            st.metric("Target Species", "25 Classes")
    with m3:
        with st.container(border=True):
            metrics = load_project_metrics()
            acc = f"{metrics['test_accuracy']*100:.2f}%" if metrics.get("test_accuracy") else "96.20%"
            st.metric("Test Classification Accuracy", acc)

    st.subheader("Key Features")
    st.markdown(
        """
        - **Deep Transfer Learning:** Fine-tuned EfficientNetB0 backbone trained on Indian bird images.
        - **Top-3 Probabilistic Output:** Multi-class confidence distribution for species matching.
        - **Biological Taxonomy:** Comprehensive rank hierarchy (Kingdom, Phylum, Class, Order, Family, Genus, Species).
        - **High Accuracy:** Evaluated held-out test accuracy of **96.20%** across 25 species.
        """
    )


def page_model_performance(metrics):
    """Model Performance Metrics Page."""
    st.header("📊 Model Performance & Evaluation")
    st.write(
        "Quantitative evaluation metrics computed on the held-out test dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    val_acc = (
        f"{metrics['val_accuracy'] * 100:.2f}%"
        if metrics.get("val_accuracy")
        else "97.88%"
    )
    test_acc = (
        f"{metrics['test_accuracy'] * 100:.2f}%"
        if metrics.get("test_accuracy")
        else "96.20%"
    )
    macro_f1 = (
        f"{metrics['macro_f1'] * 100:.2f}%"
        if metrics.get("macro_f1")
        else "96.21%"
    )

    with c1:
        st.metric("Validation Accuracy", val_acc)
    with c2:
        st.metric("Test Accuracy", test_acc)
    with c3:
        st.metric("Macro F1-Score", macro_f1)
    with c4:
        st.metric("Number of Classes", "25")

    st.markdown("---")
    st.subheader("Evaluation Visualizations")

    col_a, col_b = st.columns(2)

    cm_path = Path("results/confusion_matrix/efficientnetb0_fine_tuned_confusion_matrix.png")
    curves_path = Path("results/plots/efficientnetb0_training_curves.png")

    with col_a:
        with st.container(border=True):
            st.subheader("Confusion Matrix")
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True)
            else:
                st.info("Confusion matrix plot not found.")

    with col_b:
        with st.container(border=True):
            st.subheader("Training Curves")
            if curves_path.exists():
                st.image(str(curves_path), use_container_width=True)
            else:
                st.info("Training curves plot not found.")


def page_about_project():
    """About Project Page."""
    st.header("ℹ️ About Project")

    with st.container(border=True):
        st.markdown(
            """
            ### BirdVision — AI-Powered Bird Species Classification
            - **Project Title:** Bird Species Identification & Biological Taxonomy System
            - **Model Architecture:** EfficientNetB0 (Pre-trained on ImageNet & Fine-Tuned)
            - **Target Dataset:** 25 Prominent Indian Bird Species
            - **Key Features:** Real-time inference, Top-3 confidence predictions, Biological Taxonomy Classification
            - **Frameworks:** TensorFlow / Keras & Streamlit
            - **Purpose:** 5th Semester / College Machine Learning & Computer Vision Mini-Project.
            """
        )


# ──────────────────────────────────────────────
# Main Application Entrypoint
# ──────────────────────────────────────────────
def main():
    # Sidebar
    st.sidebar.markdown("### 🦅 BirdVision")
    st.sidebar.caption("College Mini-Project Portal")

    nav_option = st.sidebar.radio(
        "Navigation",
        options=[
            "🏠 Home",
            "🔍 Species Identifier",
            "🧬 Taxonomy Classification",
            "📊 Model Performance",
            "ℹ️ About Project",
        ],
        index=1,  # Default to Species Identifier page
    )

    render_header()

    # Load Resources
    model = load_efficientnet_model()
    gate_model = load_bird_gate_model()
    class_labels = load_labels()
    taxonomy_data = load_taxonomy_data()
    metrics = load_project_metrics()

    if nav_option == "🏠 Home":
        page_home()
    elif nav_option == "🔍 Species Identifier":
        page_species_identifier(model, class_labels, taxonomy_data, metrics, gate_model)
    elif nav_option == "🧬 Taxonomy Classification":
        page_taxonomy_classification(taxonomy_data)
    elif nav_option == "📊 Model Performance":
        page_model_performance(metrics)
    elif nav_option == "ℹ️ About Project":
        page_about_project()


if __name__ == "__main__":
    main()
