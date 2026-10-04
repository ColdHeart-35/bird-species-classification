"""
BirdVision — AI-Powered Bird Species Identification
Professional Streamlit Application for College ML Mini-Project Presentation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

# Add project root to sys.path if needed
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import IMG_SIZE, LABEL_MAP_PATH, load_class_labels
from src.predict_cnn import run_inference, MODEL_CONFIGS

# ──────────────────────────────────────────────
# Page Configuration & Custom CSS Styling
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="BirdVision — AI Bird Species Identification",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern dark/light bird-themed presentation UI
st.markdown(
    """
    <style>
    /* Global styling overrides */
    .main {
        background-color: #0e1117;
    }
    
    /* Header hero styling */
    .hero-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #11998e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #ffffff, #a8ff78);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 500;
        color: #e0e0e0;
        margin-bottom: 0.8rem;
    }
    
    .hero-desc {
        font-size: 1.05rem;
        color: #b0bec5;
        max-width: 750px;
        margin: 0 auto;
        line-height: 1.5;
    }
    
    /* Card components */
    .custom-card {
        background-color: #1a1f2c;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    .result-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 2px solid #38bdf8;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.15);
        margin-bottom: 1.5rem;
    }
    
    .predicted-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .predicted-name {
        font-size: 2.3rem;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 0.5rem;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid #38bdf8;
        color: #38bdf8;
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
    }
    
    /* Taxonomy Table Styling */
    .taxonomy-card {
        background-color: #1a202c;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1rem;
    }
    
    .taxonomy-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #2d3748;
    }
    
    .taxonomy-row:last-child {
        border-bottom: none;
    }
    
    .taxonomy-label {
        color: #9ca3af;
        font-weight: 600;
        width: 40%;
    }
    
    .taxonomy-value {
        color: #f3f4f6;
        font-weight: 500;
        width: 60%;
    }
    
    /* Metrics summary boxes */
    .metric-box {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Caching Model and Resource Loaders
# ──────────────────────────────────────────────
@st.cache_resource
def load_efficientnet_model():
    """Load the fine-tuned EfficientNetB0 Keras model (cached across sessions)."""
    import tensorflow as tf
    from tensorflow import keras

    model_path = MODEL_CONFIGS["efficientnet"]["path"]
    if not model_path.exists():
        return None
    return keras.models.load_model(str(model_path))


@st.cache_data
def load_cached_class_labels() -> dict[int, str] | None:
    """Load class label mapping (cached)."""
    if not LABEL_MAP_PATH.exists():
        return None
    return load_class_labels(LABEL_MAP_PATH)


@st.cache_data
def load_cached_taxonomy() -> dict[str, dict]:
    """Load species taxonomy dictionary from data/species_taxonomy.json."""
    tax_path = Path("data/species_taxonomy.json")
    if tax_path.exists():
        try:
            return json.loads(tax_path.read_text())
        except Exception:
            return {}
    return {}


@st.cache_data
def load_cached_metrics() -> dict | None:
    """Load test metrics JSON if available."""
    metrics_path = Path("results/metrics/efficientnetb0_fine_tuned_test_metrics.json")
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except Exception:
            return None
    return None


# ──────────────────────────────────────────────
# Main Application Layout
# ──────────────────────────────────────────────
def main():
    # Sidebar Setup
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #38bdf8; margin: 0;">🦅 BirdVision</h2>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">EfficientNetB0 Species ID</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🦅 Species Identifier",
            "📚 Species Taxonomy Explorer",
            "📊 Model Performance & Metrics",
            "ℹ️ About Model & Project",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Project Status:**  
        ✅ Model: EfficientNetB0 (Fine-Tuned)  
        🎯 Val Accuracy: **96.50%**  
        🧪 Test Accuracy: **96.20%**  
        🏷️ Classes: **25 Bird Species**  
        """
    )

    # ──────────────────────────────────────────────
    # PAGE 1: SPECIES IDENTIFIER
    # ──────────────────────────────────────────────
    if page == "🦅 Species Identifier":
        # Header Banner
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-title">BirdVision</div>
                <div class="hero-subtitle">AI-Powered Bird Species Identification</div>
                <div class="hero-desc">
                    Upload a bird image and let our fine-tuned EfficientNetB0 model identify the species with high precision and full taxonomic classification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("### 📤 Upload Bird Image")
            uploaded_file = st.file_uploader(
                "Upload a Bird Image",
                type=["jpg", "jpeg", "png"],
                help="Supports JPG, JPEG, and PNG images",
            )

            if uploaded_file is not None:
                try:
                    # Load PIL image for preview
                    image = Image.open(uploaded_file)
                    st.image(
                        image,
                        caption=f"Uploaded Image: {uploaded_file.name}",
                        use_column_width=True,
                    )

                    st.markdown(
                        f"""
                        <div style="background: #1e293b; padding: 0.75rem 1rem; border-radius: 8px; font-size: 0.9rem; color: #94a3b8; margin-top: 0.5rem;">
                            📄 <b>File:</b> {uploaded_file.name} | 
                            📐 <b>Dimensions:</b> {image.width} × {image.height} px | 
                            💾 <b>Size:</b> {uploaded_file.size / 1024:.1f} KB
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error opening image file: {str(e)}")

        with col_right:
            st.markdown("### 🔍 Model Inference")

            if uploaded_file is None:
                st.info("👈 Upload an image on the left to run species identification.")
            else:
                identify_btn = st.button(
                    "Identify Bird", type="primary", use_container_width=True
                )

                if identify_btn:
                    # Load model & labels (cached)
                    with st.spinner("Loading EfficientNetB0 model & running inference..."):
                        model = load_efficientnet_model()
                        class_labels = load_cached_class_labels()
                        taxonomy_db = load_cached_taxonomy()

                    if model is None:
                        st.error(
                            "❌ Model file missing at `models/efficientnet_finetuned.keras`. Please verify models directory."
                        )
                    elif class_labels is None:
                        st.error(
                            "❌ Class labels missing at `models/class_labels.json`."
                        )
                    else:
                        try:
                            # Reset pointer and read bytes
                            uploaded_file.seek(0)
                            raw_bytes = uploaded_file.read()

                            # Run prediction using exact same pipeline as CLI
                            result = run_inference(
                                image_input=raw_bytes,
                                model=model,
                                class_labels=class_labels,
                                img_size=IMG_SIZE,
                                top_k=3,
                            )

                            predicted_species = result["predicted_species"]
                            confidence = result["confidence"]
                            raw_class = result["raw_class"]
                            top_predictions = result["top_predictions"]

                            # Store prediction result in session state to maintain state across interactions
                            st.session_state["last_result"] = result
                        except Exception as err:
                            st.error(f"An error occurred during prediction: {err}")

                # Display Results if available in session_state
                if "last_result" in st.session_state:
                    res = st.session_state["last_result"]
                    predicted_species = res["predicted_species"]
                    confidence = res["confidence"]
                    raw_class = res["raw_class"]
                    top_predictions = res["top_predictions"]
                    taxonomy_db = load_cached_taxonomy()

                    st.markdown("---")
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="predicted-title">PREDICTED SPECIES</div>
                            <div class="predicted-name">{predicted_species}</div>
                            <div class="confidence-badge">Confidence: {confidence:.2%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Top 3 Predictions Progress Bars
                    st.markdown("#### 📊 Top 3 Predictions")
                    for pred in top_predictions:
                        rank = pred["rank"]
                        spec = pred["species"]
                        conf = pred["confidence"]

                        st.write(f"**{rank}. {spec}** — `{conf:.2%}`")
                        st.progress(min(max(float(conf), 0.0), 1.0))

                    # Species Taxonomy Section
                    st.markdown("---")
                    st.markdown("### 🧬 Species Taxonomy")

                    tax_data = taxonomy_db.get(raw_class, None)
                    if tax_data:
                        st.markdown(
                            f"""
                            <div class="taxonomy-card">
                                <div class="taxonomy-row">
                                    <span class="taxonomy-label">Common Name</span>
                                    <span class="taxonomy-value"><b>{tax_data.get('common_name', predicted_species)}</b></span>
                                </div>
                                <div class="taxonomy-row">
                                    <span class="taxonomy-label">Scientific Name</span>
                                    <span class="taxonomy-value"><i>{tax_data.get('scientific_name', 'N/A')}</i></span>
                                </div>
                                <div class="taxonomy-row">
                                    <span class="taxonomy-label">Kingdom</span>
                                    <span class="taxonomy-value">{tax_data.get('kingdom', 'Animalia')}</span>
                                </div>
                                <div class="taxonomy-row">
                                    <span class="taxonomy-label">Phylum</span>
                                    <span class="taxonomy-value">{tax_data.get('phylum', 'Chordata')}</span>
                                </div>
                                <div class="taxonomy-row">
                                    <span class="taxonomy-label">Class</span>
                                    <span class="taxonomy-value">{tax_data.get('class', 'Aves')}</span>
                                </div>
                                <div class="taxonomy-row">
                                    <span class="taxonomy-label">Order</span>
                                    <span class="taxonomy-value">{tax_data.get('order', 'N/A')}</span>
                                </div>
                                <div class="taxonomy-row">
                                    <span class="taxonomy-label">Family</span>
                                    <span class="taxonomy-value">{tax_data.get('family', 'N/A')}</span>
                                </div>
                                <div class="taxonomy-row">
                                    <span class="taxonomy-label">Genus</span>
                                    <span class="taxonomy-value"><i>{tax_data.get('genus', 'N/A')}</i></span>
                                </div>
                                <div class="taxonomy-row">
                                    <span class="taxonomy-label">Species</span>
                                    <span class="taxonomy-value"><i>{tax_data.get('species', 'N/A')}</i></span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning(
                            f"Taxonomy information not found in database for class: `{raw_class}`"
                        )

    # ──────────────────────────────────────────────
    # PAGE 2: SPECIES TAXONOMY EXPLORER
    # ──────────────────────────────────────────────
    elif page == "📚 Species Taxonomy Explorer":
        st.markdown("# 📚 25 Bird Species Taxonomy Database")
        st.write(
            "Explore the full biological taxonomy for all 25 bird species recognized by the EfficientNetB0 model."
        )

        taxonomy_db = load_cached_taxonomy()
        class_labels = load_cached_class_labels()

        if not taxonomy_db or not class_labels:
            st.error("Taxonomy database or class labels not available.")
        else:
            search = st.text_input("🔍 Search species or scientific name...", "")

            species_keys = sorted(taxonomy_db.keys())
            if search:
                species_keys = [
                    k
                    for k in species_keys
                    if search.lower() in k.lower()
                    or search.lower()
                    in taxonomy_db[k].get("scientific_name", "").lower()
                    or search.lower() in taxonomy_db[k].get("common_name", "").lower()
                ]

            st.write(f"Showing **{len(species_keys)}** species:")

            for raw_class in species_keys:
                item = taxonomy_db[raw_class]
                with st.expander(
                    f"🦜 {item.get('common_name', raw_class)} ({item.get('scientific_name', '')})"
                ):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"**Common Name:** {item.get('common_name')}")
                        st.markdown(
                            f"**Scientific Name:** *{item.get('scientific_name')}*"
                        )
                        st.markdown(f"**Order:** {item.get('order')}")
                        st.markdown(f"**Family:** {item.get('family')}")
                    with cols[1]:
                        st.markdown(f"**Kingdom:** {item.get('kingdom')}")
                        st.markdown(f"**Phylum:** {item.get('phylum')}")
                        st.markdown(f"**Class:** {item.get('class')}")
                        st.markdown(
                            f"**Genus / Species:** *{item.get('genus')} {item.get('species')}*"
                        )

    # ──────────────────────────────────────────────
    # PAGE 3: MODEL PERFORMANCE & METRICS
    # ──────────────────────────────────────────────
    elif page == "📊 Model Performance & Metrics":
        st.markdown("# 📊 Model Performance & Evaluation Metrics")
        st.write(
            "Quantitative evaluation metrics and visualization artifacts generated during model evaluation."
        )

        # Overview metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                """
                <div class="metric-box">
                    <div class="metric-value">96.50%</div>
                    <div class="metric-label">Validation Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                """
                <div class="metric-box">
                    <div class="metric-value">96.20%</div>
                    <div class="metric-label">Test Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                """
                <div class="metric-box">
                    <div class="metric-value">96.34%</div>
                    <div class="metric-label">Macro Precision</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                """
                <div class="metric-box">
                    <div class="metric-value">96.21%</div>
                    <div class="metric-label">Macro F1-Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Tabs for Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Model Comparison",
                "🎯 Confusion Matrix",
                "📉 Training Curves",
                "📊 Per-Class Accuracy",
            ]
        )

        with tab1:
            st.subheader("Model Comparison: HOG + SVM vs EfficientNetB0")
            comp_path = Path("results/plots/model_comparison.png")
            if comp_path.exists():
                st.image(
                    str(comp_path),
                    caption="HOG + Linear SVM (17.8%) vs EfficientNetB0 Fine-Tuned (96.2%)",
                    use_column_width=True,
                )
            else:
                st.info("Model comparison plot not found.")

        with tab2:
            st.subheader("Confusion Matrix (25 Species Test Set)")
            cm_path = Path(
                "results/confusion_matrix/efficientnetb0_fine_tuned_confusion_matrix.png"
            )
            if cm_path.exists():
                st.image(
                    str(cm_path),
                    caption="Confusion Matrix for EfficientNetB0 Fine-Tuned Model",
                    use_column_width=True,
                )
            else:
                st.info("Confusion matrix plot not found.")

        with tab3:
            st.subheader("EfficientNetB0 Two-Stage Training Curves")
            curves_path = Path("results/plots/efficientnetb0_training_curves.png")
            if curves_path.exists():
                st.image(
                    str(curves_path),
                    caption="Stage 1 (Top Layers) & Stage 2 (Fine-tuning) Training vs Validation Accuracy & Loss",
                    use_column_width=True,
                )
            else:
                st.info("Training curves plot not found.")

        with tab4:
            st.subheader("Per-Class Test Accuracy Breakdown")
            per_class_path = Path(
                "results/plots/efficientnetb0_fine_tuned_per_class_accuracy.png"
            )
            if per_class_path.exists():
                st.image(
                    str(per_class_path),
                    caption="Test Accuracy breakdown across all 25 bird species",
                    use_column_width=True,
                )
            else:
                st.info("Per-class accuracy plot not found.")

    # ──────────────────────────────────────────────
    # PAGE 4: ABOUT MODEL & PROJECT
    # ──────────────────────────────────────────────
    elif page == "ℹ️ About Model & Project":
        st.markdown("# ℹ️ About the Model & Architecture")

        st.markdown(
            """
            ### 🏗️ Architecture: EfficientNetB0 (Fine-Tuned)
            
            This application uses transfer learning with **EfficientNetB0** pre-trained on ImageNet, fine-tuned specifically for the 25-species bird dataset.
            
            #### ⚙️ Key Technical Highlights:
            - **Input Resolution:** `224 × 224 × 3`
            - **Preprocessing:** Resized RGB input, scaled pixel values strictly to `[0.0, 1.0]` (matching standard training pipeline).
            - **Two-Stage Fine-Tuning Strategy:**
              1. **Stage 1 (Feature Extraction):** Frozen backbone, trained custom classification head (Global Average Pooling → Dropout(0.3) → Dense(256) → Dense(25, Softmax)). (10 epochs, learning rate = 0.001)
              2. **Stage 2 (Fine-Tuning):** Unfrozen top 30 layers of EfficientNetB0 for end-to-end gradient updates with a small learning rate (15 epochs, learning rate = 1e-5).
            
            #### 📊 Performance Summary:
            - **Validation Accuracy:** `96.50%`
            - **Test Accuracy:** `96.20%` (evaluated on held-out test split of 1,500 images)
            - **Macro Precision:** `~96.34%`
            - **Macro Recall:** `~96.20%`
            - **Macro F1 Score:** `~96.21%`
            
            > 💡 **Note on Model Accuracy vs Prediction Confidence:**
            > - **Model Test Accuracy (96.20%)** measures the overall proportion of correct classifications made by the model across the entire benchmark test set.
            > - **Prediction Confidence (e.g. 98.4%)** is the Softmax probability score assigned by the neural network to a specific image during a single inference pass.
            """
        )


if __name__ == "__main__":
    main()
