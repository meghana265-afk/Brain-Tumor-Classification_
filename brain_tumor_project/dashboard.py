"""
Brain Tumor Classification - Interactive Streamlit Dashboard
A comprehensive web-based interface for the brain tumor classification project.

Run with: streamlit run brain_tumor_project/dashboard.py
"""

import streamlit as st
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    tf = None

# Define constants directly (no src import to avoid conflicts)
IMG_SIZE = 150
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
PROJECT_ROOT = Path(__file__).parent
TRAIN_DIR = PROJECT_ROOT / "Training"
TEST_DIR = PROJECT_ROOT / "Testing"
MODEL_PATH = PROJECT_ROOT / "models"


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
        }
        h1 {
            color: #1f77b4;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 1rem;
        }
        h2 {
            color: #0d47a1;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🧠 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a Page:",
    ["🏠 Home", "📊 Dataset", "🤖 Models", "🔍 Prediction", "📈 Results", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Brain Tumor Classification Project**\n\n"
    "Deep learning system for MRI-based brain tumor classification.\n\n"
    "- 7,023 images\n"
    "- 4 tumor classes\n"
    "- 2 models (CNN + VGG16)\n"
    "- 90% accuracy"
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_model_cached(model_path):
    """Load trained model with caching."""
    if not TF_AVAILABLE:
        st.error("TensorFlow is not available. Please install it.")
        return None
    
    model_path = Path(model_path)
    if not model_path.exists():
        st.warning(f"Model file not found: {model_path}")
        return None
    
    try:
        with st.spinner(f"Loading model from {model_path.name}..."):
            model = tf.keras.models.load_model(str(model_path))
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def get_dataset_stats():
    """Get dataset statistics."""
    stats = {
        "Total Images": 7023,
        "Training Images": 5712,
        "Testing Images": 1311,
        "Image Size": "150x150 pixels",
        "Classes": 4,
        "Class Names": ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    }
    
    class_distribution = {
        "Glioma": {"train": 1321, "test": 300},
        "Meningioma": {"train": 1339, "test": 306},
        "No Tumor": {"train": 1595, "test": 405},
        "Pituitary": {"train": 1457, "test": 300}
    }
    
    return stats, class_distribution


def preprocess_image(image_path, size=IMG_SIZE):
    """Preprocess image for model prediction."""
    try:
        img = load_img(image_path, target_size=(size, size))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None


def predict_image(model, image_array):
    """Make prediction on image."""
    if model is None or image_array is None:
        st.error("Model or image array is None")
        return None
    
    try:
        # Suppress TensorFlow output
        with st.spinner("Analyzing image..."):
            predictions = model.predict(image_array, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        return {
            "class": CLASS_NAMES[predicted_class_idx],
            "confidence": confidence,
            "probabilities": predictions[0]
        }
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None


# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    st.title("🧠 Brain Tumor Classification System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Interactive Dashboard
        
        This web-based interface provides:
        
        ### 🎯 Key Features
        - **Dataset Explorer**: View dataset statistics and distribution
        - **Model Information**: Learn about baseline and enhanced models
        - **Image Prediction**: Upload MRI scans for instant classification
        - **Results Analysis**: View model performance metrics
        - **Project Details**: Complete project information
        
        ### 📊 Project Overview
        - **Total Dataset**: 7,023 brain MRI images
        - **Classes**: 4 tumor types (Glioma, Meningioma, No Tumor, Pituitary)
        - **Models**: Baseline CNN (50%) + Enhanced VGG16 (90%)
        - **Accuracy**: 90%+ with enhanced model
        
        ### 🚀 Quick Start
        1. Go to **Dataset** page to explore data
        2. Check **Models** page to learn architectures
        3. Try **Prediction** to classify an image
        4. View **Results** for performance metrics
        
        ### ⚠️ Disclaimer
        This system is for **educational purposes only**.
        Not approved for clinical use without proper validation.
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Quick Stats
        """)
        st.metric("Total Images", "7,023")
        st.metric("Train/Test Split", "81% / 19%")
        st.metric("Image Size", "150x150")
        st.metric("Model Accuracy", "90%+")


# ============================================================================
# PAGE: DATASET
# ============================================================================

elif page == "📊 Dataset":
    st.title("📊 Dataset Overview")
    
    stats, class_dist = get_dataset_stats()
    
    # Dataset statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", stats["Total Images"])
    with col2:
        st.metric("Training Set", stats["Training Images"])
    with col3:
        st.metric("Testing Set", stats["Testing Images"])
    with col4:
        st.metric("Classes", stats["Classes"])
    
    st.markdown("---")
    
    # Class distribution
    st.subheader("Class Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Training Data")
        train_data = {
            class_name: class_dist[class_name]["train"] 
            for class_name in ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        }
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
        ax.bar(train_data.keys(), train_data.values(), color=colors)
        ax.set_ylabel("Number of Images", fontsize=12)
        ax.set_title("Training Data Distribution", fontsize=14, fontweight="bold")
        ax.tick_params(axis='x', rotation=45)
        for i, (k, v) in enumerate(train_data.items()):
            ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Testing Data")
        test_data = {
            class_name: class_dist[class_name]["test"] 
            for class_name in ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        }
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
        ax.bar(test_data.keys(), test_data.values(), color=colors)
        ax.set_ylabel("Number of Images", fontsize=12)
        ax.set_title("Testing Data Distribution", fontsize=14, fontweight="bold")
        ax.tick_params(axis='x', rotation=45)
        for i, (k, v) in enumerate(test_data.items()):
            ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Class information table
    st.subheader("Class Information")
    
    class_info = {
        "Class Name": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "Description": [
            "Malignant brain tumor",
            "Tumor in brain lining",
            "Healthy brain scan",
            "Pituitary gland tumor"
        ],
        "Train Images": [1321, 1339, 1595, 1457],
        "Test Images": [300, 306, 405, 300]
    }
    
    df_classes = pd.DataFrame(class_info)
    st.dataframe(df_classes, use_container_width=True)


# ============================================================================
# PAGE: MODELS
# ============================================================================

elif page == "🤖 Models":
    st.title("🤖 Model Architecture & Details")
    
    model_choice = st.radio("Select Model:", ["Baseline CNN", "Enhanced VGG16"])
    
    if model_choice == "Baseline CNN":
        st.subheader("Baseline CNN Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Architecture
            
            **4 Convolutional Blocks:**
            - Conv2D (32 filters, 3x3) → ReLU → MaxPool → BatchNorm
            - Conv2D (64 filters, 3x3) → ReLU → MaxPool → BatchNorm
            - Conv2D (128 filters, 3x3) → ReLU → MaxPool → BatchNorm
            - Conv2D (256 filters, 3x3) → ReLU → MaxPool → BatchNorm
            
            **Classification Head:**
            - Flatten
            - Dense (256) → ReLU → Dropout (0.5)
            - Dense (128) → ReLU → Dropout (0.3)
            - Dense (4) → Softmax
            
            ### Training Configuration
            - **Optimizer**: Adam (lr=0.001)
            - **Loss Function**: Categorical Crossentropy
            - **Batch Size**: 32
            - **Epochs**: 10
            - **Callbacks**: Early Stopping, Model Checkpoint
            """)
        
        with col2:
            st.markdown("""
            ### Specifications
            
            📊 **Metrics**
            - Parameters: 3.6M
            - Training Time: 10 min
            - Accuracy: 50%
            - F1-Score: 0.44
            
            🎯 **Purpose**
            - Establish baseline
            - Benchmark performance
            - Prove concept
            """)
    
    else:  # Enhanced VGG16
        st.subheader("Enhanced VGG16 Transfer Learning Model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Architecture
            
            **Base Model (VGG16 - Pre-trained):**
            - Block 1-3: Frozen (ImageNet features)
            - Block 4-5: Unfrozen (Fine-tuned)
            
            **Custom Classification Head:**
            - GlobalAveragePooling2D
            - Dense (512) → ReLU → BatchNorm → Dropout (0.5)
            - Dense (256) → ReLU → BatchNorm → Dropout (0.5)
            - Dense (4) → Softmax
            
            ### Two-Stage Training
            
            **Stage 1: Feature Extraction (15 epochs)**
            - Freeze entire VGG16 base
            - Train only top layers
            - Learning rate: 0.001
            
            **Stage 2: Fine-Tuning (25 epochs)**
            - Unfreeze last 4 VGG16 layers
            - Fine-tune entire network
            - Learning rate: 0.0001
            
            ### Advanced Techniques
            - 8 Data Augmentation methods
            - Class weight balancing
            - Learning rate scheduling (ReduceLROnPlateau)
            - Early stopping on validation loss
            """)
        
        with col2:
            st.markdown("""
            ### Specifications
            
            📊 **Metrics**
            - Parameters: 14.7M
            - Training Time: 30 min
            - Accuracy: 90%+
            - F1-Score: 0.89
            
            🎯 **Purpose**
            - Production-ready
            - Transfer learning
            - High accuracy
            """)


# ============================================================================
# PAGE: PREDICTION
# ============================================================================

elif page == "🔍 Prediction":
    st.title("🔍 Brain Tumor Classification")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload MRI Image")
        st.info("Upload a brain MRI scan image (JPG, PNG) for classification")
        
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        model_choice = st.selectbox("Select Model:", ["Baseline CNN", "Enhanced VGG16"])
    
    if uploaded_file is not None:
        with col2:
            st.markdown("### Image Preview & Prediction")
            
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Image", use_column_width=True)
        
        # Make prediction
        with st.spinner("Processing image..."):
            # Save uploaded file temporarily (cross-platform)
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_image_path = tmp_file.name
            
            try:
                # Load and preprocess
                img_array, img = preprocess_image(temp_image_path)
                
                if img_array is not None:
                    # Load appropriate model
                    model_filename = "saved_model.h5" if model_choice == "Baseline CNN" else "best_enhanced_model.h5"
                    model_path = MODEL_PATH / model_filename
                    
                    model = load_model_cached(str(model_path))
                    
                    if model is not None:
                        # Make prediction
                        result = predict_image(model, img_array)
                        
                        if result is not None:
                            st.success("✅ Prediction Complete")
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Predicted Class", result["class"].title())
                            with col2:
                                st.metric("Confidence", f"{result['confidence']*100:.2f}%")
                            with col3:
                                st.metric("Model Used", model_choice)
                            
                            # Probability distribution
                            st.markdown("### Prediction Probabilities")
                            
                            prob_data = pd.DataFrame({
                                "Class": [c.title() for c in CLASS_NAMES],
                                "Probability": result["probabilities"]
                            })
                            prob_data = prob_data.sort_values("Probability", ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            colors = ["#1f77b4" if x == result["class"].title() else "#d3d3d3" for x in prob_data["Class"]]
                            ax.barh(prob_data["Class"], prob_data["Probability"], color=colors)
                            ax.set_xlabel("Probability", fontsize=12)
                            ax.set_title("Class Probability Distribution", fontsize=14, fontweight="bold")
                            ax.set_xlim(0, 1)
                            
                            for i, (idx, row) in enumerate(prob_data.iterrows()):
                                ax.text(row["Probability"] + 0.02, i, f"{row['Probability']:.4f}", 
                                       va='center', fontweight='bold')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.error(f"Could not load model: {model_path}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
    
    else:
        st.info("👆 Upload an image to get started")


# ============================================================================
# PAGE: RESULTS
# ============================================================================

elif page == "📈 Results":
    st.title("📈 Model Performance Results")
    
    st.subheader("Model Comparison")
    
    comparison_data = {
        "Metric": [
            "Accuracy",
            "Precision (Macro)",
            "Recall (Macro)",
            "F1-Score (Macro)",
            "Cohen's Kappa",
            "Parameters",
            "Training Time"
        ],
        "Baseline CNN": [
            "50.11%",
            "0.4228",
            "0.4937",
            "0.4399",
            "0.3342",
            "3.6M",
            "10 min"
        ],
        "Enhanced VGG16": [
            "90-95%",
            "0.9087",
            "0.8823",
            "0.8945",
            "0.8697",
            "14.7M",
            "30 min"
        ],
        "Improvement": [
            "+40%",
            "+48.6%",
            "+38.9%",
            "+45.5%",
            "+53.6%",
            "4x larger",
            "+20 min"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # Per-class performance
    st.subheader("Per-Class Performance (Enhanced Model)")
    
    per_class_data = {
        "Class": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "Precision": [0.89, 0.87, 0.95, 0.92],
        "Recall": [0.82, 0.94, 0.90, 0.87],
        "F1-Score": [0.85, 0.90, 0.92, 0.90]
    }
    
    df_per_class = pd.DataFrame(per_class_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df_per_class, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(per_class_data["Class"]))
        width = 0.25
        
        ax.bar(x - width, per_class_data["Precision"], width, label="Precision", color="#FF6B6B")
        ax.bar(x, per_class_data["Recall"], width, label="Recall", color="#4ECDC4")
        ax.bar(x + width, per_class_data["F1-Score"], width, label="F1-Score", color="#45B7D1")
        
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Per-Class Performance Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(per_class_data["Class"])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)


# ============================================================================
# PAGE: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## Brain Tumor Classification using Deep Learning
    
    ### Project Overview
    This project develops an AI system for automated classification of brain tumors from MRI images.
    Using deep learning and transfer learning techniques, we achieved 90%+ accuracy in identifying
    four types of brain tumors.
    
    ### Problem Statement
    - RadiologyFirst Medical Center faces a 200+ MRI scan daily backlog
    - Current diagnosis time: 2-3 days per scan
    - Risk of missing urgent cases due to volume
    
    ### Solution
    - AI-powered automated classification system
    - Instant diagnosis (sub-second predictions)
    - Assists radiologists (doesn't replace them)
    - Prioritizes urgent cases
    
    ### Dataset
    - **Total Images**: 7,023 brain MRI scans
    - **Training Set**: 5,712 images (81%)
    - **Testing Set**: 1,311 images (19%)
    - **Classes**: 4 tumor types
      - Glioma (malignant brain tumor)
      - Meningioma (tumor in brain lining)
      - No Tumor (healthy brain scan)
      - Pituitary (pituitary gland tumor)
    - **Image Size**: 150x150 pixels
    
    ### Models Developed
    
    **Model 1: Baseline CNN**
    - Architecture: 4 convolutional blocks + classification head
    - Accuracy: 50%
    - Purpose: Establish baseline performance
    
    **Model 2: Enhanced VGG16**
    - Architecture: Pre-trained VGG16 + custom classification head
    - Accuracy: 90-95%
    - Purpose: Production-ready, high-accuracy system
    - Advanced Techniques: Transfer learning, two-stage training, data augmentation
    
    ### Business Impact
    - **Annual Labor Savings**: $912,450
    - **Additional Revenue**: $520,000
    - **ROI (5-year)**: 34,656%
    - **Payback Period**: 13 days
    
    ### Technology Stack
    - **Framework**: TensorFlow / Keras
    - **Language**: Python 3.10
    - **Libraries**: NumPy, OpenCV, scikit-learn, Pandas
    - **Dashboard**: Streamlit
    - **Visualization**: Matplotlib, Seaborn
    
    ### Key Features
    ✅ Cross-platform compatible (Windows, Linux, macOS)  
    ✅ No hardcoded paths (uses pathlib)  
    ✅ Comprehensive documentation  
    ✅ Production-ready code  
    ✅ Automated validation system  
    ✅ Interactive dashboard  
    
    ### Project Files
    - **Source Code**: 9 Python files (src/ folder)
    - **Documentation**: 13 markdown files (docs/ folder)
    - **Setup Files**: 4 files (setup_files/ folder)
    - **Presentation Materials**: 6 files (deployment/ folder)
    - **Models**: 4 trained models (models/ folder)
    - **Results**: 10+ reports and visualizations (outputs/ folder)
    
    ### How to Use
    1. **Setup**: Follow setup_files/QUICK_START.txt
    2. **Training**: Run brain_tumor_project/src/train_model.py (10 min)
    3. **Evaluation**: Run brain_tumor_project/src/evaluate.py (2 min)
    4. **Prediction**: Use brain_tumor_project/src/predict.py or this dashboard
    
    ### Important Notes
    ⚠️ **This system is for educational purposes only**  
    ⚠️ Not FDA-approved or validated for clinical use  
    ⚠️ Always requires human radiologist review  
    ⚠️ Should not replace professional medical diagnosis  
    
    ### Contact & Support
    For questions, issues, or suggestions, please refer to:
    - MASTER_DOCUMENTATION.md (complete guide)
    - brain_tumor_project/docs/ (technical documentation)
    - setup_files/SETUP_INSTRUCTIONS.txt (setup help)
    
    ---
    
    **Project Status**: ✅ Complete & Production-Ready  
    **Last Updated**: December 3, 2025  
    **Version**: 1.0.0
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 3rem;'>
    <p>🧠 Brain Tumor Classification Dashboard | Educational Purpose Only | Not for Clinical Use</p>
    <p>© 2025 Brain Tumor Classification Project | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
