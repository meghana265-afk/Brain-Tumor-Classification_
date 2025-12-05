"""
Brain Tumor Classification - Clean & Simple Dashboard
Easy to read with basic colors and clear typography

Run with: streamlit run dashboard_app/app_clean.py
"""

import streamlit as st
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import tempfile

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    st.error(f"TensorFlow not available: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 150
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_DISPLAY_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

DASHBOARD_DIR = Path(__file__).parent
APP_ROOT = DASHBOARD_DIR.parent
BRAIN_TUMOR_DIR = APP_ROOT / "brain_tumor_project"
MODEL_DIR = BRAIN_TUMOR_DIR / "models"

BASELINE_MODEL_PATH = MODEL_DIR / "saved_model.h5"
ENHANCED_MODEL_PATH = MODEL_DIR / "best_enhanced_model.h5"

# ============================================================================
# STREAMLIT CONFIG - CLEAN & SIMPLE
# ============================================================================

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean CSS
st.markdown("""
    <style>
        /* Basic clean styling */
        body {
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
            color: #333333;
        }
        
        h1 {
            color: #003d7a;
            border-bottom: 3px solid #003d7a;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        h2 {
            color: #004d99;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        
        h3 {
            color: #333333;
            margin-top: 15px;
        }
        
        .metric-box {
            background-color: #ffffff;
            border-left: 4px solid #003d7a;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .info-box {
            background-color: #e3f2fd;
            border-left: 4px solid #1976d2;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            line-height: 1.6;
        }
        
        .success-box {
            background-color: #e8f5e9;
            border-left: 4px solid #388e3c;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            line-height: 1.6;
        }
        
        .warning-box {
            background-color: #fff3e0;
            border-left: 4px solid #f57c00;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            line-height: 1.6;
        }
        
        .data-table {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🧠 Brain Tumor Classification")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page:",
    ["Home", "Dataset", "Models", "Prediction", "Results", "About"]
)

st.sidebar.markdown("---")
st.sidebar.write("""
**Quick Info:**
- 7,023 MRI images
- 4 tumor types
- 90% accuracy
- Real-time predictions
""")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_simple_brain_image(tumor_type="normal"):
    """Create a simple brain image."""
    img = Image.new('RGB', (150, 150), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw brain outline
    draw.ellipse([20, 20, 130, 130], outline='black', width=2, fill='lightgray')
    
    # Add tumor
    if tumor_type == "glioma":
        draw.ellipse([60, 50, 90, 80], fill='red')
    elif tumor_type == "meningioma":
        draw.ellipse([30, 40, 70, 80], fill='blue')
    elif tumor_type == "pituitary":
        draw.ellipse([65, 70, 85, 90], fill='orange')
    
    return img

@st.cache(allow_output_mutation=True)
def load_model_cached(model_path):
    """Load model with caching."""
    if not TF_AVAILABLE:
        return None
    
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    
    try:
        return load_model(str(model_path))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache(allow_output_mutation=True)
def get_dataset_stats():
    """Dataset statistics."""
    return {
        "Total Images": 7023,
        "Training Images": 5712,
        "Testing Images": 1311,
        "Classes": 4,
        "Class Distribution": {
            "Glioma": {"train": 1321, "test": 300},
            "Meningioma": {"train": 1339, "test": 306},
            "No Tumor": {"train": 1595, "test": 405},
            "Pituitary": {"train": 1457, "test": 300}
        }
    }

def preprocess_image(image_path):
    """Preprocess image."""
    try:
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, True
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, False

def predict_tumor(model, image_array):
    """Make prediction."""
    try:
        if model is None or image_array is None:
            return None
        
        predictions = model.predict(image_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        
        return {
            "class": CLASS_NAMES[predicted_idx],
            "display_name": CLASS_DISPLAY_NAMES[predicted_idx],
            "confidence": confidence,
            "all_probabilities": dict(zip(CLASS_DISPLAY_NAMES, predictions[0]))
        }
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "Home":
    st.title("🧠 Brain Tumor Classification System")
    
    st.markdown("""
    <div class='info-box'>
    <h3>Welcome to the Brain Tumor Classification Dashboard</h3>
    <p>This system uses deep learning to classify brain tumors from MRI images.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", "7,023")
    with col2:
        st.metric("Tumor Types", "4")
    with col3:
        st.metric("Accuracy", "90%")
    with col4:
        st.metric("Models", "2")
    
    st.markdown("---")
    
    st.markdown("""
    <div class='success-box'>
    <h3>Key Features</h3>
    <ul>
        <li>Upload MRI brain images</li>
        <li>Get instant predictions from 2 models</li>
        <li>View confidence scores</li>
        <li>Explore dataset statistics</li>
        <li>Compare model performance</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## Sample Brain Images")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Glioma (Red)**")
        img = create_simple_brain_image("glioma")
        st.image(img, use_column_width=True)
        st.write("Most common malignant tumor")
    
    with col2:
        st.markdown("**Meningioma (Blue)**")
        img = create_simple_brain_image("meningioma")
        st.image(img, use_column_width=True)
        st.write("Slow-growing tumor")
    
    with col3:
        st.markdown("**No Tumor (Gray)**")
        img = create_simple_brain_image("normal")
        st.image(img, use_column_width=True)
        st.write("Healthy brain scan")
    
    with col4:
        st.markdown("**Pituitary (Orange)**")
        img = create_simple_brain_image("pituitary")
        st.image(img, use_column_width=True)
        st.write("Hormonal gland tumor")
    
    st.markdown("---")
    
    st.markdown("""
    <div class='warning-box'>
    <h3>⚠️ Important Disclaimer</h3>
    <p>This system is for educational purposes only. It is NOT approved for clinical diagnosis. 
    Always consult medical professionals for proper diagnosis and treatment.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DATASET
# ============================================================================

elif page == "Dataset":
    st.title("📊 Dataset Information")
    
    stats = get_dataset_stats()
    
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
    st.markdown("## Class Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Training Set")
        train_data = {
            'Class': list(stats['Class Distribution'].keys()),
            'Count': [stats['Class Distribution'][c]['train'] for c in stats['Class Distribution'].keys()]
        }
        train_df = pd.DataFrame(train_data)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#ff4444', '#4444ff', '#44ff44', '#ffaa44']
        ax.bar(train_df['Class'], train_df['Count'], color=colors)
        ax.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
        ax.set_title('Training Set', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Testing Set")
        test_data = {
            'Class': list(stats['Class Distribution'].keys()),
            'Count': [stats['Class Distribution'][c]['test'] for c in stats['Class Distribution'].keys()]
        }
        test_df = pd.DataFrame(test_data)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#ff4444', '#4444ff', '#44ff44', '#ffaa44']
        ax.bar(test_df['Class'], test_df['Count'], color=colors)
        ax.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
        ax.set_title('Testing Set', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("## Detailed Statistics")
    
    details = []
    for class_name, dist in stats['Class Distribution'].items():
        details.append({
            'Tumor Type': class_name,
            'Training': dist['train'],
            'Testing': dist['test'],
            'Total': dist['train'] + dist['test']
        })
    
    details_df = pd.DataFrame(details)
    st.dataframe(details_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: MODELS
# ============================================================================

elif page == "Models":
    st.title("🤖 Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <h3>Model 1: Baseline CNN</h3>
        <p><strong>Architecture:</strong></p>
        <ul>
            <li>Conv2D layers (32, 64, 128 filters)</li>
            <li>MaxPooling layers</li>
            <li>Dense layers with Dropout</li>
        </ul>
        <p><strong>Performance:</strong></p>
        <ul>
            <li>Accuracy: 50%</li>
            <li>Fast training</li>
            <li>Quick predictions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='success-box'>
        <h3>Model 2: Enhanced VGG16</h3>
        <p><strong>Architecture:</strong></p>
        <ul>
            <li>Transfer Learning (VGG16)</li>
            <li>Pre-trained ImageNet weights</li>
            <li>Custom fine-tuning</li>
        </ul>
        <p><strong>Performance:</strong></p>
        <ul>
            <li>Accuracy: 90%</li>
            <li>Better predictions</li>
            <li>Production-ready</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## Performance Comparison")
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Baseline CNN': ['50%', '48%', '50%', '49%'],
        'Enhanced VGG16': ['90%', '89%', '90%', '89%']
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['Baseline CNN', 'Enhanced VGG16']
    accuracy = [0.50, 0.90]
    precision = [0.48, 0.89]
    recall = [0.50, 0.90]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, accuracy, width, label='Accuracy', color='#ff4444')
    ax.bar(x, precision, width, label='Precision', color='#4444ff')
    ax.bar(x + width, recall, width, label='Recall', color='#44ff44')
    
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    st.pyplot(fig)

# ============================================================================
# PAGE: PREDICTION
# ============================================================================

elif page == "Prediction":
    st.title("🔍 Tumor Classification")
    
    st.markdown("""
    <div class='info-box'>
    <h3>Upload and Predict</h3>
    <p>Upload a brain MRI image to get instant predictions from both models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image (JPG, PNG)",
            type=['jpg', 'jpeg', 'png']
        )
    
    with col2:
        st.markdown("### Select Model")
        model_choice = st.radio(
            "Choose model:",
            ["Enhanced VGG16", "Baseline CNN"]
        )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        st.markdown("---")
        st.markdown("### Uploaded Image")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_img = Image.open(uploaded_file)
            st.image(uploaded_img, use_column_width=True)
        
        img_array, success = preprocess_image(tmp_path)
        
        if success:
            st.markdown("---")
            st.markdown("### Predictions")
            
            if model_choice == "Enhanced VGG16":
                model = load_model_cached(ENHANCED_MODEL_PATH)
                model_name = "Enhanced VGG16"
            else:
                model = load_model_cached(BASELINE_MODEL_PATH)
                model_name = "Baseline CNN"
            
            if model is not None:
                result = predict_tumor(model, img_array)
                
                if result is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='metric-box'>
                        <h3>Predicted Class</h3>
                        <p style='font-size: 24px; font-weight: bold; color: #003d7a;'>
                        {result['display_name']}
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='metric-box'>
                        <h3>Confidence</h3>
                        <p style='font-size: 24px; font-weight: bold; color: #003d7a;'>
                        {result['confidence']*100:.1f}%
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### Confidence Scores (All Classes)")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    probs = [result['all_probabilities'][name] for name in CLASS_DISPLAY_NAMES]
                    colors = ['#ff4444', '#4444ff', '#44ff44', '#ffaa44']
                    
                    bars = ax.barh(CLASS_DISPLAY_NAMES, probs, color=colors)
                    ax.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
                    ax.set_title(f'Prediction Scores ({model_name})', fontsize=12, fontweight='bold')
                    ax.set_xlim([0, 1])
                    
                    for i, (bar, prob) in enumerate(zip(bars, probs)):
                        ax.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center', fontweight='bold')
                    
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    
                    st.markdown("""
                    <div class='warning-box'>
                    <h3>Important Notes</h3>
                    <ul>
                        <li>This is an AI prediction for educational purposes only</li>
                        <li>Do NOT use for clinical diagnosis</li>
                        <li>Always consult medical professionals</li>
                        <li>Consider confidence score in context</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Could not load model")
        
        import os
        try:
            os.remove(tmp_path)
        except:
            pass
    
    else:
        st.info("👆 Upload an image to get predictions")

# ============================================================================
# PAGE: RESULTS
# ============================================================================

elif page == "Results":
    st.title("📈 Model Results")
    
    st.markdown("""
    <div class='success-box'>
    <h3>Performance Metrics</h3>
    <p>Detailed analysis of both models on the test set.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Overall Performance")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['Baseline CNN', 'Enhanced VGG16']
    accuracy = [0.50, 0.90]
    precision = [0.48, 0.89]
    recall = [0.50, 0.90]
    f1 = [0.49, 0.89]
    
    x = np.arange(len(models))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#ff4444')
    ax.bar(x - 0.5*width, precision, width, label='Precision', color='#4444ff')
    ax.bar(x + 0.5*width, recall, width, label='Recall', color='#44ff44')
    ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='#ffaa44')
    
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Model Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("## Per-Class Performance")
    
    per_class_data = {
        'Class': CLASS_DISPLAY_NAMES,
        'Precision': [0.88, 0.91, 0.87, 0.89],
        'Recall': [0.90, 0.89, 0.91, 0.89],
        'F1-Score': [0.89, 0.90, 0.89, 0.89]
    }
    
    per_class_df = pd.DataFrame(per_class_data)
    st.dataframe(per_class_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: ABOUT
# ============================================================================

elif page == "About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    <div class='info-box'>
    <h3>Brain Tumor Classification System</h3>
    <p>This is an educational project demonstrating the application of deep learning 
    for medical image analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='success-box'>
        <h3>Project Goals</h3>
        <ul>
            <li>Develop accurate classification models</li>
            <li>Compare different approaches</li>
            <li>Demonstrate AI in healthcare</li>
            <li>Educate on medical imaging</li>
            <li>Create an intuitive interface</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='success-box'>
        <h3>Dataset Information</h3>
        <ul>
            <li><strong>Total:</strong> 7,023 images</li>
            <li><strong>Training:</strong> 5,712 (81%)</li>
            <li><strong>Testing:</strong> 1,311 (19%)</li>
            <li><strong>Classes:</strong> 4 tumor types</li>
            <li><strong>Size:</strong> 150x150 pixels</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='info-box'>
    <h3>Technology Stack</h3>
    <ul>
        <li>Python 3.10</li>
        <li>TensorFlow / Keras</li>
        <li>Streamlit</li>
        <li>NumPy, Pandas, Scikit-learn</li>
        <li>Matplotlib, Seaborn</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='warning-box'>
    <h3>⚠️ Important Disclaimer</h3>
    <p><strong>This system is for educational purposes only.</strong></p>
    <ul>
        <li>NOT approved for clinical diagnosis</li>
        <li>NOT a replacement for medical professionals</li>
        <li>Results should NOT be used for medical decisions</li>
        <li>Always consult qualified doctors</li>
        <li>Model accuracy varies by tumor type</li>
        <li>Image quality affects predictions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", "7,023")
    with col2:
        st.metric("Accuracy", "90%")
    with col3:
        st.metric("Models", "2")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px; font-size: 12px;'>
    <p><strong>Brain Tumor Classification System</strong></p>
    <p>Educational Dashboard | Not for Clinical Use | Always Consult Medical Professionals</p>
    <p>© 2024 Brain Tumor Classification Project</p>
</div>
""", unsafe_allow_html=True)
