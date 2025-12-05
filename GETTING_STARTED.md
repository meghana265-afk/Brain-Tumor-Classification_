# Brain Tumor Classification - Complete Getting Started Guide

## Table of Contents
1. [Clone Repository](#clone-repository)
2. [Setup Environment](#setup-environment)
3. [Run Each File](#run-each-file)
4. [Dashboard](#dashboard)
5. [Training Models](#training-models)
6. [Making Predictions](#making-predictions)
7. [Troubleshooting](#troubleshooting)

---

## Clone Repository

### Step 1: Clone from GitHub

```bash
# Using HTTPS (recommended for beginners)
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git

# Navigate to project
cd Brain-Tumor-Classification_
```

### Step 2: Verify Contents

```bash
# List all files
ls -la

# Check Python files
ls brain_tumor_project/src/
```

**Expected structure:**
```
Brain-Tumor-Classification_/
├── brain_tumor_project/
│   ├── src/                    # Source code (10 files)
│   ├── deployment/             # Presentation files
│   ├── docs/                   # Documentation
│   └── outputs/                # Visualizations
├── dashboard_app/              # Streamlit dashboard
├── DOCS/                       # Comprehensive guides
├── README.md                   # Project overview
└── COMPLETE_GUIDE.txt          # All instructions
```

---

## Setup Environment

### Step 1: Create Virtual Environment

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install tensorflow==2.10.0
pip install numpy==1.23.5
pip install scikit-learn
pip install matplotlib
pip install pillow
pip install opencv-python
pip install streamlit==1.28.1
pip install pandas
```

Or use one command:

```bash
pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas
```

### Step 3: Verify Installation

```bash
# Test imports
python -c "import tensorflow; import numpy; print('✅ All packages installed!')"
```

---

## Run Each File

### File Structure & Purpose

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `config.py` | Configuration & paths | None | Settings |
| `preprocess.py` | Image preprocessing | Raw images | Processed data |
| `train_model.py` | Train baseline CNN | Images folder | saved_model.h5 |
| `train_model_enhanced.py` | Train VGG16 model | Images folder | best_enhanced_model.h5 |
| `evaluate.py` | Evaluate baseline model | Images folder | Metrics & plots |
| `evaluate_enhanced.py` | Evaluate VGG16 model | Images folder | Metrics & plots |
| `predict.py` | Make predictions | Image path | Tumor prediction |
| `compare_models.py` | Compare both models | Both models | Comparison report |
| `utils.py` | Helper functions | Various | Used by other files |

---

### 1. Config File (Configuration)

**Purpose:** Define paths, model settings, and hyperparameters

```powershell
cd brain_tumor_project/src
python config.py
```

**What it does:**
- Loads configuration settings
- Sets up paths for data and models
- Defines model parameters

**Output:** None (just configuration setup)

**Code preview:**
```python
# config.py
import os
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, '../../Training')  # Training data
TEST_PATH = os.path.join(BASE_PATH, '../../Testing')   # Test data
MODEL_PATH = os.path.join(BASE_PATH, '../../models')   # Save models
```

---

### 2. Preprocess File (Data Preparation)

**Purpose:** Prepare and normalize images for training

```powershell
python preprocess.py
```

**What it does:**
- Loads images from Training/ folder
- Resizes to 224x224 pixels
- Normalizes pixel values (0-1)
- Creates train/val split
- Saves processed data

**Input Required:**
- `Training/glioma/` - Glioma tumor images
- `Training/meningioma/` - Meningioma tumor images
- `Training/notumor/` - No tumor images
- `Training/pituitary/` - Pituitary tumor images

**Output:**
```
X_train.npy     # Training images array
y_train.npy     # Training labels array
X_val.npy       # Validation images array
y_val.npy       # Validation labels array
```

---

### 3. Train Baseline Model

**Purpose:** Train a CNN model from scratch

```powershell
python train_model.py
```

**What it does:**
- Loads preprocessed data
- Creates CNN architecture (3 conv layers)
- Trains on GPU/CPU
- Shows progress (epochs, loss, accuracy)
- Saves best model

**Training Output:**
```
Epoch 1/50
123/123 [==============================] - 45s 367ms/step - loss: 1.2345 - accuracy: 0.5234
Epoch 2/50
123/123 [==============================] - 43s 350ms/step - loss: 0.8765 - accuracy: 0.7123
...
✅ Model saved: saved_model.h5
```

**Expected Results:**
- Training accuracy: ~75-80%
- Validation accuracy: ~70-75%
- Test accuracy: **76.89%**

---

### 4. Train Enhanced Model (VGG16)

**Purpose:** Train transfer learning model with pre-trained VGG16

```powershell
python train_model_enhanced.py
```

**What it does:**
- Loads VGG16 (pre-trained on ImageNet)
- Adds custom layers for tumor classification
- Fine-tunes weights
- Trains with early stopping
- Saves best model

**Output:**
```
Epoch 1/100
123/123 [==============================] - 28s 228ms/step - loss: 2.1234 - accuracy: 0.3456
...
✅ Model saved: best_enhanced_model.h5
Early stopping triggered!
```

**Expected Results:**
- Training accuracy: ~90%
- Validation accuracy: ~88%
- Test accuracy: **86.19%** ⭐ (Best model)

---

### 5. Evaluate Baseline Model

**Purpose:** Test and analyze baseline CNN performance

```powershell
python evaluate.py
```

**What it does:**
- Loads saved_model.h5
- Tests on Testing/ folder
- Calculates metrics (accuracy, precision, recall, F1)
- Generates confusion matrix
- Creates visualizations

**Outputs:**
```
accuracy_plot.png              # Accuracy vs epochs
loss_plot.png                  # Loss vs epochs
confusion_matrix.png           # Prediction matrix
classification_report.txt      # Detailed metrics
```

**Expected Output:**
```
=== BASELINE MODEL EVALUATION ===
Accuracy: 76.89%
Precision: 0.77
Recall: 0.77
F1-Score: 0.77

Confusion Matrix saved: confusion_matrix.png
Classification report saved: classification_report.txt
```

---

### 6. Evaluate Enhanced Model

**Purpose:** Test VGG16 model performance

```powershell
python evaluate_enhanced.py
```

**What it does:**
- Loads best_enhanced_model.h5
- Tests on Testing/ folder
- Calculates detailed metrics
- Creates comparison plots
- Generates evaluation report

**Outputs:**
```
enhanced_model_evaluation.png   # Performance visualization
enhanced_evaluation_report.txt  # Metrics summary
```

**Expected Output:**
```
=== ENHANCED MODEL EVALUATION ===
Accuracy: 86.19%
Precision: 0.86
Recall: 0.86
F1-Score: 0.86

✅ Enhanced model outperforms baseline by 9.3%
```

---

### 7. Make Predictions

**Purpose:** Use trained model to predict on new images

```powershell
# Automatic (finds test images)
python predict.py

# Or with specific image
python predict.py "path/to/your/image.jpg"
```

**What it does:**
- Loads best_enhanced_model.h5
- Takes image input
- Preprocesses image
- Makes prediction
- Shows confidence score

**Example Output:**
```
Loading model: best_enhanced_model.h5
Processing image: test_image.jpg
Prediction: Glioma (Confidence: 94.5%)

Class Probabilities:
  - Glioma: 94.5%
  - Meningioma: 3.2%
  - No Tumor: 1.8%
  - Pituitary: 0.5%
```

---

### 8. Compare Models

**Purpose:** Side-by-side comparison of both models

```powershell
python compare_models.py
```

**What it does:**
- Loads both models
- Tests on same data
- Calculates metrics for each
- Creates comparison visualizations
- Generates comparison report

**Outputs:**
```
model_comparison_report.txt         # Comparison summary
model_comparison_comprehensive.png  # Visual comparison
```

**Expected Output:**
```
=== MODEL COMPARISON ===

Baseline CNN:
  - Accuracy: 76.89%
  - Training time: 2.5 hours
  - Model size: 55.31 MB

Enhanced VGG16:
  - Accuracy: 86.19% ⭐ BEST
  - Training time: 1.8 hours
  - Model size: 60.80 MB

Improvement: +9.3% accuracy
```

---

## Dashboard

### Interactive Web Interface

**Purpose:** Real-time predictions and model exploration

### Step 1: Run Dashboard

```powershell
cd dashboard_app

# Option 1: Run directly
streamlit run app_clean.py

# Option 2: From root directory
streamlit run dashboard_app/app_clean.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

  Press CTRL+C to stop
```

### Step 2: Access Dashboard

Open browser → `http://localhost:8501`

### Dashboard Features

**Home Page:**
- Project overview
- Key statistics
- Model performance summary

**Dataset Page:**
- Image counts per class
- Sample images
- Dataset distribution

**Models Page:**
- Model architecture details
- Training history
- Performance metrics

**Prediction Page:**
- Upload image
- Real-time prediction
- Confidence scores
- Class probabilities

**Results Page:**
- Model comparison
- Accuracy metrics
- Confusion matrices

**About Page:**
- Project description
- Team information
- References

### Using the Dashboard

1. **Navigate tabs** at top
2. **Upload an MRI image** on Prediction page
3. **Get instant prediction** with confidence
4. **Compare models** on Results page
5. **View metrics** on Models page

---

## Training Models

### Complete Training Workflow

### Step 1: Prepare Data

```powershell
cd brain_tumor_project/src

# 1. Run preprocessing
python preprocess.py
# Output: X_train.npy, y_train.npy, X_val.npy, y_val.npy
```

### Step 2: Train Both Models

```powershell
# Train baseline model
python train_model.py
# Output: saved_model.h5

# Train enhanced model
python train_model_enhanced.py
# Output: best_enhanced_model.h5
```

### Step 3: Evaluate Both

```powershell
# Evaluate baseline
python evaluate.py
# Outputs: confusion_matrix.png, classification_report.txt, etc.

# Evaluate enhanced
python evaluate_enhanced.py
# Outputs: enhanced_model_evaluation.png, enhanced_evaluation_report.txt
```

### Step 4: Compare Results

```powershell
python compare_models.py
# Output: model_comparison_comprehensive.png
```

### Full Training Command (Sequential)

```powershell
cd brain_tumor_project/src
python preprocess.py && python train_model.py && python train_model_enhanced.py && python evaluate.py && python evaluate_enhanced.py && python compare_models.py
```

**Expected Time:**
- Preprocessing: 5-10 minutes
- Baseline training: 2.5 hours
- Enhanced training: 1.8 hours
- Evaluation: 15 minutes
- **Total: ~4-5 hours**

---

## Making Predictions

### Method 1: Python Script

```powershell
cd brain_tumor_project/src

# Predict on single image
python predict.py "C:\path\to\tumor_image.jpg"

# Auto-find test images
python predict.py
```

### Method 2: Dashboard

```powershell
streamlit run dashboard_app/app_clean.py
```

Go to "Prediction" tab → Upload image → Get result

### Method 3: Python Console

```python
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model('brain_tumor_project/models/best_enhanced_model.h5')

# Load and preprocess image
img = Image.open('test_image.jpg')
img = img.resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
predicted_class = classes[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Prediction: {predicted_class} ({confidence:.1f}% confidence)")
```

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```powershell
# Reinstall all packages
pip install --upgrade pip
pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas
```

### Issue: Out of Memory (OOM) errors

**Solution:**
```python
# In train_model.py or train_model_enhanced.py
# Reduce batch size
BATCH_SIZE = 16  # Was 32
EPOCHS = 25      # Was 50
```

### Issue: Data not found

**Solution:**
- Verify `Training/` and `Testing/` folders exist in project root
- Check folder structure:
  ```
  Training/
  ├── glioma/
  ├── meningioma/
  ├── notumor/
  └── pituitary/
  ```

### Issue: GPU not detected

**Solution:**
```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU, it will run on CPU (slower but works)
```

### Issue: Streamlit port already in use

**Solution:**
```powershell
# Use different port
streamlit run app_clean.py --server.port 8502
```

### Issue: Model files not found

**Solution:**
1. First train the models:
   ```powershell
   python brain_tumor_project/src/train_model.py
   python brain_tumor_project/src/train_model_enhanced.py
   ```

2. Or download pre-trained models from documentation

---

## Quick Command Reference

### Setup

```powershell
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git
cd Brain-Tumor-Classification_
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas
```

### Training

```powershell
cd brain_tumor_project/src
python preprocess.py
python train_model.py
python train_model_enhanced.py
python evaluate.py
python evaluate_enhanced.py
python compare_models.py
```

### Dashboard

```powershell
streamlit run dashboard_app/app_clean.py
```

### Prediction

```powershell
cd brain_tumor_project/src
python predict.py
```

---

## File Execution Summary

| Step | File | Command | Time | Output |
|------|------|---------|------|--------|
| 1 | preprocess.py | `python preprocess.py` | 5-10m | .npy files |
| 2 | train_model.py | `python train_model.py` | 2.5h | saved_model.h5 |
| 3 | train_model_enhanced.py | `python train_model_enhanced.py` | 1.8h | best_enhanced_model.h5 |
| 4 | evaluate.py | `python evaluate.py` | 5m | plots & metrics |
| 5 | evaluate_enhanced.py | `python evaluate_enhanced.py` | 5m | plots & metrics |
| 6 | compare_models.py | `python compare_models.py` | 2m | comparison report |
| 7 | predict.py | `python predict.py` | 1m | prediction result |
| 8 | app_clean.py | `streamlit run ...` | ongoing | web interface |

---

## Next Steps

1. ✅ Clone repository
2. ✅ Setup environment
3. ✅ Run files in order
4. ✅ Access dashboard
5. ✅ Make predictions
6. ✅ Explore results

**Happy exploring!** 🚀

---

**Repository:** https://github.com/meghana265-afk/Brain-Tumor-Classification_
**Questions?** Check COMPLETE_GUIDE.txt or README.md
