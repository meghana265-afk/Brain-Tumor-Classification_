# EXECUTION ORDER FOR P2 DEMONSTRATION

## HOW TO RUN ALL FILES IN CORRECT ORDER

---

## **PHASE 1: BASELINE MODEL (Milestone 1)**

### **Step 1: Train Baseline Model**
⏱️ Time: ~10 minutes

```powershell
python brain_tumor_project\src\train_model.py
```

**What it does:**
- Trains custom CNN from scratch
- Creates baseline model with 3.6M parameters
- Expected accuracy: ~50%

**Output Files:**
- `saved_model.h5` (41.7 MB) - Final baseline model
- `best_model.h5` (55.3 MB) - Best checkpoint
- `accuracy_plot.png` - Training accuracy curves
- `loss_plot.png` - Training loss curves

---

### **Step 2: Evaluate Baseline Model**
⏱️ Time: ~2 minutes

```powershell
python brain_tumor_project\src\evaluate.py
```

**What it does:**
- Evaluates baseline model on test dataset
- Generates comprehensive metrics
- Creates confusion matrix

**Output Files:**
- `evaluation_report.txt` - Complete evaluation metrics
- `confusion_matrix.png` - Visual confusion matrix

---

## **PHASE 2: ENHANCED MODEL (Milestone 2 - P2)**

### **Step 3: Train Enhanced Model (VGG16 Transfer Learning)**
⏱️ Time: ~25-30 minutes

```powershell
python brain_tumor_project\src\train_model_enhanced.py
```

**What it does:**
- Uses VGG16 pre-trained on ImageNet
- Two-stage training (feature extraction + fine-tuning)
- Advanced data augmentation
- Class weight balancing
- Expected accuracy: 85-95%

**Output Files:**
- `best_enhanced_model.h5` (~60 MB) - Best enhanced checkpoint
- `enhanced_model.h5` (~60 MB) - Final enhanced model
- `training_log.csv` - Epoch-by-epoch training history
- `enhanced_accuracy_plot.png` - Enhanced training curves
- `enhanced_loss_plot.png` - Enhanced loss curves

---

### **Step 4: Evaluate Enhanced Model**
⏱️ Time: ~2 minutes

```powershell
python brain_tumor_project\src\evaluate_enhanced.py
```

**What it does:**
- Evaluates enhanced model on test dataset
- Compares with baseline model
- Shows improvement metrics

**Output Files:**
- `enhanced_evaluation_report.txt` - Enhanced evaluation metrics
- `enhanced_confusion_matrix.png` - Enhanced confusion matrix

---

### **Step 5: Compare Both Models Side-by-Side**
⏱️ Time: ~3 minutes

```powershell
python brain_tumor_project\src\compare_models.py
```

**What it does:**
- Detailed comparison of baseline vs enhanced
- Per-class performance analysis
- Improvement heatmaps
- Statistical significance tests

**Output Files:**
- `model_comparison_report.txt` - Detailed comparison report
- `model_comparison_comprehensive.png` - Visual comparison charts

---

## **PHASE 3: PREDICTION DEMONSTRATION**

### **Step 6a: Test Single Prediction (Baseline)**
⏱️ Time: ~5 seconds

```powershell
python brain_tumor_project\src\predict.py Testing\glioma\Te-glTr_0000.jpg
```

**What it does:**
- Predicts tumor type using baseline model
- Shows confidence scores for all classes

---

### **Step 6b: Test Single Prediction (Enhanced)**
⏱️ Time: ~5 seconds

```powershell
python brain_tumor_project\src\predict.py Testing\glioma\Te-glTr_0000.jpg --enhanced
```

**What it does:**
- Predicts tumor type using enhanced model
- Shows improved confidence scores

---

### **Step 6c: Test Side-by-Side Comparison**
⏱️ Time: ~10 seconds

```powershell
python brain_tumor_project\src\predict.py Testing\glioma\Te-glTr_0000.jpg --both
```

**What it does:**
- Runs both models simultaneously
- Shows side-by-side predictions
- Highlights differences in confidence

---

## **QUICK START (All Steps)**

```powershell
# Phase 1: Baseline
python brain_tumor_project\src\train_model.py
python brain_tumor_project\src\evaluate.py

# Phase 2: Enhanced (P2)
python brain_tumor_project\src\train_model_enhanced.py
python brain_tumor_project\src\evaluate_enhanced.py
python brain_tumor_project\src\compare_models.py

# Phase 3: Demo
python brain_tumor_project\src\predict.py Testing\glioma\Te-glTr_0000.jpg --both
```

---

## **TOTAL EXECUTION TIME**

- **Baseline Training:** 10 minutes
- **Baseline Evaluation:** 2 minutes
- **Enhanced Training:** 25-30 minutes
- **Enhanced Evaluation:** 2 minutes
- **Model Comparison:** 3 minutes
- **Predictions:** 1 minute

**GRAND TOTAL: ~45 minutes**

---

## **IMPORTANT NOTES**

1. **Run in order** - Enhanced model evaluation requires enhanced model to be trained first
2. **Compare models** requires both baseline and enhanced models to exist
3. **Predictions** can use either model or both
4. All scripts automatically create output directories if they don't exist
5. All files have comprehensive line-by-line comments
6. All scripts include progress indicators and status messages

---

## **SYSTEM REQUIREMENTS**

- Python 3.10.11
- TensorFlow 2.10.0
- 8GB+ RAM recommended
- GPU optional (speeds up training 5-10x)
- ~500MB disk space for models and outputs

---

## **TROUBLESHOOTING**

**If you get "Module not found" error:**
```powershell
pip install tensorflow numpy opencv-python matplotlib scikit-learn seaborn pandas
```

**If you get "Data directory not found" error:**
- Ensure `Training/` and `Testing/` folders are in the root directory
- Each should have 4 subfolders: glioma, meningioma, notumor, pituitary

**If enhanced model not found:**
- Run `train_model_enhanced.py` first (Step 3)
- It will create `best_enhanced_model.h5`

---

## **FILE STRUCTURE**

```
archive (2)/
├── brain_tumor_project/
│   ├── src/
│   │   ├── config.py
│   │   ├── train_model.py
│   │   ├── train_model_enhanced.py
│   │   ├── evaluate.py
│   │   ├── evaluate_enhanced.py
│   │   ├── compare_models.py
│   │   ├── predict.py
│   │   ├── preprocess.py
│   │   └── utils.py
│   └── [Documentation files]
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── [Model files and outputs]
```

---

**✅ ALL FILES TESTED AND WORKING!**
**✅ READY FOR P2 SUBMISSION!**
