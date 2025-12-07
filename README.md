# 🧠 Brain Tumor Classification System
## Deep Learning Project with CNN & Transfer Learning

**Status**: ✅ Complete & Production-Ready  
**Models**: 2 trained (Baseline CNN + VGG16 Enhanced)  
**Accuracy**: 76.89% (baseline) → **86.19%** (enhanced) ⭐  
**Dashboard**: Streamlit Web Interface  
**GitHub**: https://github.com/meghana265-afk/Brain-Tumor-Classification_

---

## ⚡ QUICKEST START (30 seconds)

### Choose One Method:

---

## ▶️ ONE-COMMAND RUN (Dashboard)

**Windows (PowerShell):**
```powershell
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git; cd Brain-Tumor-Classification_; python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas; streamlit run dashboard_app/app_clean.py
```

**macOS/Linux:**
```bash
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git && cd Brain-Tumor-Classification_ && python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas && streamlit run dashboard_app/app_clean.py
```

**Data reminder:** ensure `Training/` and `Testing/` folders with class subfolders (`glioma/ meningioma/ notumor/ pituitary/`) and images exist if you plan to evaluate or predict. If these folders are missing, `evaluate.py` and `predict.py` will exit with a clear error.

**Model reminder:** place pretrained weights in `brain_tumor_project/models/` as `saved_model.h5` (baseline) and optionally `best_enhanced_model.h5` (enhanced). If models are absent, training scripts must be run before evaluation or prediction.

## 📈 ONE-COMMAND METRICS (Baseline)

Requires data and `saved_model.h5` in `brain_tumor_project/models/`.

**Windows (PowerShell):**
```powershell
cd Brain-Tumor-Classification_ ; .\.venv\Scripts\Activate.ps1 ; .\.venv\Scripts\python.exe brain_tumor_project\src\evaluate.py
```

**macOS/Linux:**
```bash
cd Brain-Tumor-Classification_ && source .venv/bin/activate && .venv/bin/python brain_tumor_project/src/evaluate.py
```

Outputs to terminal plus files: `brain_tumor_project/outputs/confusion_matrix.png` and `brain_tumor_project/models/evaluation_report.txt`.

---

## 🔽 OPTION 1: Download as ZIP File (Easiest - No Git Required)

### Step 1: Download ZIP
1. Go to: https://github.com/meghana265-afk/Brain-Tumor-Classification_
2. Click **Code** (green button) → **Download ZIP**
3. Extract the ZIP file to your desired location
4. Open terminal/PowerShell in extracted folder

### Step 2: Setup (Automatic)

**Windows (PowerShell):**
```powershell
SETUP.bat
```

**Mac/Linux (Terminal):**
```bash
chmod +x SETUP.sh
./SETUP.sh
```

### Step 3: Run Dashboard
```bash
streamlit run dashboard_app/app_clean.py
```

### Advantages of ZIP:
✅ No Git installation needed  
✅ Single download  
✅ Works offline after download  
✅ Easiest for beginners  

---

## 💻 OPTION 2: Git Clone (For Version Control)

### Step 1: Clone Repository

**Windows (PowerShell):**
```powershell
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git
cd Brain-Tumor-Classification_
SETUP.bat
```

**Mac/Linux (Terminal):**
```bash
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git
cd Brain-Tumor-Classification_
chmod +x SETUP.sh
./SETUP.sh
```

### Step 2: Run Dashboard
```bash
streamlit run dashboard_app/app_clean.py
```

---

## 📊 Comparison: ZIP vs Git Clone

| Feature | ZIP Download | Git Clone |
|---------|--------------|-----------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐⭐ Easy |
| **Setup Time** | 2 minutes | 2 minutes |
| **Git Required** | ❌ No | ✅ Yes |
| **Update Project** | Manual re-download | `git pull` |
| **File Size** | ~15 MB | ~15 MB |
| **For Beginners** | ✅ Recommended | ✓ Also good |
| **For Developers** | ✓ Works | ✅ Better |
| **Version Control** | ❌ No | ✅ Yes |

**👉 Recommendation:** 
- **New to coding?** → Use **ZIP** (Option 1)
- **Know Git?** → Use **Git Clone** (Option 2)
- **Want to contribute?** → Use **Git Clone** (Option 2)

---

## ⚠️ Important Notes

### If using ZIP:
- Don't re-extract over existing folder
- Extract to a clean location
- Keep extracted folder for future use

### If using Git Clone:
- Need Git installed ([Download here](https://git-scm.com))
- Can update easily with `git pull`
- Better for version tracking

### Step 1: Get the Project (Choose Option 1 or 2 Above)

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Mac/Linux (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas
```

**Verify installation:**
```bash
python -c "import tensorflow; import numpy; print('✅ Ready to go!')"
```

### Step 4: Prepare Data (Optional)

If you have Training/ and Testing/ folders with MRI images:

```bash
cd brain_tumor_project/src
python preprocess.py
```

**Input folders needed:**
```
Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

### Step 5: Train Models (Optional - takes 4-5 hours)

```bash
cd brain_tumor_project/src

# Train baseline CNN (2.5 hours)
python train_model.py

# Train VGG16 enhanced model (1.8 hours)
python train_model_enhanced.py
```

### Step 6: Evaluate Models (Optional)

```bash
# Evaluate baseline
python evaluate.py

# Evaluate enhanced
python evaluate_enhanced.py

# Compare both
python compare_models.py
```

### Step 7: Make Predictions

```bash
# From project root
cd brain_tumor_project/src

# Predict on image
python predict.py your_image.jpg
```

### Step 8: Run Dashboard (Recommended ⭐)

```bash
# From project root
streamlit run dashboard_app/app_clean.py
```

**Then open in browser:** `http://localhost:8501`

---

## 🎯 Dashboard Features

Access the web interface at **http://localhost:8501**

### Pages Available:
- **Home** - Project overview & statistics
- **Dataset** - Data distribution & sample images
- **Models** - Architecture details & performance
- **Prediction** - Upload image → Get instant prediction
- **Results** - Model comparison & metrics
- **About** - Project information

### How to Use:
1. Open http://localhost:8501
2. Go to "Prediction" tab
3. Upload an MRI image
4. Get instant classification with confidence score
5. View class probabilities

---

## 📊 Model Details

### Baseline CNN
- **Architecture**: 3 Convolutional layers + Dense layers
- **Accuracy**: 76.89%
- **Precision**: 0.77
- **Recall**: 0.77
- **F1-Score**: 0.77
- **File**: `saved_model.h5` (55.31 MB)
- **Training Time**: 2.5 hours

### Enhanced VGG16 ⭐ (BEST)
- **Architecture**: VGG16 + Custom Dense layers
- **Accuracy**: **86.19%**
- **Precision**: 0.86
- **Recall**: 0.86
- **F1-Score**: 0.86
- **File**: `best_enhanced_model.h5` (60.80 MB)
- **Training Time**: 1.8 hours
- **Improvement**: +9.3% over baseline

---

## 📁 What Each File Does

| File | Purpose | Command | Time |
|------|---------|---------|------|
| **preprocess.py** | Prepare images | `python preprocess.py` | 5-10m |
| **train_model.py** | Train baseline CNN | `python train_model.py` | 2.5h |
| **train_model_enhanced.py** | Train VGG16 | `python train_model_enhanced.py` | 1.8h |
| **evaluate.py** | Test baseline | `python evaluate.py` | 5m |
| **evaluate_enhanced.py** | Test VGG16 | `python evaluate_enhanced.py` | 5m |
| **compare_models.py** | Compare both | `python compare_models.py` | 2m |
| **predict.py** | Predict on images | `python predict.py image.jpg` | 1m |
| **app_clean.py** | Web dashboard | `streamlit run app_clean.py` | ∞ |

---

## 🚀 Quick Commands

### Quick Dashboard
```bash
streamlit run dashboard_app/app_clean.py
```

### Quick Prediction
```bash
cd brain_tumor_project/src
python predict.py
```

### Full Training Pipeline
```bash
cd brain_tumor_project/src
python preprocess.py && python train_model.py && python train_model_enhanced.py && python evaluate.py && python evaluate_enhanced.py && python compare_models.py
```

### Update Code from GitHub
```bash
git pull origin main
```

---

## 🔧 Troubleshooting

### Issue: "Module not found"
```bash
pip install --upgrade pip
pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas
```

### Issue: Out of Memory
Reduce batch size in `train_model.py`:
```python
BATCH_SIZE = 16  # was 32
EPOCHS = 25      # was 50
```

### Issue: Data not found
Ensure folders exist:
```
Training/glioma/, Training/meningioma/, Training/notumor/, Training/pituitary/
Testing/glioma/, Testing/meningioma/, Testing/notumor/, Testing/pituitary/
```

### Issue: Port 8501 already in use
```bash
streamlit run dashboard_app/app_clean.py --server.port 8502
```

### Issue: GPU not detected
TensorFlow will use CPU automatically (slower but works fine)

---

## 📚 Documentation

- **QUICKSTART.md** - 30-second setup
- **GETTING_STARTED.md** - Complete file-by-file guide
- **PROFESSOR_SETUP.md** - For new machines
- **DOCUMENTATION_INDEX.md** - Master index
- **COMPLETE_GUIDE.txt** - All commands

---

## 🎯 Most Common Workflows

### I just want to see predictions
```bash
streamlit run dashboard_app/app_clean.py
```
→ Upload image at http://localhost:8501

### I want to train my own models
```bash
cd brain_tumor_project/src
python preprocess.py
python train_model.py
python train_model_enhanced.py
```

### I want to evaluate models
```bash
cd brain_tumor_project/src
python evaluate.py
python evaluate_enhanced.py
python compare_models.py
```

### I want to predict on specific image
```bash
cd brain_tumor_project/src
python predict.py /path/to/image.jpg
```

---

## 🏥 Tumor Classes

The model classifies MRI images into 4 categories:

1. **Glioma** - Most common brain tumor
2. **Meningioma** - Tumor of the membrane surrounding brain
3. **Pituitary** - Tumor of pituitary gland
4. **No Tumor** - Healthy MRI scan

---

## 💻 System Requirements

- **Python**: 3.7+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 50GB for datasets
- **GPU**: Optional (runs on CPU)
- **OS**: Windows, Mac, or Linux

---

## 🌐 Repository Structure

```
Brain-Tumor-Classification/
├── brain_tumor_project/
│   ├── src/                    (10 Python files)
│   ├── models/                 (Trained models - local only)
│   ├── outputs/                (Visualizations)
│   └── docs/                   (Documentation)
├── dashboard_app/
│   ├── app_clean.py           (Streamlit app)
│   └── requirements.txt
├── DOCS/                       (8 comprehensive guides)
├── SETUP.bat                   (Windows auto setup)
├── SETUP.sh                    (Mac/Linux auto setup)
├── QUICKSTART.md               (30-second setup)
├── GETTING_STARTED.md          (Full instructions)
├── PROFESSOR_SETUP.md          (Clone-to-run guide)
└── README.md                   (This file)
```

---

## ✅ Verification Checklist

After setup, verify:
- [ ] Python 3.7+ installed
- [ ] Virtual environment activated
- [ ] All packages installed (`pip list | grep tensorflow`)
- [ ] Can import: `python -c "import tensorflow, numpy"`
- [ ] Dashboard runs: `streamlit run dashboard_app/app_clean.py`
- [ ] Can access: http://localhost:8501

---

## 🚀 Next Steps

1. **Run dashboard**: `streamlit run dashboard_app/app_clean.py`
2. **Upload MRI image**: Use Prediction tab
3. **See results**: View prediction & confidence
4. **Compare models**: Check Results tab
5. **Train custom**: Follow GETTING_STARTED.md

---

## 📞 Support

- **Setup issues?** Read QUICKSTART.md or PROFESSOR_SETUP.md
- **File questions?** See GETTING_STARTED.md
- **Need all commands?** Check COMPLETE_GUIDE.txt
- **Lost?** Read DOCUMENTATION_INDEX.md

---

**Last Updated**: December 6, 2025  
**Repository**: https://github.com/meghana265-afk/Brain-Tumor-Classification  
**Status**: ✅ Production Ready

---

## 📁 Project Structure

```
📦 Brain Tumor Classification
├── 🧠 brain_tumor_project/          (Main project)
│   ├── src/                          (Source code - all commented)
│   │   ├── train_model.py           (CNN training)
│   │   ├── train_model_enhanced.py  (VGG16 transfer learning)
│   │   ├── evaluate.py              (Baseline evaluation)
│   │   ├── evaluate_enhanced.py     (Enhanced evaluation)
│   │   ├── predict.py               (Single image prediction)
│   │   ├── compare_models.py        (Side-by-side comparison)
│   │   ├── config.py                (Configuration)
│   │   ├── utils.py                 (Utilities)
│   │   └── preprocess.py            (Image preprocessing)
│   ├── models/                       (Trained models)
│   │   ├── saved_model.h5           (Baseline model - 50% accuracy)
│   │   └── best_enhanced_model.h5   (VGG16 model - 90% accuracy)
│   ├── outputs/                      (Visualizations)
│   └── data/                         (Dataset if saved)
│
├── 📊 dashboard_app/                 (Streamlit web interface)
│   └── app_clean.py                 (Dashboard - 6 pages)
│
├── 📚 DOCS/                          (Comprehensive documentation)
│   ├── DOCUMENTATION_INDEX.md        (Navigation guide)
│   ├── CODE_COMMENTS_GUIDE.md       (500+ lines - code explanation)
│   ├── CODE_EXAMPLES_WITH_COMMENTS.md (Real code examples)
│   ├── CODE_DOCUMENTATION_COMPLETE.md (Completion summary)
│   ├── FINAL_SUMMARY.md             (Final overview)
│   ├── MODEL_EVALUATION_SUMMARY.md  (Results & comparison)
│   ├── DASHBOARD_READY.md           (Dashboard status)
│   └── DASHBOARD_DIAGNOSTIC_REPORT.md (Diagnostic report)
│
├── 📂 Training/                      (Training dataset)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── 📂 Testing/                       (Test dataset)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
├── README.md                         (This file)
├── START_DASHBOARD.bat              (Windows launcher)
├── QUICK_REFERENCE.md               (Quick commands)
├── MASTER_DOCUMENTATION.md          (Complete guide)
├── FINAL_PROJECT_INDEX.md          (Project index)
├── REPOSITORY_STRUCTURE.md          (File structure)
└── VALIDATION_REPORT.md             (Validation results)
```

---

## 📊 Model Performance

### Baseline CNN Model
- **Architecture**: 4 convolutional blocks + dense layers
- **Training**: From scratch
- **Test Accuracy**: 76.89%
- **Strengths**: Good for No Tumor detection (91% F1)
- **Weaknesses**: Struggles with Meningioma (28% F1)

### Enhanced VGG16 Model ⭐ RECOMMENDED
- **Architecture**: Transfer learning + fine-tuning
- **Training**: VGG16 pre-trained on ImageNet
- **Test Accuracy**: 86.19%
- **Improvement**: +9.30% over baseline
- **Balanced**: All classes perform well
- **Best For**: Production use

---

## 🎯 Features

### Dashboard (6 Pages)
1. **Home** - Project overview
2. **Dataset** - Statistics and visualizations
3. **Models** - Architecture and comparison
4. **Prediction** - Upload image → Get prediction
5. **Results** - Detailed metrics
6. **About** - Project information

### Functionality
✅ Train baseline CNN model  
✅ Train enhanced VGG16 model  
✅ Evaluate both models  
✅ Compare models side-by-side  
✅ Make predictions on new images  
✅ Interactive web dashboard  
✅ Comprehensive documentation  

---

## 📖 Documentation

### For Quick Start
- **QUICK_REFERENCE.md** - Essential commands
- **MASTER_DOCUMENTATION.md** - Complete overview

### For Understanding Code
- **DOCS/DOCUMENTATION_INDEX.md** - Navigation guide (START HERE)
- **DOCS/CODE_COMMENTS_GUIDE.md** - Comprehensive explanation (500+ lines)
- **DOCS/CODE_EXAMPLES_WITH_COMMENTS.md** - Real code examples

### For Learning
- **DOCS/CODE_DOCUMENTATION_COMPLETE.md** - Learning paths
- **DOCS/FINAL_SUMMARY.md** - Complete summary

### For Results
- **DOCS/MODEL_EVALUATION_SUMMARY.md** - Performance metrics
- **DOCS/DASHBOARD_READY.md** - Dashboard status
- **VALIDATION_REPORT.md** - Project validation

---

## 🛠️ Technology Stack

### Deep Learning
- TensorFlow 2.10.0
- Keras
- NumPy 1.23.5

### Web Interface
- Streamlit 1.28.1
- Matplotlib
- Seaborn

### Data Processing
- Pandas
- OpenCV (cv2)
- Scikit-learn

### Environment
- Python 3.10.11
- Virtual Environment (.venv)

---

## 📋 Dataset

### Size
- **Total Images**: 7,023
- **Training**: 5,712 images
- **Testing**: 1,311 images

### Classes (4 Tumor Types)
1. **Glioma** - Most common malignant tumor
2. **Meningioma** - Slow-growing tumor
3. **No Tumor** - Healthy brain scan
4. **Pituitary** - Hormonal gland tumor

### Image Format
- 150×150 RGB images (normalized to [0,1])
- Organized by class in folders

---

## 🎓 What's Included

### Code Comments
✅ 300+ lines of inline comments  
✅ Function documentation  
✅ Layer-by-layer explanations  
✅ Mathematical operation explanations  

### Documentation (1,500+ lines)
✅ Architecture explanations  
✅ Training process walkthrough  
✅ Evaluation metrics definitions  
✅ Learning paths (beginner to advanced)  
✅ Real code examples  
✅ Quick reference guides  

### Models & Results
✅ Trained baseline CNN model  
✅ Trained VGG16 transfer learning model  
✅ Evaluation reports  
✅ Performance visualizations  
✅ Confusion matrices  

---

## ⚡ Commands Reference

### Training
```bash
# Train baseline model (5-10 minutes)
python brain_tumor_project/src/train_model.py

# Train enhanced model (15-20 minutes)
python brain_tumor_project/src/train_model_enhanced.py
```

### Evaluation
```bash
# Evaluate baseline model
python brain_tumor_project/src/evaluate.py

# Evaluate enhanced model
python brain_tumor_project/src/evaluate_enhanced.py

# Compare both models
python brain_tumor_project/src/compare_models.py
```

### Dashboard
```bash
# Launch web interface
streamlit run dashboard_app/app_clean.py

# Then open: http://localhost:8501
```

### Prediction
```bash
# Predict with baseline model
python brain_tumor_project/src/predict.py image.jpg

# Predict with enhanced model
python brain_tumor_project/src/predict.py image.jpg --enhanced

# Compare both models
python brain_tumor_project/src/predict.py image.jpg --both
```

---

## 📚 Learning Resources

### Understanding Code
1. **Start**: DOCS/DOCUMENTATION_INDEX.md
2. **Read**: DOCS/CODE_COMMENTS_GUIDE.md
3. **Study**: DOCS/CODE_EXAMPLES_WITH_COMMENTS.md
4. **Reference**: Source files in brain_tumor_project/src/

### Learning Paths
- **Beginner**: Overview + CNN basics (30 min)
- **Intermediate**: Training + evaluation (1 hour)
- **Advanced**: Transfer learning + metrics (1.5 hours)
- **Expert**: Complete system (2-3 hours)

---

## ✅ Verification

All components verified and working:
- ✅ Models trained successfully
- ✅ Evaluation runs without errors
- ✅ Dashboard operational (localhost:8501)
- ✅ Code fully commented
- ✅ Documentation comprehensive
- ✅ All predictions working

---

## 📞 Quick Help

### "How do I start?"
→ Run: `streamlit run dashboard_app/app_clean.py`

### "How do I understand the code?"
→ Read: `DOCS/DOCUMENTATION_INDEX.md`

### "What's the model performance?"
→ Check: `DOCS/MODEL_EVALUATION_SUMMARY.md`

### "How do I make predictions?"
→ Use: `python brain_tumor_project/src/predict.py image.jpg --enhanced`

---

## 🎉 Ready to Use!

Everything is set up and ready. Choose your next step:

1. **Run Dashboard**: `streamlit run dashboard_app/app_clean.py`
2. **Learn Code**: Open `DOCS/DOCUMENTATION_INDEX.md`
3. **View Results**: Check `DOCS/MODEL_EVALUATION_SUMMARY.md`
4. **Make Predictions**: Run evaluation scripts

**Happy learning! 🚀**

---

*Project completed: December 4, 2024*  
*All code documented • All models trained • Ready for submission*
