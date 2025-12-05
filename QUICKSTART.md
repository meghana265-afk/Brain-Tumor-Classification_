# Brain Tumor Classification - Quick Start

## 🚀 30 Second Setup

### Windows
```powershell
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git
cd Brain-Tumor-Classification_
SETUP.bat
```

### Mac/Linux
```bash
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git
cd Brain-Tumor-Classification_
chmod +x SETUP.sh
./SETUP.sh
```

---

## 📚 Documentation

### For Complete Instructions
👉 **Read: `GETTING_STARTED.md`** (Comprehensive guide for every file)

### Structure
- **SETUP.bat** - Automatic setup (Windows)
- **SETUP.sh** - Automatic setup (Mac/Linux)
- **GETTING_STARTED.md** - Detailed file-by-file instructions
- **COMPLETE_GUIDE.txt** - All commands reference
- **brain_tumor_project/src/** - All source code files

---

## 📊 Quick Commands

### 1. Clone & Setup
```bash
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git
cd Brain-Tumor-Classification_
python -m venv .venv
# Activate: .venv\Scripts\Activate.ps1 (Windows) or source .venv/bin/activate (Mac/Linux)
pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas
```

### 2. Prepare Data
```bash
cd brain_tumor_project/src
python preprocess.py
```

### 3. Train Models
```bash
python train_model.py                # Baseline CNN
python train_model_enhanced.py       # VGG16 Transfer Learning
```

### 4. Evaluate
```bash
python evaluate.py                   # Test baseline
python evaluate_enhanced.py          # Test enhanced
python compare_models.py             # Compare both
```

### 5. Make Predictions
```bash
python predict.py                    # Command line prediction
```

### 6. Run Dashboard
```bash
cd ../../dashboard_app
streamlit run app_clean.py
# Open: http://localhost:8501
```

---

## 📁 File Guide

| File | Purpose | Command |
|------|---------|---------|
| `config.py` | Configuration | `python config.py` |
| `preprocess.py` | Data preparation | `python preprocess.py` |
| `train_model.py` | Train baseline CNN | `python train_model.py` |
| `train_model_enhanced.py` | Train VGG16 | `python train_model_enhanced.py` |
| `evaluate.py` | Test baseline | `python evaluate.py` |
| `evaluate_enhanced.py` | Test VGG16 | `python evaluate_enhanced.py` |
| `predict.py` | Predict on images | `python predict.py` |
| `compare_models.py` | Compare models | `python compare_models.py` |
| `app_clean.py` | Web dashboard | `streamlit run app_clean.py` |

---

## 🎯 Step-by-Step (First Time)

1. **Clone repository**
   ```bash
   git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git
   cd Brain-Tumor-Classification_
   ```

2. **Create environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate  # Windows
   # or
   source .venv/bin/activate  # Mac/Linux
   ```

3. **Install packages**
   ```bash
   pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas
   ```

4. **Prepare data**
   ```bash
   cd brain_tumor_project/src
   python preprocess.py
   ```

5. **Train models** (4-5 hours total)
   ```bash
   python train_model.py
   python train_model_enhanced.py
   ```

6. **Evaluate models**
   ```bash
   python evaluate.py
   python evaluate_enhanced.py
   python compare_models.py
   ```

7. **Run dashboard**
   ```bash
   streamlit run ../../dashboard_app/app_clean.py
   ```

8. **Make predictions**
   ```bash
   python predict.py
   ```

---

## 📈 Expected Results

### Baseline CNN
- Accuracy: **76.89%**
- Precision: 0.77
- Recall: 0.77

### VGG16 (Enhanced) ⭐
- Accuracy: **86.19%** (Best)
- Precision: 0.86
- Recall: 0.86
- **Improvement: +9.3%**

---

## 🌐 Dashboard

**Access at:** `http://localhost:8501`

### Features
- 📊 Dataset visualization
- 🧠 Model architecture details
- 🔮 Real-time predictions
- 📉 Performance metrics
- 🎯 Model comparison

---

## 🐛 Troubleshooting

### Issue: Module not found
```bash
pip install --upgrade pip
pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas
```

### Issue: CUDA/GPU not found
- TensorFlow will use CPU (slower but works)
- GPU support optional for this project

### Issue: Data not found
- Ensure `Training/` and `Testing/` folders exist
- Check folder structure: `Training/{glioma,meningioma,notumor,pituitary}/`

### Issue: Port 8501 already in use
```bash
streamlit run app_clean.py --server.port 8502
```

---

## 📖 Full Documentation

**See `GETTING_STARTED.md` for:**
- Detailed instructions for every file
- Parameter explanations
- Expected outputs
- Advanced usage
- Complete troubleshooting

---

## 🔗 Repository

**GitHub:** https://github.com/meghana265-afk/Brain-Tumor-Classification_

**Branch:** main

---

## ✅ Checklist

- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install packages
- [ ] Verify Python works
- [ ] Prepare data (preprocess.py)
- [ ] Train models
- [ ] Evaluate models
- [ ] Run dashboard
- [ ] Make predictions
- [ ] Explore visualizations

---

**Ready to start?** Run `SETUP.bat` (Windows) or `SETUP.sh` (Mac/Linux) for automatic setup! 🚀
