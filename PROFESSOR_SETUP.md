# Professor Setup Guide (Clone → Run)

Use this if you’re setting up on a new machine from GitHub.

## 1) Clone the repository
```bash
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git
cd Brain-Tumor-Classification_
```

## 2) Choose your setup path
- **Windows (one-click):** run `SETUP.bat`
- **Mac/Linux (one-click):**
  ```bash
  chmod +x SETUP.sh
  ./SETUP.sh
  ```
- **Manual (any OS):**
  ```bash
  python -m venv .venv
  # Activate
  #   Windows: .\.venv\Scripts\Activate.ps1
  #   Mac/Linux: source .venv/bin/activate
  pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python streamlit==1.28.1 pandas
  ```

## 3) Data & models note (important)
- The GitHub repo contains **code and docs only** (no large data/models).
- You need local folders with images:
  - `Training/glioma/`, `Training/meningioma/`, `Training/notumor/`, `Training/pituitary/`
  - `Testing/glioma/`, `Testing/meningioma/`, `Testing/notumor/`, `Testing/pituitary/`
- If you have the pre-trained models, place them at `brain_tumor_project/models/`:
  - `saved_model.h5` (baseline)
  - `best_enhanced_model.h5` (VGG16, 86.19% accuracy)

## 4) Run the dashboard
```bash
cd dashboard_app
streamlit run app_clean.py
# Open http://localhost:8501
```

## 5) Full pipeline (from project root)
```bash
cd brain_tumor_project/src
python preprocess.py
python train_model.py
python train_model_enhanced.py
python evaluate.py
python evaluate_enhanced.py
python compare_models.py
python predict.py
```

## 6) Quick commands reference
- Clone: `git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git`
- Setup: `SETUP.bat` (Win) or `./SETUP.sh` (Mac/Linux)
- Dashboard: `streamlit run dashboard_app/app_clean.py`
- Predict: `python brain_tumor_project/src/predict.py <image_path>`

## 7) Where to read more
- `QUICKSTART.md` — 30-second setup
- `GETTING_STARTED.md` — full file-by-file instructions
- `DOCUMENTATION_INDEX.md` — master index of all docs
- `README.md` — project overview
