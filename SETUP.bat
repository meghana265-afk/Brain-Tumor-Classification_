@echo off
REM Brain Tumor Classification - Quick Setup Script
REM Run this script to automatically set up the project

echo.
echo ====================================
echo  Brain Tumor Classification Setup
echo ====================================
echo.

REM Step 1: Clone repository
echo [Step 1/5] Cloning repository...
git clone https://github.com/meghana265-afk/Brain-Tumor-Classification_.git
cd Brain-Tumor-Classification_

REM Step 2: Create virtual environment
echo.
echo [Step 2/5] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

REM Step 3: Upgrade pip
echo.
echo [Step 3/5] Upgrading pip...
python -m pip install --upgrade pip

REM Step 4: Install dependencies
echo.
echo [Step 4/5] Installing dependencies...
pip install tensorflow==2.10.0 numpy==1.23.5 scikit-learn matplotlib pillow opencv-python pandas

REM Step 5: Verify installation
echo.
echo [Step 5/5] Verifying installation...
python -c "import tensorflow; import numpy; print('✓ All packages installed successfully!')"

echo.
echo ====================================
echo  Setup Complete!
echo ====================================
echo.
echo Next steps:
echo 1. Check data folders (Training/ and Testing/)
echo 2. Run: cd brain_tumor_project\src
echo 3. Run: python preprocess.py
echo 4. Run: python train_model.py
echo 5. Run: python evaluate.py (or predict.py) inside brain_tumor_project\src
echo.
echo Documentation: Read GETTING_STARTED.md for detailed instructions
echo.
pause
