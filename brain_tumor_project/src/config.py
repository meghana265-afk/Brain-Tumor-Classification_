"""
Configuration file for Brain Tumor Classification Project
Centralized paths and hyperparameters
"""

"""
config.py
Line-by-line commented configuration and path definitions for the project.

This file centralizes all file paths and hyperparameters so other scripts
importing this module have one canonical source of configuration.
"""

# Import Path for cross-platform path manipulation
from pathlib import Path

# Calculate project root as parent of the `src/` directory (this file)
# Using Path ensures Windows and Unix paths behave consistently.
PROJECT_ROOT = Path(__file__).parent.parent

# Define parent directory (one level above project root) where `Training/` and
# `Testing/` folders are expected to exist in the user's workspace.
PARENT_DIR = PROJECT_ROOT.parent

# Path to training dataset folder (expected: ../Training/)
TRAIN_DIR = PARENT_DIR / "Training"

# Path to testing dataset folder (expected: ../Testing/)
TEST_DIR = PARENT_DIR / "Testing"

# Directory where trained models will be saved (inside project)
MODELS_DIR = PROJECT_ROOT / "models"

# Directory where plots and outputs will be saved
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create the output directories if they don't already exist. This prevents
# downstream code from failing when trying to save artifacts.
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Image size used for training and inference (square images)
IMG_SIZE = 150

# Batch size used for training and evaluation
BATCH_SIZE = 32

# Number of epochs for training (kept modest for quick iteration)
EPOCHS = 10

# Fraction of data reserved for validation if using a split; currently unused
VALIDATION_SPLIT = 0.2

# Canonical class names used by the dataset and model output ordering.
# Keep this in sync with the folders inside Training/ and Testing/.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Canonical file paths for saving model and evaluation artifacts.
MODEL_PATH = MODELS_DIR / "saved_model.h5"
EVALUATION_REPORT_PATH = MODELS_DIR / "evaluation_report.txt"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
ACCURACY_PLOT_PATH = OUTPUTS_DIR / "accuracy_plot.png"
LOSS_PLOT_PATH = OUTPUTS_DIR / "loss_plot.png"


def verify_data_dirs():
    """
    Verify that `TRAIN_DIR` and `TEST_DIR` exist and raise a helpful error
    if they are missing. This check is used by training and evaluation scripts
    to fail early with a clear message.
    """

    # If training directory does not exist, raise an informative error
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory not found at: {TRAIN_DIR}")

    # If testing directory does not exist, raise an informative error
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Testing directory not found at: {TEST_DIR}")

    # Print confirmations to the console for interactive runs
    print(f"[OK] Training data found at: {TRAIN_DIR}")
    print(f"[OK] Testing data found at: {TEST_DIR}")


if __name__ == "__main__":
    # When running this file directly, show configured paths and run verification
    print("Configuration Paths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Training Data: {TRAIN_DIR}")
    print(f"  Testing Data: {TEST_DIR}")
    print(f"  Models Dir: {MODELS_DIR}")
    print(f"  Outputs Dir: {OUTPUTS_DIR}")
    verify_data_dirs()
