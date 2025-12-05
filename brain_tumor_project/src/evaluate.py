"""
evaluate.py - Baseline Model Comprehensive Evaluation Script
=============================================================

PURPOSE:
  Evaluate the trained baseline CNN model on both training and test sets.
  Compute comprehensive ML metrics (accuracy, precision, recall, F1, confusion matrix, etc.)
  Generate visualizations and save a detailed evaluation report.

USAGE:
  python src/evaluate.py

OUTPUT:
  - Confusion matrices (training vs test)
  - Detailed metrics for each tumor class
  - Evaluation report saved to models/evaluation_report.txt
  - Confusion matrix visualization saved to outputs/confusion_matrix.png

MODELS EVALUATED:
  - Baseline CNN: saved_model.h5 (simple convolutional neural network)
"""

# TensorFlow - Load and evaluate keras models
import tensorflow as tf

# NumPy - Array operations and argmax for class predictions
import numpy as np

# Scikit-learn - Comprehensive ML metrics (accuracy, F1, confusion matrix, etc.)
from sklearn.metrics import (
    classification_report,      # Per-class precision, recall, F1
    confusion_matrix,           # Prediction confusion visualization
    accuracy_score,             # Overall accuracy
    precision_score,            # How many predicted positives were correct
    recall_score,               # How many actual positives were found
    f1_score,                   # Harmonic mean of precision and recall
    roc_auc_score,              # ROC curve area (if applicable)
    cohen_kappa_score,          # Agreement accounting for chance
    matthews_corrcoef           # Correlation coefficient for predictions
)

# Keras ImageDataGenerator - Load images from disk in batches
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Matplotlib & Seaborn - Create visualization plots
import matplotlib.pyplot as plt
import seaborn as sns

# OS and sys - Path handling and system operations
import os
import sys

# Add src directory to import path so we can import config
sys.path.insert(0, os.path.dirname(__file__))

# Import project configuration (paths, sizes, class names, etc.)
from config import (
    IMG_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR, MODEL_PATH,
    CONFUSION_MATRIX_PATH, EVALUATION_REPORT_PATH, verify_data_dirs
)


def evaluate_dataset(model, data_dir, dataset_name, class_names):
    """
    FUNCTION: evaluate_dataset
    ===========================
    Evaluate model on a complete dataset and return comprehensive metrics.
    
    PARAMETERS:
      model: Keras model - The trained model to evaluate
      data_dir: Path - Directory containing labeled images (organized by class)
      dataset_name: str - Name of dataset (e.g., "training" or "test")
      class_names: list - Names of tumor classes
    
    RETURNS:
      metrics: dict - All computed metrics (accuracy, precision, recall, F1, confusion matrix, etc.)
      y_true: array - True class labels for each image
      y_pred: array - Predicted class labels
      y_pred_proba: array - Prediction probabilities for each class
    """
    
    print(f"\n{'='*70}")
    print(f"STEP {dataset_name.upper()}: Loading {dataset_name} Data")
    print(f"{'='*70}")
    
    # Create generator for the dataset
    data_gen = ImageDataGenerator(rescale=1/255.0)
    data_ds = data_gen.flow_from_directory(
        str(data_dir), 
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="categorical", 
        shuffle=False, 
        batch_size=BATCH_SIZE
    )
    
    print(f"  Samples: {data_ds.samples}")
    print(f"  Classes: {sorted(data_ds.class_indices.keys())}")
    
    print(f"\n{'='*70}")
    print(f"STEP {dataset_name.upper()}: Running Predictions")
    print(f"{'='*70}\n")
    
    # Get predictions
    predictions = model.predict(data_ds, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = data_ds.classes
    
    # Get prediction probabilities for each class
    y_pred_proba = predictions
    
    print(f"\n{'='*70}")
    print(f"STEP {dataset_name.upper()}: Computing Metrics")
    print(f"{'='*70}")
    
    # Calculate comprehensive metrics
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    
    return metrics, y_true, y_pred, y_pred_proba


def print_metrics(metrics, class_names, dataset_name):
    """Print all metrics in a formatted way."""
    
    print(f"\n{'='*70}")
    print(f"{dataset_name.upper()} SET - COMPREHENSIVE METRICS")
    print(f"{'='*70}\n")
    
    print("OVERALL METRICS:")
    print(f"  Accuracy:              {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (Macro):     {metrics['precision_macro']:.4f}")
    print(f"  Precision (Weighted):  {metrics['precision_weighted']:.4f}")
    print(f"  Recall (Macro):        {metrics['recall_macro']:.4f}")
    print(f"  Recall (Weighted):     {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (Macro):      {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  Cohen's Kappa:         {metrics['cohen_kappa']:.4f}")
    print(f"  Matthews Corr Coef:    {metrics['matthews_corrcoef']:.4f}")
    
    print(f"\n[PER-CLASS METRICS]")
    for idx, class_name in enumerate(class_names):
        print(f"\n  {class_name.upper()}:")
        print(f"    Precision: {metrics['precision_per_class'][idx]:.4f}")
        print(f"    Recall:    {metrics['recall_per_class'][idx]:.4f}")
        print(f"    F1-Score:  {metrics['f1_per_class'][idx]:.4f}")
    
    print(f"\n[DETAILED CLASSIFICATION REPORT]")
    print(metrics['classification_report'])
    
    print(f"\n[CONFUSION MATRIX]")
    print(metrics['confusion_matrix'])


print("\n" + "="*70)
print("BRAIN TUMOR CLASSIFICATION - COMPREHENSIVE EVALUATION")
print("="*70)

print("\n[STEP 1] Verifying Data Directories")
print("="*70)
verify_data_dirs()

print("\n[STEP 2] Loading Trained Model")
print("="*70)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("[OK] Model loaded successfully")

# Get class names
temp_gen = ImageDataGenerator(rescale=1/255.0)
temp_ds = temp_gen.flow_from_directory(
    str(TEST_DIR), 
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical", 
    shuffle=False, 
    batch_size=1
)
class_indices = temp_ds.class_indices
class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

print(f"Classes detected: {class_names}")

# Evaluate on TRAINING set
print("\n" + "="*70)
print("EVALUATING ON TRAINING SET")
print("="*70)
train_metrics, train_y_true, train_y_pred, train_y_pred_proba = evaluate_dataset(
    model, TRAIN_DIR, "Training", class_names
)
print_metrics(train_metrics, class_names, "Training")

# Evaluate on TEST set
print("\n" + "="*70)
print("EVALUATING ON TEST SET")
print("="*70)
test_metrics, test_y_true, test_y_pred, test_y_pred_proba = evaluate_dataset(
    model, TEST_DIR, "Test", class_names
)


print_metrics(test_metrics, class_names, "Test")

# Plot confusion matrices side by side
print("\n" + "="*70)
print("[STEP 5] Generating Visualizations")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training confusion matrix
sns.heatmap(train_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Training Set Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

# Test confusion matrix
sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('Test Set Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(str(CONFUSION_MATRIX_PATH), dpi=300, bbox_inches='tight')
print(f"[OK] Confusion matrices saved to: {CONFUSION_MATRIX_PATH}")
plt.close()

# Generate and print final verdict
print("\n" + "="*70)
print("🎯 FINAL VERDICT")
print("="*70)

# Analyze performance
train_acc = train_metrics['accuracy']
test_acc = test_metrics['accuracy']
gap = train_acc - test_acc

print(f"\n[PERFORMANCE SUMMARY]:")
print(f"  Training Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Test Accuracy:      {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Accuracy Gap:       {gap:.4f} ({gap*100:.2f}%)")

# Determine model state
if gap > 0.15:
    overfitting_status = "⚠️  HIGH OVERFITTING DETECTED"
    verdict_color = "WARNING"
elif gap > 0.08:
    overfitting_status = "⚠️  MODERATE OVERFITTING"
    verdict_color = "CAUTION"
else:
    overfitting_status = "✅ GOOD GENERALIZATION"
    verdict_color = "GOOD"

print(f"\n[OVERFITTING ANALYSIS]: {overfitting_status}")

# Overall performance assessment
if test_acc >= 0.95:
    performance = "EXCELLENT"
    performance_emoji = "🏆"
elif test_acc >= 0.90:
    performance = "VERY GOOD"
    performance_emoji = "⭐"
elif test_acc >= 0.85:
    performance = "GOOD"
    performance_emoji = "👍"
elif test_acc >= 0.80:
    performance = "ACCEPTABLE"
    performance_emoji = "✓"
else:
    performance = "NEEDS IMPROVEMENT"
    performance_emoji = "⚠️"

print(f"\n{performance_emoji} Overall Performance: {performance}")
print(f"   Test F1-Score (Macro):     {test_metrics['f1_macro']:.4f}")
print(f"   Test F1-Score (Weighted):  {test_metrics['f1_weighted']:.4f}")
print(f"   Cohen's Kappa:             {test_metrics['cohen_kappa']:.4f}")

# Per-class analysis
print(f"\n[PER-CLASS PERFORMANCE]:")
weakest_class_idx = np.argmin(test_metrics['f1_per_class'])
strongest_class_idx = np.argmax(test_metrics['f1_per_class'])

print(f"   Strongest Class: {class_names[strongest_class_idx].upper()} "
      f"(F1: {test_metrics['f1_per_class'][strongest_class_idx]:.4f})")
print(f"   Weakest Class:   {class_names[weakest_class_idx].upper()} "
      f"(F1: {test_metrics['f1_per_class'][weakest_class_idx]:.4f})")

# Recommendations
print(f"\n[RECOMMENDATIONS]:")
if gap > 0.15:
    print("   - Consider adding regularization (dropout, L2)")
    print("   - Try data augmentation")
    print("   - Reduce model complexity")
elif gap > 0.08:
    print("   - Monitor for overfitting")
    print("   - Consider slight regularization adjustments")

if test_acc < 0.90:
    print("   - Consider training for more epochs")
    print("   - Try different learning rates")
    print("   - Experiment with model architecture")

if test_metrics['f1_per_class'][weakest_class_idx] < 0.80:
    print(f"   - Focus on improving '{class_names[weakest_class_idx]}' class")
    print(f"   - Check class balance and data quality")

# Save comprehensive report
print("\n" + "="*70)
print("[STEP 6] Saving Comprehensive Report")
print("="*70)

with open(str(EVALUATION_REPORT_PATH), "w", encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("BRAIN TUMOR CLASSIFICATION - COMPREHENSIVE EVALUATION REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("TRAINING SET RESULTS:\n")
    f.write("="*70 + "\n")
    f.write(f"Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)\n")
    f.write(f"F1-Score (Macro): {train_metrics['f1_macro']:.4f}\n")
    f.write(f"F1-Score (Weighted): {train_metrics['f1_weighted']:.4f}\n\n")
    f.write(train_metrics['classification_report'])
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(train_metrics['confusion_matrix']))
    
    f.write("\n\n" + "="*70 + "\n")
    f.write("TEST SET RESULTS:\n")
    f.write("="*70 + "\n")
    f.write(f"Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    f.write(f"F1-Score (Macro): {test_metrics['f1_macro']:.4f}\n")
    f.write(f"F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}\n")
    f.write(f"Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}\n")
    f.write(f"Matthews Corr Coef: {test_metrics['matthews_corrcoef']:.4f}\n\n")
    f.write(test_metrics['classification_report'])
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(test_metrics['confusion_matrix']))
    
    f.write("\n\n" + "="*70 + "\n")
    f.write("FINAL VERDICT\n")
    f.write("="*70 + "\n")
    f.write(f"Overall Performance: {performance}\n")
    # Remove Unicode characters from overfitting_status
    overfitting_clean = overfitting_status.replace('✅', '[OK]').replace('⚠️', '[WARNING]').replace('❌', '[ERROR]')
    f.write(f"Overfitting Status: {overfitting_clean}\n")
    f.write(f"Training Accuracy: {train_acc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Accuracy Gap: {gap:.4f}\n")
    f.write(f"\nStrongest Class: {class_names[strongest_class_idx]} (F1: {test_metrics['f1_per_class'][strongest_class_idx]:.4f})\n")
    f.write(f"Weakest Class: {class_names[weakest_class_idx]} (F1: {test_metrics['f1_per_class'][weakest_class_idx]:.4f})\n")

print(f"[OK] Comprehensive report saved to: {EVALUATION_REPORT_PATH}")

print("\n" + "="*70)
print(f"EVALUATION COMPLETE - MODEL PERFORMANCE: {performance}")
print("="*70)
