"""
evaluate_enhanced.py
Evaluation script specifically for the enhanced transfer learning model.
Provides comprehensive metrics and comparison with baseline model.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef
)
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR, MODEL_PATH, OUTPUTS_DIR, verify_data_dirs

print("\n" + "="*70)
print("ENHANCED MODEL - COMPREHENSIVE EVALUATION")
print("="*70)

# Verify directories
verify_data_dirs()

ENHANCED_MODEL_PATH = MODEL_PATH.parent / 'best_enhanced_model.h5'
BASELINE_MODEL_PATH = MODEL_PATH

print(f"\n[STEP 1] Loading Models")
print("="*70)

# Load enhanced model
if not ENHANCED_MODEL_PATH.exists():
    print(f"[ERROR] Enhanced model not found at: {ENHANCED_MODEL_PATH}")
    print("Please run train_model_enhanced.py first!")
    sys.exit(1)

enhanced_model = load_model(str(ENHANCED_MODEL_PATH))
print(f"[OK] Enhanced model loaded from: {ENHANCED_MODEL_PATH}")

# Load baseline model if available
baseline_model = None
if BASELINE_MODEL_PATH.exists():
    baseline_model = load_model(str(BASELINE_MODEL_PATH))
    print(f"[OK] Baseline model loaded from: {BASELINE_MODEL_PATH}")
else:
    print(f"[INFO] Baseline model not found. Skipping comparison.")

print(f"\n[STEP 2] Preparing Test Data")
print("="*70)

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    str(TEST_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())
print(f"[OK] Test data loaded")
print(f"     Samples: {test_generator.samples}")
print(f"     Classes: {class_names}")

print(f"\n[STEP 3] Running Enhanced Model Predictions")
print("="*70)

# Reset generator
test_generator.reset()

# Make predictions
print("[INFO] Processing test images...")
enhanced_predictions = enhanced_model.predict(test_generator, verbose=1)
enhanced_pred_classes = np.argmax(enhanced_predictions, axis=1)

# Get true labels
true_classes = test_generator.classes

print(f"[OK] Predictions complete")

print(f"\n[STEP 4] Computing Metrics for Enhanced Model")
print("="*70)

# Compute metrics
enhanced_accuracy = accuracy_score(true_classes, enhanced_pred_classes)
enhanced_precision_macro = precision_score(true_classes, enhanced_pred_classes, average='macro', zero_division=0)
enhanced_precision_weighted = precision_score(true_classes, enhanced_pred_classes, average='weighted', zero_division=0)
enhanced_recall_macro = recall_score(true_classes, enhanced_pred_classes, average='macro', zero_division=0)
enhanced_recall_weighted = recall_score(true_classes, enhanced_pred_classes, average='weighted', zero_division=0)
enhanced_f1_macro = f1_score(true_classes, enhanced_pred_classes, average='macro', zero_division=0)
enhanced_f1_weighted = f1_score(true_classes, enhanced_pred_classes, average='weighted', zero_division=0)
enhanced_cohen_kappa = cohen_kappa_score(true_classes, enhanced_pred_classes)
enhanced_matthews = matthews_corrcoef(true_classes, enhanced_pred_classes)

print(f"\nENHANCED MODEL PERFORMANCE:")
print(f"  Accuracy:              {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)")
print(f"  Precision (Macro):     {enhanced_precision_macro:.4f}")
print(f"  Precision (Weighted):  {enhanced_precision_weighted:.4f}")
print(f"  Recall (Macro):        {enhanced_recall_macro:.4f}")
print(f"  Recall (Weighted):     {enhanced_recall_weighted:.4f}")
print(f"  F1-Score (Macro):      {enhanced_f1_macro:.4f}")
print(f"  F1-Score (Weighted):   {enhanced_f1_weighted:.4f}")
print(f"  Cohen's Kappa:         {enhanced_cohen_kappa:.4f}")
print(f"  Matthews Corr Coef:    {enhanced_matthews:.4f}")

# Classification report
print(f"\nDETAILED CLASSIFICATION REPORT:")
print(classification_report(true_classes, enhanced_pred_classes, 
                          target_names=class_names, 
                          zero_division=0))

# Confusion matrix
enhanced_cm = confusion_matrix(true_classes, enhanced_pred_classes)

# If baseline model exists, evaluate it
if baseline_model:
    print(f"\n[STEP 5] Evaluating Baseline Model for Comparison")
    print("="*70)
    
    test_generator.reset()
    baseline_predictions = baseline_model.predict(test_generator, verbose=1)
    baseline_pred_classes = np.argmax(baseline_predictions, axis=1)
    
    baseline_accuracy = accuracy_score(true_classes, baseline_pred_classes)
    baseline_f1_macro = f1_score(true_classes, baseline_pred_classes, average='macro', zero_division=0)
    baseline_cm = confusion_matrix(true_classes, baseline_pred_classes)
    
    print(f"\nBASELINE MODEL PERFORMANCE:")
    print(f"  Accuracy:         {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"  F1-Score (Macro): {baseline_f1_macro:.4f}")
    
    print(f"\nIMPROVEMENT:")
    acc_improvement = (enhanced_accuracy - baseline_accuracy) * 100
    f1_improvement = (enhanced_f1_macro - baseline_f1_macro) * 100
    print(f"  Accuracy improvement:  {acc_improvement:+.2f}%")
    print(f"  F1-Score improvement:  {f1_improvement:+.2f}%")

print(f"\n[STEP 6] Generating Visualizations")
print("="*70)

# Create comprehensive visualization
if baseline_model:
    fig = plt.figure(figsize=(16, 5))
    
    # Enhanced model confusion matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(enhanced_cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    plt.title(f'Enhanced Model\nAccuracy: {enhanced_accuracy:.2%}', fontsize=12, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Baseline model confusion matrix
    plt.subplot(1, 3, 2)
    sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    plt.title(f'Baseline Model\nAccuracy: {baseline_accuracy:.2%}', fontsize=12, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Comparison bar chart
    plt.subplot(1, 3, 3)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    enhanced_values = [enhanced_accuracy, enhanced_precision_macro, enhanced_recall_macro, enhanced_f1_macro]
    baseline_values = [baseline_accuracy, 
                      precision_score(true_classes, baseline_pred_classes, average='macro', zero_division=0),
                      recall_score(true_classes, baseline_pred_classes, average='macro', zero_division=0),
                      baseline_f1_macro]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, baseline_values, width, label='Baseline', color='steelblue', alpha=0.8)
    plt.bar(x + width/2, enhanced_values, width, label='Enhanced', color='forestgreen', alpha=0.8)
    
    plt.ylabel('Score')
    plt.title('Model Comparison', fontsize=12, fontweight='bold')
    plt.xticks(x, metrics, rotation=0)
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bv, ev) in enumerate(zip(baseline_values, enhanced_values)):
        plt.text(i - width/2, bv + 0.02, f'{bv:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, ev + 0.02, f'{ev:.3f}', ha='center', va='bottom', fontsize=9)
    
else:
    # Only enhanced model
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(enhanced_cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    plt.title(f'Enhanced Model - Confusion Matrix\nAccuracy: {enhanced_accuracy:.2%}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

plt.tight_layout()
comparison_plot_path = OUTPUTS_DIR / 'enhanced_model_evaluation.png'
plt.savefig(str(comparison_plot_path), dpi=150, bbox_inches='tight')
print(f"[OK] Evaluation visualization saved to: {comparison_plot_path}")

print(f"\n[STEP 7] Saving Evaluation Report")
print("="*70)

report_path = MODEL_PATH.parent / 'enhanced_evaluation_report.txt'
with open(str(report_path), 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("ENHANCED MODEL - EVALUATION REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("ENHANCED MODEL PERFORMANCE:\n")
    f.write("-"*70 + "\n")
    f.write(f"Accuracy:              {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)\n")
    f.write(f"Precision (Macro):     {enhanced_precision_macro:.4f}\n")
    f.write(f"Precision (Weighted):  {enhanced_precision_weighted:.4f}\n")
    f.write(f"Recall (Macro):        {enhanced_recall_macro:.4f}\n")
    f.write(f"Recall (Weighted):     {enhanced_recall_weighted:.4f}\n")
    f.write(f"F1-Score (Macro):      {enhanced_f1_macro:.4f}\n")
    f.write(f"F1-Score (Weighted):   {enhanced_f1_weighted:.4f}\n")
    f.write(f"Cohen's Kappa:         {enhanced_cohen_kappa:.4f}\n")
    f.write(f"Matthews Corr Coef:    {enhanced_matthews:.4f}\n\n")
    
    f.write("DETAILED CLASSIFICATION REPORT:\n")
    f.write("-"*70 + "\n")
    f.write(classification_report(true_classes, enhanced_pred_classes, 
                                 target_names=class_names, 
                                 zero_division=0))
    f.write("\n")
    
    if baseline_model:
        f.write("\nCOMPARISON WITH BASELINE MODEL:\n")
        f.write("-"*70 + "\n")
        f.write(f"Baseline Accuracy:     {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)\n")
        f.write(f"Enhanced Accuracy:     {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)\n")
        f.write(f"Improvement:           {acc_improvement:+.2f}%\n\n")
        f.write(f"Baseline F1-Score:     {baseline_f1_macro:.4f}\n")
        f.write(f"Enhanced F1-Score:     {enhanced_f1_macro:.4f}\n")
        f.write(f"Improvement:           {f1_improvement:+.2f}%\n")

print(f"[OK] Report saved to: {report_path}")

print("\n" + "="*70)
print("[SUCCESS] Enhanced model evaluation complete!")
print("="*70)
print(f"\nResults Summary:")
print(f"  Enhanced Model Accuracy: {enhanced_accuracy:.2%}")
if baseline_model:
    print(f"  Baseline Model Accuracy: {baseline_accuracy:.2%}")
    print(f"  Improvement: {acc_improvement:+.2f}%")
print("="*70 + "\n")
