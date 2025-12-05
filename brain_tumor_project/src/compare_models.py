"""
compare_models.py
Comprehensive comparison between baseline and enhanced models.
Generates detailed side-by-side analysis and recommendations.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, f1_score
)
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR, MODEL_PATH, OUTPUTS_DIR, verify_data_dirs

print("\n" + "="*70)
print("MODEL COMPARISON: BASELINE vs ENHANCED")
print("="*70)

# Verify directories
verify_data_dirs()

ENHANCED_MODEL_PATH = MODEL_PATH.parent / 'best_enhanced_model.h5'
BASELINE_MODEL_PATH = MODEL_PATH

print(f"\n[STEP 1] Loading Both Models")
print("="*70)

# Check if models exist
if not BASELINE_MODEL_PATH.exists():
    print(f"[ERROR] Baseline model not found at: {BASELINE_MODEL_PATH}")
    print("Please run train_model.py first!")
    sys.exit(1)

if not ENHANCED_MODEL_PATH.exists():
    print(f"[ERROR] Enhanced model not found at: {ENHANCED_MODEL_PATH}")
    print("Please run train_model_enhanced.py first!")
    sys.exit(1)

# Load models
baseline_model = load_model(str(BASELINE_MODEL_PATH))
enhanced_model = load_model(str(ENHANCED_MODEL_PATH))

print(f"[OK] Baseline model loaded")
print(f"     Parameters: {baseline_model.count_params():,}")
print(f"[OK] Enhanced model loaded")
print(f"     Parameters: {enhanced_model.count_params():,}")

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
true_classes = test_generator.classes

print(f"[OK] Test data loaded")
print(f"     Samples: {test_generator.samples}")
print(f"     Classes: {class_names}")

print(f"\n[STEP 3] Evaluating Baseline Model")
print("="*70)

test_generator.reset()
print("[INFO] Running baseline predictions...")
baseline_predictions = baseline_model.predict(test_generator, verbose=1)
baseline_pred_classes = np.argmax(baseline_predictions, axis=1)

baseline_metrics = {
    'accuracy': accuracy_score(true_classes, baseline_pred_classes),
    'precision': precision_recall_fscore_support(true_classes, baseline_pred_classes, average='macro', zero_division=0)[0],
    'recall': precision_recall_fscore_support(true_classes, baseline_pred_classes, average='macro', zero_division=0)[1],
    'f1_score': f1_score(true_classes, baseline_pred_classes, average='macro', zero_division=0)
}

# Per-class metrics
baseline_per_class = precision_recall_fscore_support(true_classes, baseline_pred_classes, 
                                                     average=None, zero_division=0)

print(f"[OK] Baseline evaluation complete")
print(f"     Accuracy: {baseline_metrics['accuracy']:.2%}")

print(f"\n[STEP 4] Evaluating Enhanced Model")
print("="*70)

test_generator.reset()
print("[INFO] Running enhanced predictions...")
enhanced_predictions = enhanced_model.predict(test_generator, verbose=1)
enhanced_pred_classes = np.argmax(enhanced_predictions, axis=1)

enhanced_metrics = {
    'accuracy': accuracy_score(true_classes, enhanced_pred_classes),
    'precision': precision_recall_fscore_support(true_classes, enhanced_pred_classes, average='macro', zero_division=0)[0],
    'recall': precision_recall_fscore_support(true_classes, enhanced_pred_classes, average='macro', zero_division=0)[1],
    'f1_score': f1_score(true_classes, enhanced_pred_classes, average='macro', zero_division=0)
}

# Per-class metrics
enhanced_per_class = precision_recall_fscore_support(true_classes, enhanced_pred_classes, 
                                                     average=None, zero_division=0)

print(f"[OK] Enhanced evaluation complete")
print(f"     Accuracy: {enhanced_metrics['accuracy']:.2%}")

print(f"\n[STEP 5] Computing Comparisons")
print("="*70)

# Calculate improvements
improvements = {}
for metric in baseline_metrics.keys():
    baseline_val = baseline_metrics[metric]
    enhanced_val = enhanced_metrics[metric]
    improvement = (enhanced_val - baseline_val) * 100
    improvements[metric] = improvement
    print(f"{metric.replace('_', ' ').title():<15} | Baseline: {baseline_val:.4f} | Enhanced: {enhanced_val:.4f} | Change: {improvement:+.2f}%")

# Per-class comparison
print(f"\nPER-CLASS F1-SCORE COMPARISON:")
print(f"{'Class':<15} | {'Baseline':<10} | {'Enhanced':<10} | {'Change':<10}")
print("-" * 55)
for i, class_name in enumerate(class_names):
    baseline_f1 = baseline_per_class[2][i]
    enhanced_f1 = enhanced_per_class[2][i]
    change = (enhanced_f1 - baseline_f1) * 100
    print(f"{class_name:<15} | {baseline_f1:<10.4f} | {enhanced_f1:<10.4f} | {change:+.2f}%")

print(f"\n[STEP 6] Generating Comprehensive Visualizations")
print("="*70)

# Create comprehensive comparison figure
fig = plt.figure(figsize=(20, 12))

# 1. Overall Metrics Comparison (Bar Chart)
plt.subplot(2, 3, 1)
metrics_list = list(baseline_metrics.keys())
baseline_values = [baseline_metrics[m] for m in metrics_list]
enhanced_values = [enhanced_metrics[m] for m in metrics_list]

x = np.arange(len(metrics_list))
width = 0.35

bars1 = plt.bar(x - width/2, baseline_values, width, label='Baseline', color='#FF6B6B', alpha=0.8)
bars2 = plt.bar(x + width/2, enhanced_values, width, label='Enhanced', color='#4ECDC4', alpha=0.8)

plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('Overall Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, [m.replace('_', ' ').title() for m in metrics_list], rotation=15, ha='right')
plt.ylim([0, 1])
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Per-Class F1-Score Comparison
plt.subplot(2, 3, 2)
baseline_f1_per_class = baseline_per_class[2]
enhanced_f1_per_class = enhanced_per_class[2]

x = np.arange(len(class_names))
bars1 = plt.bar(x - width/2, baseline_f1_per_class, width, label='Baseline', color='#FF6B6B', alpha=0.8)
bars2 = plt.bar(x + width/2, enhanced_f1_per_class, width, label='Enhanced', color='#4ECDC4', alpha=0.8)

plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, class_names, rotation=15, ha='right')
plt.ylim([0, 1])
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# 3. Improvement Heatmap
plt.subplot(2, 3, 3)
improvement_data = []
for i, class_name in enumerate(class_names):
    row = []
    for metric_idx in range(3):  # Precision, Recall, F1
        baseline_val = baseline_per_class[metric_idx][i]
        enhanced_val = enhanced_per_class[metric_idx][i]
        improvement = (enhanced_val - baseline_val) * 100
        row.append(improvement)
    improvement_data.append(row)

improvement_df = pd.DataFrame(improvement_data, 
                              index=class_names,
                              columns=['Precision', 'Recall', 'F1-Score'])

sns.heatmap(improvement_df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Improvement (%)'}, linewidths=1)
plt.title('Per-Class Improvement Heatmap', fontsize=14, fontweight='bold')
plt.ylabel('Class')

# 4. Baseline Confusion Matrix
plt.subplot(2, 3, 4)
baseline_cm = confusion_matrix(true_classes, baseline_pred_classes)
sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Reds', alpha=0.8,
            xticklabels=class_names, yticklabels=class_names, 
            cbar_kws={'label': 'Count'})
plt.title(f'Baseline Model Confusion Matrix\nAccuracy: {baseline_metrics["accuracy"]:.2%}', 
          fontsize=12, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 5. Enhanced Confusion Matrix
plt.subplot(2, 3, 5)
enhanced_cm = confusion_matrix(true_classes, enhanced_pred_classes)
sns.heatmap(enhanced_cm, annot=True, fmt='d', cmap='Greens', alpha=0.8,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title(f'Enhanced Model Confusion Matrix\nAccuracy: {enhanced_metrics["accuracy"]:.2%}', 
          fontsize=12, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 6. Prediction Confidence Comparison
plt.subplot(2, 3, 6)
baseline_confidence = np.max(baseline_predictions, axis=1)
enhanced_confidence = np.max(enhanced_predictions, axis=1)

plt.hist(baseline_confidence, bins=30, alpha=0.6, label='Baseline', color='#FF6B6B')
plt.hist(enhanced_confidence, bins=30, alpha=0.6, label='Enhanced', color='#4ECDC4')
plt.xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
comparison_plot_path = OUTPUTS_DIR / 'model_comparison_comprehensive.png'
plt.savefig(str(comparison_plot_path), dpi=150, bbox_inches='tight')
print(f"[OK] Comprehensive comparison saved to: {comparison_plot_path}")

print(f"\n[STEP 7] Generating Comparison Report")
print("="*70)

report_path = MODEL_PATH.parent / 'model_comparison_report.txt'
with open(str(report_path), 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("MODEL COMPARISON REPORT: BASELINE vs ENHANCED\n")
    f.write("="*70 + "\n\n")
    
    f.write("OVERALL METRICS COMPARISON:\n")
    f.write("-"*70 + "\n")
    f.write(f"{'Metric':<20} | {'Baseline':<12} | {'Enhanced':<12} | {'Change':<12}\n")
    f.write("-"*70 + "\n")
    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        enhanced_val = enhanced_metrics[metric]
        change = improvements[metric]
        f.write(f"{metric.replace('_', ' ').title():<20} | {baseline_val:<12.4f} | {enhanced_val:<12.4f} | {change:+.2f}%\n")
    
    f.write("\n\nPER-CLASS F1-SCORE COMPARISON:\n")
    f.write("-"*70 + "\n")
    f.write(f"{'Class':<15} | {'Baseline':<12} | {'Enhanced':<12} | {'Change':<12}\n")
    f.write("-"*70 + "\n")
    for i, class_name in enumerate(class_names):
        baseline_f1 = baseline_per_class[2][i]
        enhanced_f1 = enhanced_per_class[2][i]
        change = (enhanced_f1 - baseline_f1) * 100
        f.write(f"{class_name:<15} | {baseline_f1:<12.4f} | {enhanced_f1:<12.4f} | {change:+.2f}%\n")
    
    f.write("\n\nKEY FINDINGS:\n")
    f.write("-"*70 + "\n")
    
    # Determine verdict
    acc_improvement = improvements['accuracy']
    if acc_improvement > 20:
        verdict = "SIGNIFICANT IMPROVEMENT"
    elif acc_improvement > 10:
        verdict = "SUBSTANTIAL IMPROVEMENT"
    elif acc_improvement > 5:
        verdict = "MODERATE IMPROVEMENT"
    elif acc_improvement > 0:
        verdict = "SLIGHT IMPROVEMENT"
    else:
        verdict = "NO IMPROVEMENT"
    
    f.write(f"Overall Verdict: {verdict}\n")
    f.write(f"Accuracy Improvement: {acc_improvement:+.2f}%\n")
    f.write(f"F1-Score Improvement: {improvements['f1_score']:+.2f}%\n\n")
    
    # Find best improved class
    improvements_per_class = []
    for i in range(len(class_names)):
        baseline_f1 = baseline_per_class[2][i]
        enhanced_f1 = enhanced_per_class[2][i]
        improvement = (enhanced_f1 - baseline_f1) * 100
        improvements_per_class.append((class_names[i], improvement))
    
    improvements_per_class.sort(key=lambda x: x[1], reverse=True)
    
    f.write("Most Improved Class:\n")
    f.write(f"  {improvements_per_class[0][0]}: {improvements_per_class[0][1]:+.2f}%\n\n")
    
    f.write("Least Improved Class:\n")
    f.write(f"  {improvements_per_class[-1][0]}: {improvements_per_class[-1][1]:+.2f}%\n\n")
    
    f.write("\nRECOMMENDATIONS:\n")
    f.write("-"*70 + "\n")
    if enhanced_metrics['accuracy'] >= 0.85:
        f.write("- Model performance is good. Consider deploying the enhanced model.\n")
    else:
        f.write("- Consider further training or data augmentation.\n")
    
    if improvements['accuracy'] > 10:
        f.write("- Enhanced model shows significant improvement. Use it for production.\n")
    elif improvements['accuracy'] > 0:
        f.write("- Enhanced model is better. Recommended for production use.\n")
    else:
        f.write("- Baseline model performs similarly. Review training procedure.\n")
    
    # Check for weak classes
    weak_classes = [name for name, score in zip(class_names, enhanced_per_class[2]) if score < 0.7]
    if weak_classes:
        f.write(f"- Focus on improving these weak classes: {', '.join(weak_classes)}\n")

print(f"[OK] Comparison report saved to: {report_path}")

print("\n" + "="*70)
print("[SUCCESS] Model comparison complete!")
print("="*70)
print(f"\nSummary:")
print(f"  Baseline Accuracy:  {baseline_metrics['accuracy']:.2%}")
print(f"  Enhanced Accuracy:  {enhanced_metrics['accuracy']:.2%}")
print(f"  Improvement:        {improvements['accuracy']:+.2f}%")
print(f"\nRecommendation: {'Use Enhanced Model' if improvements['accuracy'] > 0 else 'Review Training'}")
print("="*70 + "\n")
