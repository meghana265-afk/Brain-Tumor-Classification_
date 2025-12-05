"""
train_model_enhanced.py
Enhanced training script implementing advanced techniques for better performance:
- Transfer Learning (VGG16 pre-trained on ImageNet)
- Advanced data augmentation
- Learning rate scheduling
- Class weight balancing
- Multi-stage training (feature extraction + fine-tuning)
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    CSVLogger
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    IMG_SIZE, BATCH_SIZE, TRAIN_DIR, TEST_DIR,
    MODEL_PATH, OUTPUTS_DIR, verify_data_dirs
)

print("\n" + "="*70)
print("BRAIN TUMOR CLASSIFICATION - ENHANCED TRAINING WITH TRANSFER LEARNING")
print("="*70)

# Verify directories
verify_data_dirs()

# Configuration
ENHANCED_EPOCHS_STAGE1 = 15  # Feature extraction stage
ENHANCED_EPOCHS_STAGE2 = 25  # Fine-tuning stage
ENHANCED_MODEL_PATH = MODEL_PATH.parent / 'enhanced_model.h5'
BEST_ENHANCED_MODEL_PATH = MODEL_PATH.parent / 'best_enhanced_model.h5'
TRAINING_LOG_PATH = MODEL_PATH.parent / 'training_log.csv'

print(f"\n[STEP 1] Building Enhanced Model with Transfer Learning")
print("="*70)

def build_enhanced_model():
    """
    Build model using VGG16 pre-trained on ImageNet as feature extractor.
    VGG16 has learned general image features (edges, textures, shapes) from 
    1.4M images, which transfers well to medical imaging.
    """
    # Load VGG16 without top classification layer
    # include_top=False: Remove final dense layers
    # weights='imagenet': Use pre-trained weights
    # input_shape: Our image dimensions
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially for feature extraction
    # We'll unfreeze later for fine-tuning
    base_model.trainable = False
    
    print(f"[OK] Loaded VGG16 base model (frozen)")
    print(f"     Base model has {len(base_model.layers)} layers")
    
    # Build custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Better than Flatten for transfer learning
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(4, activation='softmax')(x)  # 4 classes
    
    model = Model(inputs=base_model.input, outputs=x)
    
    print(f"[OK] Custom classification head added")
    print(f"     Total parameters: {model.count_params():,}")
    
    return model

model = build_enhanced_model()

print(f"\n[STEP 2] Computing Class Weights for Imbalanced Data")
print("="*70)

# Create temporary data generator to count samples per class
temp_gen = ImageDataGenerator(rescale=1./255)
temp_flow = temp_gen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Calculate class weights to handle imbalanced dataset
from sklearn.utils.class_weight import compute_class_weight

# Get class distribution
class_counts = {}
for class_name, class_idx in temp_flow.class_indices.items():
    count = len([f for f in os.listdir(TRAIN_DIR / class_name) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    class_counts[class_name] = count
    print(f"     {class_name}: {count} images")

# Compute weights
total_samples = sum(class_counts.values())
class_weights = {}
for class_name, count in class_counts.items():
    class_idx = temp_flow.class_indices[class_name]
    weight = total_samples / (len(class_counts) * count)
    class_weights[class_idx] = weight
    print(f"     {class_name} weight: {weight:.3f}")

print(f"[OK] Class weights computed")

print(f"\n[STEP 3] Setting Up Advanced Data Augmentation")
print("="*70)

# Enhanced data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,           # Rotate up to 25 degrees
    width_shift_range=0.25,      # Shift horizontally
    height_shift_range=0.25,     # Shift vertically
    shear_range=0.2,             # Shearing transformation
    zoom_range=0.25,             # Random zoom
    horizontal_flip=True,        # Flip horizontally
    fill_mode='nearest',         # Fill empty pixels
    brightness_range=[0.8, 1.2]  # Vary brightness
)

# Validation data: only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

print("[OK] Data augmentation configured")
print("     Augmentations: rotation, shift, shear, zoom, flip, brightness")

# Create data generators
train_generator = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    str(TEST_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"[OK] Data generators created")
print(f"     Training samples: {train_generator.samples}")
print(f"     Testing samples: {test_generator.samples}")
print(f"     Classes: {list(train_generator.class_indices.keys())}")

print(f"\n[STAGE 1] Feature Extraction Training (Base Model Frozen)")
print("="*70)

# Compile model for feature extraction
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for stage 1
callbacks_stage1 = [
    ModelCheckpoint(
        str(BEST_ENHANCED_MODEL_PATH),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(str(TRAINING_LOG_PATH), append=True)
]

print("[OK] Starting Stage 1 training...")
print(f"     Epochs: {ENHANCED_EPOCHS_STAGE1}")
print(f"     Optimizer: Adam (lr=0.001)")

history_stage1 = model.fit(
    train_generator,
    epochs=ENHANCED_EPOCHS_STAGE1,
    validation_data=test_generator,
    callbacks=callbacks_stage1,
    class_weight=class_weights,
    verbose=1
)

print(f"\n[OK] Stage 1 complete!")
stage1_val_acc = max(history_stage1.history['val_accuracy'])
print(f"     Best validation accuracy: {stage1_val_acc:.4f} ({stage1_val_acc*100:.2f}%)")

print(f"\n[STAGE 2] Fine-Tuning (Unfreezing Last Conv Blocks)")
print("="*70)

# Unfreeze last 4 layers of VGG16 for fine-tuning
base_model = model.layers[0]
base_model.trainable = True

# Freeze all layers except last 4
for layer in base_model.layers[:-4]:
    layer.trainable = False

print(f"[OK] Unfroze last 4 layers of base model")
trainable_count = sum([1 for layer in model.layers if layer.trainable])
print(f"     Trainable layers: {trainable_count}")

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("[OK] Recompiled with lower learning rate (0.0001)")

# Callbacks for stage 2
callbacks_stage2 = [
    ModelCheckpoint(
        str(BEST_ENHANCED_MODEL_PATH),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-8,
        verbose=1
    ),
    CSVLogger(str(TRAINING_LOG_PATH), append=True)
]

print("[OK] Starting Stage 2 training...")
print(f"     Epochs: {ENHANCED_EPOCHS_STAGE2}")

history_stage2 = model.fit(
    train_generator,
    epochs=ENHANCED_EPOCHS_STAGE2,
    validation_data=test_generator,
    callbacks=callbacks_stage2,
    class_weight=class_weights,
    verbose=1
)

print(f"\n[OK] Stage 2 complete!")
stage2_val_acc = max(history_stage2.history['val_accuracy'])
print(f"     Best validation accuracy: {stage2_val_acc:.4f} ({stage2_val_acc*100:.2f}%)")

print(f"\n[STEP 4] Saving Final Model")
print("="*70)

# Save final model
model.save(str(ENHANCED_MODEL_PATH))
print(f"[OK] Enhanced model saved to: {ENHANCED_MODEL_PATH}")
print(f"[OK] Best model saved to: {BEST_ENHANCED_MODEL_PATH}")

print(f"\n[STEP 5] Generating Training Visualizations")
print("="*70)

# Combine histories
all_accuracy = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
all_val_accuracy = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
all_loss = history_stage1.history['loss'] + history_stage2.history['loss']
all_val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']

# Plot accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(all_accuracy, label='Training Accuracy', linewidth=2)
plt.plot(all_val_accuracy, label='Validation Accuracy', linewidth=2)
plt.axvline(x=ENHANCED_EPOCHS_STAGE1, color='red', linestyle='--', label='Fine-tuning starts')
plt.title('Enhanced Model - Training & Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(all_loss, label='Training Loss', linewidth=2)
plt.plot(all_val_loss, label='Validation Loss', linewidth=2)
plt.axvline(x=ENHANCED_EPOCHS_STAGE1, color='red', linestyle='--', label='Fine-tuning starts')
plt.title('Enhanced Model - Training & Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
enhanced_plot_path = OUTPUTS_DIR / 'enhanced_training_curves.png'
plt.savefig(str(enhanced_plot_path), dpi=150, bbox_inches='tight')
print(f"[OK] Training curves saved to: {enhanced_plot_path}")

print(f"\n[STEP 6] Training Summary")
print("="*70)

final_train_acc = all_accuracy[-1]
final_val_acc = all_val_accuracy[-1]
best_val_acc = max(all_val_accuracy)
improvement = (best_val_acc - stage1_val_acc) * 100

print(f"\nStage 1 (Feature Extraction):")
print(f"  Best validation accuracy: {stage1_val_acc:.4f} ({stage1_val_acc*100:.2f}%)")

print(f"\nStage 2 (Fine-tuning):")
print(f"  Best validation accuracy: {stage2_val_acc:.4f} ({stage2_val_acc*100:.2f}%)")

print(f"\nOverall Performance:")
print(f"  Final training accuracy:   {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"  Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"  Best validation accuracy:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"  Improvement from Stage 1:  {improvement:.2f}%")

print("\n" + "="*70)
print("[SUCCESS] Enhanced training complete!")
print("="*70)
print(f"\nNext steps:")
print(f"  1. Run evaluation: python src/evaluate_enhanced.py")
print(f"  2. Compare models: python src/compare_models.py")
print(f"  3. Make predictions: python src/predict.py <image_path> --enhanced")
print("="*70 + "\n")
