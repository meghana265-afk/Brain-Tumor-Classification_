"""
train_model.py
Line-by-line commented training script. The comments explain what each block
does and why it is included. This script builds a CNN, trains it, and saves
artifacts such as the best model and training plots.
"""

# Import TensorFlow for building and training the neural network
import tensorflow as tf

# Image data utility for creating training and validation generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Callbacks for early stopping and checkpointing the best model weights
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Matplotlib for plotting training curves
import matplotlib.pyplot as plt

# OS and sys for path handling and runtime behaviors
import os
import sys

# Ensure we can import from the src directory by adding it to sys.path
sys.path.insert(0, os.path.dirname(__file__))

# Import configuration values and helper verify function
from config import (
    IMG_SIZE, BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR,
    MODEL_PATH, ACCURACY_PLOT_PATH, LOSS_PLOT_PATH, verify_data_dirs
)


# Print header to make console output easy to scan
print("\n" + "="*60)
print("BRAIN TUMOR CLASSIFICATION - TRAINING")
print("="*60)


# Verify the training and testing directories exist before doing work
verify_data_dirs()


# Convert Path objects to strings for use with Keras ImageDataGenerator
# This is necessary because Keras expects string paths, not Path objects
train_path = str(TRAIN_DIR)
test_path = str(TEST_DIR)


# Print user-friendly message
print(f"\nPreparing data pipeline...")

# Create ImageDataGenerator objects for data loading
# ImageDataGenerator(rescale=1/255.0): Normalize pixel values from [0-255] to [0-1]
# Normalization is CRITICAL for neural networks:
#   - Prevents gradient explosion/vanishing
#   - Accelerates training convergence
#   - Helps weight initialization
# rescale=1/255: Divides all pixel values by 255 (equivalent to / 255.0)
train_gen = ImageDataGenerator(rescale=1/255.0)
test_gen = ImageDataGenerator(rescale=1/255.0)


# Create data generators that load images from disk in batches
# flow_from_directory(): Loads images from folder structure:
#   Training/
#     └─ glioma/
#        ├─ image1.jpg
#        └─ image2.jpg
#     └─ meningioma/
#        └─ ...
#   etc.
#
# Parameters explained:
#   train_path: Directory path to Training folder
#   target_size=(IMG_SIZE, IMG_SIZE): Resize all images to 150x150 pixels
#   class_mode="categorical": For multi-class classification (outputs one-hot encoded labels)
#     Example: Glioma = [1, 0, 0, 0]
#   batch_size=BATCH_SIZE: Process 32 images at a time (balance memory vs gradient stability)
#   seed=42: Set random seed for reproducibility (same order every run)
train_ds = train_gen.flow_from_directory(
    train_path, target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical", batch_size=BATCH_SIZE, seed=42
)

test_ds = test_gen.flow_from_directory(
    test_path, target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical", batch_size=BATCH_SIZE, shuffle=False, seed=42
)


# Print dataset summary so users can verify counts and class names
print(f"\n{'='*60}")
print(f"Dataset Summary:")
print(f"{'='*60}")
print(f"  Train samples: {train_ds.samples}")
print(f"  Test samples: {test_ds.samples}")
print(f"  Classes: {sorted(train_ds.class_indices.keys())}")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")


# Build a Sequential CNN model with 4 convolutional blocks
# Sequential: Stack layers one after another (linear flow)
# CNN (Convolutional Neural Network): Learns hierarchical features from images
print(f"\nBuilding CNN Model...\n")
model = tf.keras.Sequential([
    # BLOCK 1: Initial feature extraction (low-level features: edges, colors)
    # Conv2D(32, (3, 3)): 32 feature maps with 3x3 filters
    #   - Each filter slides over image to detect patterns
    #   - 32 filters = looking for 32 different patterns simultaneously
    #   - (3, 3) = filter size (3x3 pixels)
    # activation="relu": Rectified Linear Unit = max(0, x)
    #   - Introduces nonlinearity (model can learn non-linear patterns)
    #   - Keeps positive activations, zeroes negative (sparse representation)
    # input_shape=(IMG_SIZE, IMG_SIZE, 3): Input format
    #   - 150x150 pixels
    #   - 3 channels (RGB: Red, Green, Blue)
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # MaxPooling2D(2, 2): Reduce spatial dimensions by 2x
    #   - Takes 2x2 window and outputs maximum value
    #   - Reduces parameters and computation
    #   - Helps model focus on most important features
    #   - After: 150x150 → 75x75
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # BatchNormalization(): Normalize feature activations
    #   - Centers activations around mean=0, std=1
    #   - Benefits: faster training, allows higher learning rates, reduces sensitivity to weight initialization
    #   - Applied after pooling to stabilize gradient flow
    tf.keras.layers.BatchNormalization(),

    # BLOCK 2: Mid-level feature extraction (textures, shapes)
    # Conv2D(64, (3, 3)): Increase to 64 filters
    #   - Now extracting from already-extracted features
    #   - Can learn combinations of low-level features
    #   - More filters = more complex feature combinations possible
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),  # 75x75 → 37x37
    tf.keras.layers.BatchNormalization(),

    # BLOCK 3: Higher-level feature extraction (complex patterns)
    # Conv2D(128, (3, 3)): Further increase to 128 filters
    #   - Even more abstract features
    #   - Learned patterns from previous blocks
    #   - Examples: tumor regions, boundary patterns
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),  # 37x37 → 18x18
    tf.keras.layers.BatchNormalization(),

    # BLOCK 4: Most abstract features
    # Conv2D(256, (3, 3)): Maximum filters (256)
    #   - Final convolution block
    #   - Highest-level features before flattening
    #   - By this point, spatial info is about tumor/no-tumor distinction
    tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),  # 18x18 → 9x9
    tf.keras.layers.BatchNormalization(),

    # Flatten: Convert 3D feature maps to 1D vector
    #   - Input: 9x9x256 = 20,736 values
    #   - Output: 1D vector of 20,736 values
    #   - Needed for fully connected (dense) layers
    tf.keras.layers.Flatten(),
    
    # DENSE LAYERS: Classifier (learns to distinguish tumor types)
    # Dense(256): 256 fully connected neurons
    #   - Connect every neuron from previous layer to all 256 neurons
    #   - Learns combinations of extracted features
    #   - activation="relu": Same nonlinearity as conv layers
    tf.keras.layers.Dense(256, activation="relu"),
    
    # Dropout(0.5): Random deactivation during training
    #   - Probability 0.5: each neuron has 50% chance to be disabled
    #   - Forces network to learn redundant representations
    #   - Prevents co-adaptation (neurons becoming dependent on each other)
    #   - Acts as ensemble of models (different subnetworks)
    #   - Significantly reduces overfitting
    #   - NOT applied at test time (all neurons active)
    tf.keras.layers.Dropout(0.5),
    
    # Dense(128): Reduce dimensionality further
    #   - 256 → 128 features
    #   - Still learning high-level patterns
    tf.keras.layers.Dense(128, activation="relu"),
    
    # Dropout(0.3): More conservative dropout
    #   - Only 30% deactivation (closer to output)
    #   - Want less disruption near output layer
    tf.keras.layers.Dropout(0.3),

    # Dense(4, activation="softmax"): Output layer
    #   - 4 neurons: one for each tumor type (glioma, meningioma, notumor, pituitary)
    #   - activation="softmax": Converts outputs to probability distribution
    #     - exp(z_i) / sum(exp(z)) for each output
    #     - All outputs sum to 1.0
    #     - Can interpret as probabilities
    #   - Example output: [0.8, 0.1, 0.05, 0.05] = 80% Glioma, 10% Meningioma, etc.
    tf.keras.layers.Dense(4, activation="softmax")
])


# Compile the model with Adam optimizer and categorical crossentropy loss
# Compilation configures model for training (doesn't train yet)
# optimizer=Adam(learning_rate=0.001):
#   - Adam: Adaptive Moment Estimation
#   - Computes individual learning rates for each parameter
#   - learning_rate=0.001: How much to update weights per step
#     * Larger → faster training but may miss optimum
#     * Smaller → slower but more careful optimization
#   - Automatically adjusts based on gradient statistics
#   - Better than basic SGD for most problems
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # loss="categorical_crossentropy":
    #   - For multi-class classification (exactly one correct class)
    #   - Measures difference between predicted probabilities and true label
    #   - Formula: -sum(y_true * log(y_pred))
    #   - Encourages predicted probability of correct class to approach 1.0
    #   - Penalizes confidence in wrong classes
    loss="categorical_crossentropy",
    # metrics=["accuracy"]:
    #   - Metric to track during training (not used for optimization)
    #   - Accuracy = correct predictions / total predictions
    #   - More interpretable than loss value
    #   - Can also add: precision, recall, AUC, F1, etc.
    metrics=["accuracy"]
)


print("Model Architecture:")
print("-" * 60)
# model.summary(): Print layer-by-layer breakdown
#   - Shows layer name, output shape, number of parameters
#   - Useful for understanding model size and complexity
#   - Total parameters = sum of all layer parameters
#   - Trainable parameters = parameters that will be updated during training
model.summary()


# Configure callbacks: special actions to take during training
# Callbacks are functions that run at certain training events:
#   - After each epoch
#   - After each batch
#   - When validation improves/worsens
#   - Etc.

# CALLBACK 1: ModelCheckpoint
# Saves model weights to disk when validation metric improves
checkpoint = ModelCheckpoint(
    str(MODEL_PATH),                # Where to save the model
    save_best_only=True,            # Only save if this is the best so far
    monitor="val_accuracy",         # Track validation accuracy
    mode="max",                     # "max" = maximize (accuracy should be higher)
                                    # "min" = minimize (loss should be lower)
    verbose=1                       # Print message when model is saved
)
# Why this is important:
#   - Training stops after N epochs (fixed duration)
#   - But best model might be at epoch 5 out of 10
#   - This saves the actually best model, not the last one

# CALLBACK 2: EarlyStopping
# Stops training if validation metric stops improving
early_stop = EarlyStopping(
    monitor="val_loss",             # Track validation loss
    patience=3,                     # Wait 3 epochs for improvement before stopping
    restore_best_weights=True,      # Restore weights from best epoch
    verbose=1                       # Print stopping message
)
# Why this is important:
#   - Prevents training too long (wastes computation)
#   - Stops when model stops improving on validation set
#   - patience=3: Allow 3 epochs without improvement (in case of variance)
#   - Example: If val_loss was: [0.5, 0.48, 0.47, 0.475, 0.476, 0.478]
#     → Stops at epoch 6 (3 epochs of no improvement since epoch 3)


print(f"\n{'='*60}")
print("Training Model...")
print(f"{'='*60}\n")


# Start the training loop - this is where the learning happens
# history = model.fit(): Returns a History object tracking metrics per epoch
history = model.fit(
    train_ds,                       # Training data generator (batches of images)
    validation_data=test_ds,        # Validation data (used to compute val_accuracy, val_loss)
                                    # NOTE: Using test set as validation set (not ideal for production)
                                    # Better practice: split train into train/val, keep test separate
    epochs=EPOCHS,                  # Number of complete passes through training data (=10)
                                    # Epoch 1: see all 5712 images once
                                    # Epoch 2: see all 5712 images again (with better weights)
                                    # Etc.
    callbacks=[checkpoint, early_stop],  # Run callbacks (save best, stop early)
    verbose=1                       # Print progress bar after each epoch
)
# What happens during training:
# 1. For each epoch:
#    - Forward pass: Compute predictions for all training images
#    - Compute loss: Compare predictions to true labels
#    - Backward pass: Compute gradients of loss with respect to weights
#    - Update weights: Subtract gradient * learning_rate from weights
#    - Evaluate on validation set
# 2. After each epoch: Print loss, accuracy, val_loss, val_accuracy
# 3. Callbacks run: Check if best model, check if should stop
# 4. Continue until early stopping OR EPOCHS reached


print(f"\n{'='*60}")
print("Training Complete!")
print(f"{'='*60}")
print(f"\nModel saved to: {MODEL_PATH}")


# Generate training plots (accuracy and loss) using the history object
# history.history is a dictionary: {'loss': [...], 'accuracy': [...], 'val_loss': [...], 'val_accuracy': [...]}
# Each key maps to a list of values per epoch
print("\nGenerating training plots...")

# Create figure with 1 row, 2 columns (side-by-side plots)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LEFT PLOT: Accuracy over epochs
# X-axis: Epoch number
# Y-axis: Accuracy (0.0 to 1.0)
# Two lines: training accuracy (during training) and validation accuracy (on test set)
axes[0].plot(
    history.history["accuracy"],              # Training accuracy per epoch
    label="Train Accuracy",                   # Legend label
    linewidth=2,                              # Line thickness
    marker='o'                                # Circle marker at each point
)
axes[0].plot(
    history.history["val_accuracy"],          # Validation (test set) accuracy per epoch
    label="Validation Accuracy",
    linewidth=2,
    marker='s'                                # Square marker
)
# What to look for:
#   - Both lines increasing = model learning
#   - Training >> validation = overfitting
#   - Both low = underfitting
axes[0].set_xlabel("Epoch", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Accuracy", fontsize=12, fontweight='bold')
axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
axes[0].legend(fontsize=11)                  # Show legend with labels
axes[0].grid(True, alpha=0.3)                # Faint gridlines for readability

# RIGHT PLOT: Loss over epochs
# X-axis: Epoch number
# Y-axis: Loss (lower is better, typically 0.5 to 3.0)
# Two lines: training loss and validation loss
axes[1].plot(
    history.history["loss"],                  # Training loss per epoch
    label="Train Loss",
    linewidth=2,
    marker='o'
)
axes[1].plot(
    history.history["val_loss"],              # Validation loss per epoch
    label="Validation Loss",
    linewidth=2,
    marker='s'
)
# What to look for:
#   - Both lines decreasing = model learning
#   - Training >> validation = overfitting
#   - Flat line = learning has plateaued
axes[1].set_xlabel("Epoch", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Loss", fontsize=12, fontweight='bold')
axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

# Arrange plots without overlap
plt.tight_layout()

# Save the figure to disk for later inspection
# dpi=300: High resolution (300 dots per inch, professional quality)
# bbox_inches="tight": Remove extra whitespace around plot
plt.savefig(str(ACCURACY_PLOT_PATH), dpi=300, bbox_inches="tight")
print(f"[OK] Accuracy plot saved to: {ACCURACY_PLOT_PATH}")

# Close the figure to free memory (important if training many models)
plt.close()

print(f"\n{'='*60}")
print("All training files saved successfully!")
print(f"{'='*60}")
