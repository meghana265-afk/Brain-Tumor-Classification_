# Code Examples with Line-by-Line Comments
## Brain Tumor Classification Project

This document contains real code examples with detailed inline comments explaining every line.

---

## Example 1: Building the Baseline CNN Model

```python
# Build a Sequential CNN model with 4 convolutional blocks
# Sequential: Stack layers one after another (linear flow)
# CNN (Convolutional Neural Network): Learns hierarchical features from images
print(f"\nBuilding CNN Model...\n")
model = tf.keras.Sequential([
    
    # BLOCK 1: Initial feature extraction (low-level features: edges, colors)
    # Conv2D(32, (3, 3)): Create 32 feature maps with 3x3 filters
    #   - Each of 32 filters slides over image to detect patterns
    #   - 3x3 = size of sliding window
    #   - 32 = number of such filters (detecting 32 different patterns)
    # activation="relu": Use ReLU = max(0, x)
    #   - Adds nonlinearity (neural networks learn non-linear functions)
    #   - Zeros out negative values (sparse representation)
    # input_shape=(IMG_SIZE, IMG_SIZE, 3): Input data format
    #   - 150x150 pixels (height × width)
    #   - 3 channels = RGB (Red, Green, Blue)
    tf.keras.layers.Conv2D(
        32,                                          # Number of filters
        (3, 3),                                      # Filter/kernel size
        activation="relu",                           # Activation function
        input_shape=(IMG_SIZE, IMG_SIZE, 3)         # Input format
    ),
    
    # MaxPooling2D(2, 2): Reduce spatial dimensions by 2x
    #   - Takes 2×2 window and outputs maximum value
    #   - Reduces from 150×150 to 75×75 (half the size)
    #   - Benefits:
    #     * Fewer parameters to train (faster)
    #     * Translation invariance (robust to small shifts)
    #     * Highlights strongest activations
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # BatchNormalization(): Normalize activations between layers
    #   - Normalizes: mean=0, std=1
    #   - Benefits:
    #     * Faster training (larger learning rates possible)
    #     * More stable (less sensitive to weight initialization)
    #     * Reduces internal covariate shift
    #   - Applied after pooling to stabilize gradient flow
    tf.keras.layers.BatchNormalization(),

    # BLOCK 2: Mid-level feature extraction (textures, shapes)
    # Conv2D(64, (3, 3)): Increase to 64 filters
    #   - Now extracting from already-extracted features
    #   - Can learn combinations of low-level features
    #   - More filters = more complex feature combinations
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),  # 75×75 → 37×37
    tf.keras.layers.BatchNormalization(),

    # BLOCK 3: Higher-level feature extraction
    # Conv2D(128, (3, 3)): Further increase to 128 filters
    #   - Very abstract features
    #   - Learned patterns from previous blocks
    #   - Examples: tumor regions, boundary patterns
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),  # 37×37 → 18×18
    tf.keras.layers.BatchNormalization(),

    # BLOCK 4: Most abstract features
    # Conv2D(256, (3, 3)): Final convolutional block
    #   - Highest-level features before flattening
    #   - Size: 18×18×256 feature maps
    tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),  # 18×18 → 9×9
    tf.keras.layers.BatchNormalization(),

    # Flatten: Convert 3D feature maps to 1D vector
    #   - Input shape: (9, 9, 256) = 20,736 values
    #   - Output shape: 20,736 values in 1D vector
    #   - Why: Dense layers expect 1D input (not 3D)
    tf.keras.layers.Flatten(),
    
    # Dense(256): Fully connected layer with 256 neurons
    #   - Every input neuron connects to all 256 neurons
    #   - activation="relu": ReLU nonlinearity
    #   - Learns combinations of extracted features
    #   - Learns patterns specific to tumor classification
    tf.keras.layers.Dense(256, activation="relu"),
    
    # Dropout(0.5): Random deactivation during training
    #   - Probability 0.5: each neuron has 50% chance to be disabled per sample
    #   - Why dropout:
    #     * Prevents overfitting (can't memorize training data)
    #     * Forces redundant learning (multiple paths to same output)
    #     * Acts as ensemble of 2^n models (exponential models in ensemble)
    #     * Reduces co-adaptation (neurons depending on each other)
    #   - At test time: all neurons active (use average of ensemble)
    #   - Mathematical effect: effectively trains multiple sub-networks
    tf.keras.layers.Dropout(0.5),
    
    # Dense(128): Intermediate layer (256 → 128 neurons)
    #   - Reduce dimensionality (extract most important 128 features)
    #   - Still learning high-level patterns
    tf.keras.layers.Dense(128, activation="relu"),
    
    # Dropout(0.3): More conservative dropout
    #   - 30% deactivation (less disruption near output)
    #   - Trade-off: prevent overfitting but avoid too much noise
    tf.keras.layers.Dropout(0.3),

    # Dense(4, activation="softmax"): Output layer
    #   - 4 neurons: one for each tumor type
    #     * Neuron 0: probability of Glioma
    #     * Neuron 1: probability of Meningioma
    #     * Neuron 2: probability of No Tumor
    #     * Neuron 3: probability of Pituitary
    #   - activation="softmax": Convert outputs to probability distribution
    #     * Formula: softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)
    #     * Properties:
    #       - All outputs in range [0, 1]
    #       - All outputs sum to 1.0 (valid probability distribution)
    #       - Interpretable as probabilities
    #     * Example output: [0.8, 0.1, 0.05, 0.05]
    #       - 80% Glioma, 10% Meningioma, 5% No Tumor, 5% Pituitary
    tf.keras.layers.Dense(4, activation="softmax")
])

print("Model Architecture:")
print("-" * 60)
# model.summary(): Print detailed architecture breakdown
#   Shows:
#   - Layer name and type
#   - Output shape (how dimensions change)
#   - Number of parameters (trainable weights)
#   - Total parameters = sum of all layer parameters
#   - Useful for:
#     * Understanding model complexity
#     * Identifying bottlenecks
#     * Estimating memory usage
#     * Debugging architecture issues
model.summary()
```

---

## Example 2: Compiling the Model

```python
# Compile the model with Adam optimizer and categorical crossentropy loss
# Compilation configures model for training (doesn't train yet, just setup)
# optimizer=Adam(learning_rate=0.001):
#   - Adam: Adaptive Moment Estimation optimizer
#   - Maintains moving averages of gradients and squared gradients
#   - Computes individual adaptive learning rates per parameter
#   - learning_rate=0.001 = 0.001 = 0.1%
#     * How much to change weights per gradient step
#     * Larger rate → faster training but may overshoot optimum
#     * Smaller rate → slower but more precise convergence
#     * 0.001 is typical for Adam (often 0.001 to 0.00001)
#   - Why Adam > SGD:
#     * Adapts learning rate per parameter
#     * Handles sparse gradients well
#     * Converges faster in practice
#     * Less sensitive to learning rate choice
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    
    # loss="categorical_crossentropy":
    #   - For multi-class classification (one true class per sample)
    #   - Measures difference between predicted and true probability distributions
    #   - Formula: loss = -sum(y_true[i] * log(y_pred[i]))
    #   - y_true: one-hot encoded (e.g., [1, 0, 0, 0] for Glioma)
    #   - y_pred: predicted probabilities (e.g., [0.8, 0.1, 0.05, 0.05])
    #   - Calculation:
    #     * loss = -(1*log(0.8) + 0*log(0.1) + 0*log(0.05) + 0*log(0.05))
    #     * loss = -log(0.8) ≈ 0.223
    #   - Properties:
    #     * Loss = 0 when predicted = true (perfect prediction)
    #     * Loss → ∞ when confidence in wrong class → 1
    #     * Penalizes confident wrong predictions heavily
    #   - Alternative losses:
    #     * binary_crossentropy: 2-class classification
    #     * sparse_categorical_crossentropy: integer labels (not one-hot)
    #     * mean_squared_error: regression problems
    loss="categorical_crossentropy",
    
    # metrics=["accuracy"]:
    #   - Metrics to track during training (not used for optimization)
    #   - Different from loss:
    #     * Loss: used for gradient computation and weight updates
    #     * Metrics: just for monitoring and interpretation
    #   - Accuracy = correct_predictions / total_predictions
    #     * Ranges from 0.0 (all wrong) to 1.0 (all correct)
    #     * More interpretable than loss value
    #   - Can track multiple metrics:
    #     * metrics=["accuracy", "precision", "recall", "auc"]
    #   - Computed on:
    #     * Training data: train metric
    #     * Validation data: val metric
    #   - Other useful metrics:
    #     * Precision: TP / (TP + FP) - when model predicts positive, how often correct
    #     * Recall: TP / (TP + FN) - of all actual positives, how many found
    #     * AUC: Area under ROC curve - separability of classes
    #     * F1: Harmonic mean of precision and recall
    metrics=["accuracy"]
)
```

---

## Example 3: Setting Up Training Callbacks

```python
# Configure callbacks: special actions to take during training
# Callbacks = functions that run at certain training events:
#   - After each epoch
#   - After each batch
#   - When validation improves/degrades
#   - Etc.

# CALLBACK 1: ModelCheckpoint
# Saves model weights to disk when validation metric improves
checkpoint = ModelCheckpoint(
    str(MODEL_PATH),                # WHERE: Save model to this file path
    save_best_only=True,            # WHAT: Only save if best so far
                                    # Example: If val_acc = [0.5, 0.6, 0.61, 0.60]
                                    # Saves only after epoch 2 (0.61 is best)
    monitor="val_accuracy",         # MONITOR: Track this metric
                                    # Alternatives: "val_loss", "accuracy", "loss"
    mode="max",                     # MODE: "max" = maximize (higher is better)
                                    # Alternatives: "min" = minimize (lower is better)
    verbose=1                       # PRINT: Print message when saving
)
# Why this is important:
#   - Training runs for EPOCHS iterations (e.g., 10 epochs fixed)
#   - But best model might be at epoch 5 (not epoch 10)
#   - Without this callback: last model is saved (might be worse)
#   - With this callback: best model is saved (best performance)
# Example scenario:
#   Epoch 1: val_acc = 0.75 → SAVE (first, best so far)
#   Epoch 2: val_acc = 0.80 → SAVE (better than 0.75)
#   Epoch 3: val_acc = 0.78 → SKIP (worse than 0.80)
#   Epoch 4: val_acc = 0.79 → SKIP (worse than 0.80)
#   Result: Saved model from epoch 2 (best validation accuracy)

# CALLBACK 2: EarlyStopping
# Stops training automatically if validation metric stops improving
early_stop = EarlyStopping(
    monitor="val_loss",             # MONITOR: Track validation loss
                                    # (loss should decrease = improve)
    patience=3,                     # PATIENCE: Wait 3 epochs for improvement
                                    # Example: If val_loss = [0.5, 0.48, 0.47, 0.475, 0.476, 0.478]
                                    # Epoch 1: 0.5 (best, wait counter = 0)
                                    # Epoch 2: 0.48 (better, wait counter = 0)
                                    # Epoch 3: 0.47 (better, wait counter = 0)
                                    # Epoch 4: 0.475 (WORSE, wait counter = 1)
                                    # Epoch 5: 0.476 (WORSE, wait counter = 2)
                                    # Epoch 6: 0.478 (WORSE, wait counter = 3 = PATIENCE)
                                    # → STOP training (waited 3 epochs without improvement)
    restore_best_weights=True,      # RESTORE: Use weights from best epoch
                                    # Without this: uses weights from stopped epoch
                                    # With this: uses weights from best epoch
    verbose=1                       # PRINT: Print stopping message
)
# Why this is important:
#   - Prevents wasting computation time training too long
#   - Stops when model clearly stops learning
#   - patience=3: allows 3 epochs variance (in case of random fluctuations)
#   - Saves time, especially for large models
# Trade-off:
#   - Too low patience: stops too early (before reaching optimum)
#   - Too high patience: trains too long (wastes computation)
#   - 3-5 typical for medium datasets
```

---

## Example 4: Training Loop

```python
print(f"\n{'='*60}")
print("Training Model...")
print(f"{'='*60}\n")

# Start the training loop - this is where actual learning happens
# model.fit(): Run training and return History object
# History: tracks all metrics (loss, accuracy, val_loss, val_accuracy) per epoch
history = model.fit(
    train_ds,                       # TRAINING DATA: batches of labeled images
                                    # Type: ImageDataGenerator flow
                                    # Yields: (batch_of_images, batch_of_labels)
                                    # Size: batches of 32 images
    validation_data=test_ds,        # VALIDATION DATA: for computing val_loss, val_accuracy
                                    # Evaluated after each epoch
                                    # NOTE: Using test set (not ideal, should use separate val set)
                                    # Better: train/val split from training data, keep test separate
    epochs=EPOCHS,                  # EPOCHS: 10 complete passes through training data
                                    # Epoch 1: See all 5712 images once, update weights 178 times (5712/32)
                                    # Epoch 2: See all images again (with better weights), update 178 times
                                    # ... repeat 10 times
    callbacks=[checkpoint, early_stop],  # CALLBACKS: Actions to take per epoch
                                    # checkpoint: save best model
                                    # early_stop: stop if val_loss doesn't improve
    verbose=1                       # VERBOSE: Print progress bar each epoch
)

# What happens INSIDE model.fit():
# For each epoch:
#   1. For each batch in training data:
#      a. Forward pass: compute predictions
#         - Feed images through network
#         - Output: probabilities for each class
#      b. Compute loss: compare predictions to true labels
#         - loss = -sum(y_true * log(y_pred))
#         - Higher loss = worse predictions
#      c. Backward pass (backpropagation):
#         - Compute gradients (how much each weight affects loss)
#         - Use chain rule to propagate error backward
#      d. Update weights:
#         - new_weight = old_weight - learning_rate * gradient
#         - Gradient > 0: decrease weight
#         - Gradient < 0: increase weight
#   2. After all batches (one epoch done):
#      a. Run callbacks (checkpoint, early_stop)
#      b. Evaluate on validation data (compute val_loss, val_accuracy)
#      c. Print progress (Epoch 1/10, loss: 2.1, acc: 0.4, val_loss: 1.8, val_acc: 0.5)
#
# Data flow example:
# - Training set: 5712 images
# - Batch size: 32
# - Batches per epoch: 5712 / 32 = 179 batches
# - Epoch 1:
#   * Process batches 1-179
#   * After batch 1: weights update (gradient step 1)
#   * After batch 2: weights update (gradient step 2)
#   * ...
#   * After batch 179: weights update (gradient step 179)
#   * Calculate val_loss, val_accuracy on full test set
#   * Run callbacks (save if best, check if stop)
# - Epoch 2: repeat with updated weights
```

---

## Example 5: Evaluating Metrics

```python
def evaluate_dataset(model, data_dir, dataset_name, class_names):
    """
    FUNCTION: evaluate_dataset
    ===========================
    Evaluate model on a complete dataset and return comprehensive metrics.
    
    PARAMETERS:
      model: Keras model - trained model to evaluate
      data_dir: Path - directory with labeled images
      dataset_name: str - name of dataset ("training" or "test")
      class_names: list - tumor class names
    
    RETURNS:
      metrics: dict - computed metrics (accuracy, precision, etc.)
      y_true: array - true class labels [0, 1, 2, 1, 0, ...]
      y_pred: array - predicted class labels [0, 1, 1, 1, 0, ...]
      y_pred_proba: array - prediction probabilities for each class
    """
    
    print(f"\n{'='*70}")
    print(f"STEP {dataset_name.upper()}: Loading {dataset_name} Data")
    print(f"{'='*70}")
    
    # Create generator to load images from disk
    data_gen = ImageDataGenerator(rescale=1/255.0)  # Normalize to [0,1]
    data_ds = data_gen.flow_from_directory(
        str(data_dir),                          # Directory path
        target_size=(IMG_SIZE, IMG_SIZE),       # Resize to 150x150
        class_mode="categorical",               # One-hot encoded labels
        shuffle=False,                          # Keep order (important for matching with true labels)
        batch_size=BATCH_SIZE                   # 32 images per batch
    )
    
    print(f"  Samples: {data_ds.samples}")       # Total images
    print(f"  Classes: {sorted(data_ds.class_indices.keys())}")  # Class names
    
    print(f"\n{'='*70}")
    print(f"STEP {dataset_name.upper()}: Running Predictions")
    print(f"{'='*70}\n")
    
    # Run predictions on all data
    # model.predict(): Feed all images through network
    # Returns: predictions.shape = (num_images, 4)
    # Example: [[0.8, 0.1, 0.05, 0.05],   # Image 1: 80% Glioma
    #           [0.1, 0.7, 0.1, 0.1],     # Image 2: 70% Meningioma
    #           [0.3, 0.3, 0.3, 0.1],     # Image 3: 30% each (uncertain)
    #           ...]
    predictions = model.predict(data_ds, verbose=1)  # verbose=1 shows progress bar
    
    # Get predicted class for each image (highest probability)
    # np.argmax(axis=1): find index of maximum value for each image
    # Example: [0.8, 0.1, 0.05, 0.05] → argmax = 0 (index of 0.8)
    y_pred = np.argmax(predictions, axis=1)  # [0, 1, 3, 0, 2, ...]
    
    # Get true class labels
    # data_ds.classes: true class for each image (in same order as predictions)
    y_true = data_ds.classes                  # [0, 1, 3, 0, 2, ...]
    
    # Store prediction probabilities
    y_pred_proba = predictions                # [[0.8, 0.1, ...], [0.1, 0.7, ...], ...]
    
    print(f"\n{'='*70}")
    print(f"STEP {dataset_name.upper()}: Computing Metrics")
    print(f"{'='*70}")
    
    # Calculate comprehensive metrics
    metrics = {}
    
    # METRIC 1: Accuracy = (TP + TN) / (TP + TN + FP + FN)
    #   - "TP" = True Positive = correctly predicted Glioma as Glioma
    #   - "TN" = True Negative = correctly predicted non-Glioma as non-Glioma
    #   - "FP" = False Positive = incorrectly predicted non-Glioma as Glioma
    #   - "FN" = False Negative = incorrectly predicted Glioma as non-Glioma
    #   - Formula: correct predictions / total predictions
    #   - Range: 0.0 (all wrong) to 1.0 (all correct)
    #   - Example: 1000 predictions, 850 correct → accuracy = 0.85 = 85%
    #   - Limitation: doesn't account for class imbalance
    #     * If 99% of images are "no tumor", model that always predicts "no tumor" has 99% accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # METRIC 2-3: Precision (Macro & Weighted)
    #   - For each class i: Precision_i = TP_i / (TP_i + FP_i)
    #   - "How many of my positive predictions were correct?"
    #   - Example for Glioma:
    #     * Model predicted Glioma 100 times
    #     * 80 were actually Glioma
    #     * Precision = 80 / 100 = 0.80
    #   - Macro: average precision across classes (treats each class equally)
    #   - Weighted: weighted average (accounts for class size)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # METRIC 4-5: Recall (Macro & Weighted)
    #   - For each class i: Recall_i = TP_i / (TP_i + FN_i)
    #   - "Of all actual positives, how many did I find?"
    #   - Example for Glioma:
    #     * Actually 120 Gliomas in dataset
    #     * Model found 80 of them
    #     * Recall = 80 / 120 = 0.67
    #   - High recall = catch most positive cases (few false negatives)
    #   - High precision = few false alarms (few false positives)
    #   - Trade-off: increasing precision usually decreases recall
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # METRIC 6-7: F1-Score (Macro & Weighted)
    #   - For each class: F1 = 2 * (Precision * Recall) / (Precision + Recall)
    #   - Harmonic mean of precision and recall
    #   - Why harmonic mean? Equally penalizes low precision or low recall
    #   - Example: Precision=0.8, Recall=0.6
    #     * Arithmetic mean = (0.8 + 0.6) / 2 = 0.7
    #     * Harmonic mean = 2 * (0.8 * 0.6) / (0.8 + 0.6) = 0.686
    #     * Harmonic is lower (penalizes imbalance)
    #   - Range: 0.0 (worst) to 1.0 (perfect)
    #   - Better than accuracy when you care about both precision and recall
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # METRIC 8: Cohen's Kappa
    #   - Measures inter-rater agreement (how much better than chance)
    #   - Formula: kappa = (accuracy - expected_accuracy) / (1 - expected_accuracy)
    #   - Interpretation:
    #     * 0.00-0.20: Slight agreement
    #     * 0.20-0.40: Fair agreement
    #     * 0.40-0.60: Moderate agreement
    #     * 0.60-0.80: Substantial agreement
    #     * 0.80-1.00: Perfect agreement
    #   - Better than accuracy for imbalanced datasets
    #   - Example: If everyone is "no tumor" (99% accuracy)
    #     * Expected accuracy (by chance) = 0.99
    #     * Kappa might be only 0.1 (not much better than chance)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # METRIC 9: Matthews Correlation Coefficient
    #   - Correlation coefficient between predictions and actual values
    #   - Range: -1.0 (perfect wrong) to +1.0 (perfect right)
    #   - 0.0 = random guessing
    #   - Better than accuracy for imbalanced datasets
    #   - More stable than other metrics for rare classes
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    # METRIC 10-12: Per-Class Metrics
    # Compute precision, recall, F1 for each class individually
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # METRIC 13: Confusion Matrix
    # Shows where model makes mistakes
    # Rows = True labels, Columns = Predicted labels
    # Example (4 classes):
    #                 Pred_Glioma  Pred_Menin  Pred_NoTumor  Pred_Pituitary
    # True_Glioma        290         1            0              9
    # True_Menin         149        52           50             55
    # True_NoTumor         6         7           376            16
    # True_Pituitary       7         3            0             290
    # 
    # Diagonal = correct predictions
    # Off-diagonal = mistakes
    #   - (0,1): 1 Glioma misclassified as Meningioma
    #   - (1,0): 149 Meningioma misclassified as Glioma (big problem!)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # METRIC 14: Classification Report
    # Summary of all metrics per class
    # Includes: precision, recall, f1, support (number of true samples)
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    
    return metrics, y_true, y_pred, y_pred_proba
```

---

## Example 6: Transfer Learning (VGG16)

```python
def build_enhanced_model():
    """
    Build model using VGG16 pre-trained on ImageNet.
    
    VGG16 Architecture:
      - Trained on 1.4 million ImageNet images
      - 16 layers (13 convolutional + 3 fully connected)
      - Learned general image features:
        * Edges, corners, textures (early layers)
        * Shapes, objects (middle layers)
        * Complete objects (later layers)
      - Can transfer these learned features to new task (brain tumor classification)
    
    Why Transfer Learning Helps:
      - Baseline model: trained from scratch (random weights)
      - VGG16 model: starts with learned weights (warm start)
      - With limited data: VGG16 prevents overfitting
      - With limited time: VGG16 trains faster
      - Better accuracy: leverages knowledge from 1.4M images
    """
    
    # Load VGG16 architecture with pre-trained ImageNet weights
    # include_top=False: Don't load final classification layer
    #   - VGG16's final layer is 1000 classes (ImageNet)
    #   - We need 4 classes (brain tumors)
    #   - So we remove top layer and add our own
    # weights='imagenet': Load pre-trained weights
    #   - Instead of random initialization
    #   - Contains learned features from 1.4M images
    base_model = VGG16(
        include_top=False,                      # Remove classification layer
        weights='imagenet',                     # Pre-trained weights
        input_shape=(IMG_SIZE, IMG_SIZE, 3)    # Our image size
    )
    
    # Freeze base model weights (STAGE 1: Feature Extraction)
    # base_model.trainable = False means:
    #   - Don't update weights in VGG16
    #   - Only train the custom layers we'll add
    #   - VGG16 acts as feature extractor only
    # Why freeze?
    #   - VGG16 weights are already good (trained on 1.4M images)
    #   - Don't want to break them with small data (overfit)
    #   - Faster training (fewer weights to update)
    #   - Reduces computational cost
    base_model.trainable = False
    
    print(f"[OK] Loaded VGG16 base model (frozen)")
    print(f"     Base model has {len(base_model.layers)} layers")
    
    # Build custom classification head
    # Input: feature maps from VGG16 (9x9x512)
    # Output: 4 class probabilities
    
    x = base_model.output                       # Start with VGG16 output
    
    # GlobalAveragePooling2D: Convert feature maps to vector
    # Input shape: (9, 9, 512) = 41,472 values
    # Output shape: 512 values
    # What it does: average each feature channel
    #   - Feature map 1: average all 81 values (9×9) → 1 value
    #   - Feature map 2: average all 81 values → 1 value
    #   - ...
    #   - Feature map 512: average all 81 values → 1 value
    # Why better than Flatten?
    #   - Flatten would create 41,472 weights (huge, prone to overfit)
    #   - GlobalAveragePooling creates 512 features (fewer, more generalizable)
    #   - More robust to spatial shifts in images
    x = GlobalAveragePooling2D()(x)             # Shape: (512,)
    
    # BatchNormalization: Normalize activations
    #   - Stabilizes training
    #   - Allows higher learning rates
    #   - Reduces gradient issues
    x = BatchNormalization()(x)
    
    # Dense(512): Learn class-specific patterns
    #   - 512 inputs → 512 neurons
    #   - activation='relu': nonlinearity
    #   - Learns combinations of VGG16 features
    x = Dense(512, activation='relu')(x)
    
    # Dropout(0.5): Prevent overfitting
    #   - 50% of neurons disabled during training
    #   - Encourages redundant learning
    x = Dropout(0.5)(x)
    
    # BatchNormalization: Stabilize gradients
    x = BatchNormalization()(x)
    
    # Dense(256): Further refinement
    #   - 512 → 256 (reduce dimensionality)
    #   - Extract most important 256 features
    x = Dense(256, activation='relu')(x)
    
    # Dropout(0.4): Still prevent overfitting
    #   - Less aggressive than before (closer to output)
    x = Dropout(0.4)(x)
    
    # Dense(4, softmax): Output layer
    #   - 4 tumor types
    #   - Softmax: output probabilities
    x = Dense(4, activation='softmax')(x)
    
    # Create model object
    model = Model(inputs=base_model.input, outputs=x)
    
    print(f"[OK] Custom classification head added")
    print(f"     Total parameters: {model.count_params():,}")
    
    return model

# STAGE 1: Feature Extraction (freeze VGG16)
# Train for 15 epochs with frozen base model
# Only train: GlobalAveragePooling2D → BatchNorm → Dense(512) → ... → Dense(4)

# STAGE 2: Fine-tuning (unfreeze last layers)
# After Stage 1, unfreeze last 4 layers of VGG16
# This allows VGG16 to slightly adapt to brain tumor task
# BUT: use lower learning rate (0.0001 vs 0.001)
# Why lower learning rate for fine-tuning?
#   - VGG16 weights are already good
#   - Don't want big updates that break them
#   - Want small adjustments (fine-tune)
```

---

## Key Takeaways

1. **Every line of code does something specific** - comments explain the "what" and "why"
2. **Neural networks have layers** - each layer transforms data and learns features
3. **Training updates weights** - based on gradients (direction of improvement)
4. **Callbacks control training** - save best model, stop early, adjust learning rate
5. **Metrics evaluate performance** - accuracy, precision, recall, F1 measure different aspects
6. **Transfer learning reuses knowledge** - pre-trained models boost performance with less data

---

*Created: December 4, 2024*  
*Project: Brain Tumor Classification*  
*For: Understanding and Learning Deep Learning Concepts*
