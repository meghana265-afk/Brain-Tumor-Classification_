"""
preprocess.py
Line-by-line commented image loading and preprocessing utilities.

These helpers are optional for workflows that want to load images into
NumPy arrays (e.g., for custom training loops or analysis). Keras
ImageDataGenerator is used elsewhere for convenience.
"""

# OpenCV for image loading and resizing
import cv2

# OS for directory traversal
import os

# NumPy for array handling
import numpy as np

# Standard image size used throughout the project
IMG_SIZE = 150


def load_and_preprocess(path):
    """Load an image from `path`, resize to `IMG_SIZE`, and normalize to [0,1].

    Raises a ValueError if the image cannot be read. This function is useful
    for ad-hoc preprocessing outside the Keras ImageDataGenerator pipeline.
    """

    # Read the image with OpenCV
    img = cv2.imread(path)
    if img is None:
        # Clear error for debugging if file is corrupt or path is wrong
        raise ValueError(f"Could not read image: {path}")

    # Resize to the model input resolution
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values to the [0,1] range expected by the model
    img = img / 255.0

    return img


def load_dataset(folder):
    """Load images and labels from a directory structured by class.

    The directory should contain subfolders for each class. Returns a tuple:
    (images_array, labels_array, class_list)
    """

    images = []
    labels = []

    # Discover class names by listing subdirectories and sorting for stable order
    classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])

    # Iterate each class folder and collect images
    for idx, c in enumerate(classes):
        class_path = os.path.join(folder, c)
        for fname in os.listdir(class_path):
            # Filter common image extensions
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = load_and_preprocess(os.path.join(class_path, fname))
                images.append(img)
                labels.append(idx)

    # Convert lists to NumPy arrays for consumption by ML code
    return np.array(images), np.array(labels), classes
