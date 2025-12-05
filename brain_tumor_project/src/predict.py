"""
predict.py
Enhanced prediction script supporting both baseline and enhanced models.

Usage:
  python predict.py <image_path>              # Use baseline model
  python predict.py <image_path> --enhanced   # Use enhanced model
  python predict.py <image_path> --both       # Compare both models
"""

# Load TensorFlow to deserialize the saved model
import tensorflow as tf

# OpenCV (cv2) for image loading and resizing
import cv2

# NumPy for array manipulation
import numpy as np

# Standard libraries for argument parsing and file checks
import sys
import os
import argparse

# Import canonical configuration (paths, sizes, class names)
sys.path.insert(0, os.path.dirname(__file__))
from config import IMG_SIZE, MODEL_PATH, CLASS_NAMES

# Model paths
BASELINE_MODEL_PATH = MODEL_PATH
ENHANCED_MODEL_PATH = MODEL_PATH.parent / 'best_enhanced_model.h5'

# Global models (loaded on demand)
baseline_model = None
enhanced_model = None

def load_models(use_baseline=True, use_enhanced=False):
    """Load requested models."""
    global baseline_model, enhanced_model
    
    if use_baseline and baseline_model is None:
        if not BASELINE_MODEL_PATH.exists():
            print(f"[ERROR] Baseline model not found at: {BASELINE_MODEL_PATH}")
            print("Please run train_model.py first!")
            return False
        try:
            baseline_model = tf.keras.models.load_model(str(BASELINE_MODEL_PATH))
            print(f"[OK] Baseline model loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load baseline model: {e}")
            return False
    
    if use_enhanced and enhanced_model is None:
        if not ENHANCED_MODEL_PATH.exists():
            print(f"[ERROR] Enhanced model not found at: {ENHANCED_MODEL_PATH}")
            print("Please run train_model_enhanced.py first!")
            return False
        try:
            enhanced_model = tf.keras.models.load_model(str(ENHANCED_MODEL_PATH))
            print(f"[OK] Enhanced model loaded")
        except Exception as e:
            print(f"[ERROR] Failed to load enhanced model: {e}")
            return False
    
    return True


def preprocess_image(path):
    """Load and preprocess image for prediction."""
    # Validate file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # Read image with OpenCV
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    # Resize to model input size and normalize to [0,1]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    # Add batch dimension (1, H, W, C)
    img = np.expand_dims(img, axis=0)
    
    return img


def predict_with_model(model, img_array, model_name="Model"):
    """Run prediction and return results."""
    pred = model.predict(img_array, verbose=0)[0]
    class_idx = np.argmax(pred)
    confidence = pred[class_idx]
    
    return {
        'class_idx': class_idx,
        'class_name': CLASS_NAMES[class_idx],
        'confidence': confidence,
        'probabilities': pred
    }


def predict_image(path, use_enhanced=False, compare=False):
    """Predict tumor class for a single image.

    Args:
        path: Path to image file
        use_enhanced: If True, use enhanced model; otherwise use baseline
        compare: If True, compare both models side-by-side
    """
    print("\n" + "="*70)
    print("BRAIN TUMOR CLASSIFICATION - PREDICTION")
    print("="*70)
    
    # Load requested models
    if compare:
        if not load_models(use_baseline=True, use_enhanced=True):
            return
    elif use_enhanced:
        if not load_models(use_baseline=False, use_enhanced=True):
            return
    else:
        if not load_models(use_baseline=True, use_enhanced=False):
            return
    
    # Preprocess image
    print(f"\n[STEP 1] Loading and preprocessing image")
    print(f"         Image: {path}")
    try:
        img_array = preprocess_image(path)
        print(f"[OK] Image preprocessed (shape: {img_array.shape})")
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Make predictions
    if compare:
        print(f"\n[STEP 2] Running predictions with BOTH models")
        print("="*70)
        
        # Baseline prediction
        print("\nBASELINE MODEL:")
        baseline_result = predict_with_model(baseline_model, img_array, "Baseline")
        print(f"  Predicted class: {baseline_result['class_name']}")
        print(f"  Confidence:      {baseline_result['confidence']:.4f} ({baseline_result['confidence']*100:.2f}%)")
        
        # Enhanced prediction
        print("\nENHANCED MODEL:")
        enhanced_result = predict_with_model(enhanced_model, img_array, "Enhanced")
        print(f"  Predicted class: {enhanced_result['class_name']}")
        print(f"  Confidence:      {enhanced_result['confidence']:.4f} ({enhanced_result['confidence']*100:.2f}%)")
        
        # Comparison
        print("\nCOMPARISON:")
        if baseline_result['class_name'] == enhanced_result['class_name']:
            print(f"  [OK] Both models agree: {baseline_result['class_name']}")
            conf_diff = enhanced_result['confidence'] - baseline_result['confidence']
            print(f"  Confidence difference: {conf_diff:+.4f} ({conf_diff*100:+.2f}%)")
        else:
            print(f"  [WARNING] Models disagree!")
            print(f"  Baseline: {baseline_result['class_name']} ({baseline_result['confidence']:.2%})")
            print(f"  Enhanced: {enhanced_result['class_name']} ({enhanced_result['confidence']:.2%})")
        
        # Detailed probabilities
        print("\nDETAILED PROBABILITIES:")
        print(f"{'Class':<15} | {'Baseline':<12} | {'Enhanced':<12} | {'Difference':<12}")
        print("-"*60)
        for i, class_name in enumerate(CLASS_NAMES):
            baseline_prob = baseline_result['probabilities'][i]
            enhanced_prob = enhanced_result['probabilities'][i]
            diff = enhanced_prob - baseline_prob
            print(f"{class_name:<15} | {baseline_prob:<12.4f} | {enhanced_prob:<12.4f} | {diff:+.4f}")
    
    else:
        # Single model prediction
        model_type = "Enhanced" if use_enhanced else "Baseline"
        model = enhanced_model if use_enhanced else baseline_model
        
        print(f"\n[STEP 2] Running prediction with {model_type} model")
        print("="*70)
        
        result = predict_with_model(model, img_array, model_type)
        
        print(f"\nPREDICTION RESULTS:")
        print(f"  Predicted class: {result['class_name']}")
        print(f"  Confidence:      {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        
        print(f"\nALL CLASS PROBABILITIES:")
        for i, class_name in enumerate(CLASS_NAMES):
            prob = result['probabilities'][i]
            bar = '#' * int(prob * 40)
            print(f"  {class_name:<15} {prob:.4f} ({prob*100:>6.2f}%) {bar}")
    
    print("\n" + "="*70)


def find_test_image():
    """Find the first available test image from the Testing directory."""
    test_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Testing')
    
    # Search for images in each class subdirectory
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            # Get all image files
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                image_path = os.path.join(class_dir, images[0])
                return image_path, class_name
    
    return None, None


if __name__ == "__main__":
    # Parse command line arguments (make image_path optional)
    parser = argparse.ArgumentParser(
        description='Brain Tumor Classification - Prediction',
        epilog='Examples:\n'
               '  python predict.py                           # Uses first test image\n'
               '  python predict.py image.jpg                 # Use baseline model\n'
               '  python predict.py image.jpg --enhanced       # Use enhanced model\n'
               '  python predict.py image.jpg --both           # Compare both models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('image_path', nargs='?', default=None, help='Path to the image file (optional)')
    parser.add_argument('--enhanced', action='store_true', 
                       help='Use enhanced model (transfer learning)')
    parser.add_argument('--both', action='store_true',
                       help='Compare both baseline and enhanced models')
    
    args = parser.parse_args()
    
    # If no image provided, find the first test image
    if args.image_path is None:
        image_path, found_class = find_test_image()
        if image_path is None:
            print("[ERROR] No test image found in Testing/ directory")
            print("Please provide an image path or ensure Testing/ directory exists")
            sys.exit(1)
        args.image_path = image_path
        print(f"[AUTO] Using test image from class: {found_class}")
    
    # Run prediction
    if args.both:
        predict_image(args.image_path, use_enhanced=False, compare=True)
    elif args.enhanced:
        predict_image(args.image_path, use_enhanced=True, compare=False)
    else:
        predict_image(args.image_path, use_enhanced=False, compare=False)
