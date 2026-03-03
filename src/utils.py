"""
utils.py — Helper utilities for Diabetic Retinopathy Detection
Ben Graham preprocessing, GradCAM, seed setting, and misc helpers.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set seeds for reproducibility across numpy, TF, and Python."""
    import os, random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Ben Graham Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crop dark circular borders from retinal fundus images.
    Works on both grayscale and RGB images.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]
    return img


def ben_graham_preprocess(img: np.ndarray, img_size: int = 224, sigma: int = 10) -> np.ndarray:
    """
    Ben Graham retinal image preprocessing.
    
    Process:
      1. Crop black borders
      2. Resize to target size
      3. Subtract Gaussian-blurred version (enhances microstructures)
      4. Apply circular mask
    
    Args:
        img:      RGB image as numpy array (H, W, 3)
        img_size: Target size to resize to (square)
        sigma:    Gaussian blur sigma — controls enhancement strength
    
    Returns:
        Preprocessed image as uint8 numpy array (img_size, img_size, 3)
    
    Reference:
        Ben Graham, 1st place solution, Kaggle DR 2015
        https://kaggle.com/competitions/diabetic-retinopathy-detection/discussion/15801
    """
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)
    b = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(b, (img_size // 2, img_size // 2),
               int(img_size * 0.9 // 2), (1, 1, 1), -1, 8, 0)
    img = img * b + 128 * (1 - b)
    return img.astype(np.uint8)


def load_image(path: str, img_size: int = 224, preprocess: bool = True) -> np.ndarray:
    """Load, optionally preprocess, and normalize an image for inference."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if preprocess:
        img = ben_graham_preprocess(img, img_size)
    else:
        img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────────────────────────────────────

def get_gradcam_heatmap(
    model: keras.Model,
    img_array: np.ndarray,
    layer_name: str = 'top_conv'
) -> tuple[np.ndarray, int]:
    """
    Generate a GradCAM heatmap for a single image.
    
    Args:
        model:      Trained Keras model
        img_array:  Normalized image array (H, W, 3) — no batch dimension
        layer_name: Name of the last convolutional layer to use
    
    Returns:
        (heatmap, predicted_class_index) tuple
        heatmap is normalized to [0, 1]
    """
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array[np.newaxis], training=False)
        pred_index   = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads        = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay GradCAM heatmap on the original image.
    
    Args:
        original_img: RGB image (H, W, 3) as uint8
        heatmap:      Normalized heatmap [0, 1] from get_gradcam_heatmap
        alpha:        Blending weight for heatmap (0=original, 1=heatmap only)
    
    Returns:
        Superimposed RGB image as uint8
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
