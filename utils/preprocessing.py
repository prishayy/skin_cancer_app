# utils/preprocessing.py
"""
Deployment preprocessing pipeline using the winning techniques selected
from the preprocessing evaluation experiment:
  - Hair Removal    : Telea
  - Noise Filtering : Median (5x5)
  - Contrast Enh.   : Gamma 2.0
  - Resizing        : Bilinear
"""

import cv2
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------
#  Winning preprocessing functions (copied verbatim from PreprocessingFunctions)
# ----------------------------------------------------------------------

def hair_telea(img_bgr: np.ndarray) -> np.ndarray:
    """
    Telea inpainting hair removal.
    img_bgr: BGR image (numpy array)
    returns: BGR image
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        _, mask = cv2.threshold(bhat, 8, 255, cv2.THRESH_BINARY)
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kd, iterations=1)
        return cv2.inpaint(img_bgr, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    except Exception:
        return img_bgr.copy()


def noise_median5(img_bgr: np.ndarray) -> np.ndarray:
    """
    Median filter with kernel size 5x5.
    """
    try:
        return cv2.medianBlur(img_bgr, 5)
    except Exception:
        return img_bgr.copy()


def contrast_gamma20(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gamma correction with gamma = 2.0.
    """
    try:
        table = (np.linspace(0, 1, 256) ** (1.0 / 2.0) * 255).astype(np.uint8)
        return cv2.LUT(img_bgr, table)
    except Exception:
        return img_bgr.copy()


def resize_bilinear(img_bgr: np.ndarray, size=(224, 224)) -> np.ndarray:
    """
    Bilinear interpolation resize to 224x224.
    """
    try:
        return cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR)
    except Exception:
        return img_bgr.copy()


# ----------------------------------------------------------------------
#  Full pipeline wrapper
# ----------------------------------------------------------------------

def preprocess_pipeline(pil_image: Image.Image) -> Image.Image:
    """
    Apply the complete winning preprocessing pipeline to a PIL image.
    Steps: hair removal → noise filtering → contrast enhancement → resizing.
    Returns a PIL Image (RGB) ready for segmentation and classification.
    """
    # Convert PIL (RGB) to numpy array (RGB) then to BGR (OpenCV convention)
    img_rgb = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Apply each winning technique in order
    img_bgr = hair_telea(img_bgr)
    img_bgr = noise_median5(img_bgr)
    img_bgr = contrast_gamma20(img_bgr)
    img_bgr = resize_bilinear(img_bgr)

    # Convert back to RGB and return as PIL Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb) 
