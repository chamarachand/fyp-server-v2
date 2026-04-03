# utils/preprocessing.py
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224

def fix_inverted_image(image: np.ndarray) -> np.ndarray:
    if np.mean(image) > 127:
        return cv2.bitwise_not(image)
    return image

def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def resize_with_padding(image: np.ndarray, target_size=IMG_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded

def preprocess_oct(image_file):
    # Load grayscale
    if isinstance(image_file, str):
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    else:
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Invalid OCT image")

    # Fix inversion
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Resize + pad
    h, w = img.shape
    scale = IMG_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    top = (IMG_SIZE - new_h) // 2
    bottom = IMG_SIZE - new_h - top
    left = (IMG_SIZE - new_w) // 2
    right = IMG_SIZE - new_w - left

    img = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=0
    )

    # Convert grayscale → 3 channels
    img = np.stack([img, img, img], axis=-1)

    # Convert to float and batch
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    # Apply same preprocessing as training
    img = preprocess_input(img)

    return img