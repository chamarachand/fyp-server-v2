import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 380

def subtle_crop(image, padding_ratio=0.02):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x = max(x - pad_x, 0)
    y = max(y - pad_y, 0)
    w = min(w + 2 * pad_x, image.shape[1] - x)
    h = min(h + 2 * pad_y, image.shape[0] - y)

    return image[y:y+h, x:x+w]

def extract_green_channel(image):
    green = image[:, :, 1]
    return cv2.merge([green, green, green])

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def resize_with_padding(image, target_size=IMG_SIZE):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h))

    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

def normalize_image(image):
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (image - mean) / std

def preprocess_upload(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    img = np.array(img)

    img = subtle_crop(img)
    img = extract_green_channel(img)
    img = apply_clahe(img)
    img = resize_with_padding(img)
    img = normalize_image(img)

    return np.expand_dims(img, axis=0)

def preprocess_local_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)

    img = subtle_crop(img)
    img = extract_green_channel(img)
    img = apply_clahe(img)
    img = resize_with_padding(img)
    img = normalize_image(img)

    return np.expand_dims(img, axis=0)