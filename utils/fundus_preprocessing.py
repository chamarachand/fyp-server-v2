import cv2
import numpy as np
import io
from typing import Union
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 380

def preprocess_fundus(image_input: Union[bytes, io.BytesIO]) -> np.ndarray:
    """Mirrors the exact training and dataset generation pipeline."""
    
    # Extract bytes if a BytesIO object is passed
    if isinstance(image_input, io.BytesIO):
        image_bytes = image_input.read()
    else:
        image_bytes = image_input
    
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image.")

    # 1. Subtle black background crop
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad_x = int(w * 0.02)
        pad_y = int(h * 0.02)
        x = max(x - pad_x, 0)
        y = max(y - pad_y, 0)
        w = min(w + 2 * pad_x, img.shape[1] - x)
        h = min(h + 2 * pad_y, img.shape[0] - y)
        img = img[y:y+h, x:x+w]

    # 2. Green channel extraction
    green = img[:, :, 1]
    img = cv2.merge([green, green, green])

    # 3. CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 4. Resize with padding
    h, w = img.shape[:2]
    scale = IMG_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    top = (IMG_SIZE - new_h) // 2
    bottom = IMG_SIZE - new_h - top
    left = (IMG_SIZE - new_w) // 2
    right = IMG_SIZE - new_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

    # 5. Format for EfficientNet Input
    img_array = np.array(padded, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension -> (1, 380, 380, 3)
    img_array = preprocess_input(img_array)       # Apply Keras EfficientNet preprocessing
    
    return img_array