from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from io import BytesIO

# Load model once at startup
validator_model = MobileNetV2(weights='imagenet')

def is_garbage_image(image_bytes):
    try:
        img = Image.open(image_bytes).convert('RGB').resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = validator_model.predict(img_array, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]
        
        label = decoded[1]
        confidence = decoded[2]

        print("label: ", label)
        print("confidence: ", confidence)

        # If it's confidently ANY known ImageNet object → reject
        if confidence > 0.35:
            return True, label
        
        return False, label
        
    except Exception:
        return True, "corrupted_file"