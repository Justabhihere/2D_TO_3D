import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def get_mask(model, img):
    pred = model.predict(img)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)
    return mask
