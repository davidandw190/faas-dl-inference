import cv2
import numpy as np

def preprocess(image_data: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img_data = np.expand_dims(img, axis=(0, 1))
    return img_data.astype(np.float32)