import cv2
import numpy as np

def preprocess_image(image_data: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104, 117, 123), swapRB=False)
    return blob