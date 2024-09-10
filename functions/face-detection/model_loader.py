import onnxruntime as ort
from logger import logger
from config import MODEL_PATH

def load_model():
    try:
        face_detector = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        logger.info("Face detection model loaded successfully.")
        return face_detector
    except Exception as e:
        logger.error(f"Failed to load face detection model: {str(e)}")
        return None

face_detector = load_model()