import onnxruntime as ort
from logger import logger
from config import MODEL_PATH

def load_model():
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        logger.info("Emotion recognition model loaded successfully.")
        return session
    except Exception as e:
        logger.error(f"Error loading the emotion recognition model: {str(e)}")
        raise

emotion_model = load_model()