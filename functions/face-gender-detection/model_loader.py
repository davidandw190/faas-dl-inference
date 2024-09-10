import cv2
from logger import logger
from config import MODEL_PATH, CONFIG_PATH

def load_model():
    try:
        net = cv2.dnn.readNet(MODEL_PATH, CONFIG_PATH)
        logger.info("Gender classification model loaded successfully.")
        return net
    except Exception as e:
        logger.error(f"Error loading the gender classification model: {str(e)}")
        raise

gender_model = load_model()