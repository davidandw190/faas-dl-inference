import cv2
import numpy as np
from typing import Dict, Any
from logger import logger

def process_image(image_data: bytes) -> Dict[str, Any]:
    if not image_data:
        return {"error": "Empty image data"}

    try:
        nparr = np.frombuffer(image_data, np.uint8)
        
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original_image is None:
            logger.error("Failed to decode image with OpenCV")
            return {"error": "Failed to decode image"}
        
        _, encoded_image = cv2.imencode('.jpg', original_image)
        image_base64 = encoded_image.tobytes().hex()
        
        return {
            "image": image_base64,
            "image_shape": original_image.shape
        }
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        return {"error": f"Image processing failed: {str(e)}"}