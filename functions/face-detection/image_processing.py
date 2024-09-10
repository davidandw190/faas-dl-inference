import cv2
import numpy as np
from typing import Dict, Any
from logger import logger
from face_detection import faceDetector

def process_image(image_data: Dict[str, Any]) -> Dict[str, Any]:
    if not image_data or "image" not in image_data:
        logger.error("Received empty or invalid image data")
        return {"error": "Empty or invalid image data"}

    image_hex = image_data["image"]
    image_shape = image_data["image_shape"]
    logger.info(f"process_image input data length: {len(image_hex)}")
    
    try:
        nparr = np.frombuffer(bytes.fromhex(image_hex), np.uint8)
        orig_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if orig_image is None:
            logger.error("Failed to decode image with OpenCV")
            return {"error": "Failed to decode image"}
        
        logger.info(f"Decoded image shape: {orig_image.shape}")
        
        boxes, _, probs = faceDetector(orig_image)
        
        results = []
        for i, box in enumerate(boxes):
            try:
                face = orig_image[box[1]:box[3], box[0]:box[2]]
                _, face_bytes = cv2.imencode('.jpg', face)
                
                results.append({
                    "face_id": i + 1,
                    "confidence": float(probs[i]),  # No rounding
                    "bounding_box": box.tolist(),
                    "face_image": face_bytes.tobytes().hex()
                })
            except Exception as e:
                logger.error(f"Error processing face {i+1}: {str(e)}")
        
        return {
            "num_faces_detected": len(boxes),
            "faces": results
        }
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        return {"error": f"Image processing failed: {str(e)}"}