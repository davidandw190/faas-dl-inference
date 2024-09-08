import json
import logging
import sys
from typing import Dict, Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "gender_googlenet.caffemodel"
CONFIG_PATH = "gender_googlenet.prototxt"

try:
    net = cv2.dnn.readNet(MODEL_PATH, CONFIG_PATH)
    logger.info("Gender classification model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading the gender classification model: {str(e)}")
    raise

gender_labels = ['Male', 'Female']

def preprocess_image(image_data: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (104, 117, 123), swapRB=False)
    return blob

def predict_gender(image_data: bytes) -> Dict[str, Any]:
    blob = preprocess_image(image_data)
    net.setInput(blob)
    output = net.forward()
    gender_index = output[0].argmax()
    gender = gender_labels[gender_index]
    confidence = float(output[0][gender_index])
    return {"gender": gender, "confidence": confidence}

def process_faces(faces: list) -> Dict[str, Any]:
    results = []
    for face in faces:
        try:
            face_id = face["face_id"]
            face_image = face["face_image"]
            
            if not face_image:
                logger.warning(f"No image data for face_id: {face_id}")
                continue
            
            face_image_bytes = bytes.fromhex(face_image)
            gender_result = predict_gender(face_image_bytes)
            
            results.append({
                "face_id": face_id,
                "gender_result": gender_result,
                "detection_confidence": face["confidence"],
                "bounding_box": face["bounding_box"]
            })
        except Exception as e:
            logger.error(f"Error processing face {face_id}: {str(e)}")
    
    return {
        "num_faces_processed": len(results),
        "gender_results": results
    }

def handle(req: bytes) -> bytes:
    try:
        logger.info("Received request for gender detection")
        
        if not req:
            return json.dumps({"error": "Empty request"}).encode('utf-8')
        
        input_data = json.loads(req.decode('utf-8'))
        
        if "faces" not in input_data or not input_data["faces"]:
            return json.dumps({"error": "No faces provided in the input"}).encode('utf-8')
        
        result = process_faces(input_data["faces"])
        
        return json.dumps(result).encode('utf-8')
    
    except json.JSONDecodeError:
        logger.error("Invalid JSON input")
        return json.dumps({"error": "Invalid JSON input"}).encode('utf-8')
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"}).encode('utf-8')

if __name__ == "__main__":
    try:
        input_data = sys.stdin.buffer.read()
        ret = handle(input_data)
        sys.stdout.buffer.write(ret)
        sys.stdout.buffer.flush()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        error_response = json.dumps({"error": f"Main execution failed: {str(e)}"}).encode('utf-8')
        sys.stdout.buffer.write(error_response)
        sys.stdout.buffer.flush()