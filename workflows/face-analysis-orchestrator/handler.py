import json
import logging
import sys
from typing import Dict, Any
import requests
import cv2
import numpy as np
from prometheus_client import Counter, Histogram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GATEWAY_URL = "http://gateway.openfaas:8080"
FACE_DETECTION_FUNCTION = "face-detection"
GENDER_DETECTION_FUNCTION = "face-gender-detection"
EMOTION_DETECTION_FUNCTION = "face-emotion-detection"

# Metrics
request_counter = Counter('face_analysis_requests_total', 'Total number of face analysis requests')
processing_time = Histogram('face_analysis_processing_seconds', 'Time spent processing face analysis')

def call_function(function_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GATEWAY_URL}/function/{function_name}"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()

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

def face_analysis_workflow(image_data: bytes) -> Dict[str, Any]:
    logger.info(f"face_analysis_workflow received data of length: {len(image_data)} bytes")
    
    processed_image = process_image(image_data)
    if "error" in processed_image:
        return processed_image

    face_detection_result = call_function(FACE_DETECTION_FUNCTION, processed_image)
    
    if "error" in face_detection_result:
        logger.error(f"Error from face-detection function: {face_detection_result['error']}")
        return face_detection_result
    
    logger.info(f"Face detection successful. Detected {face_detection_result['num_faces_detected']} faces.")
    
    gender_detection_result = call_function(GENDER_DETECTION_FUNCTION, face_detection_result)
    
    emotion_detection_result = call_function(EMOTION_DETECTION_FUNCTION, face_detection_result)
    
    combined_results = {
        "num_faces_detected": face_detection_result["num_faces_detected"],
        "faces": []
    }
    
    for face_id in range(1, face_detection_result["num_faces_detected"] + 1):
        face_info = {
            "face_id": face_id,
            "bounding_box": next(face["bounding_box"] for face in face_detection_result["faces"] if face["face_id"] == face_id),
            "detection_confidence": next(face["confidence"] for face in face_detection_result["faces"] if face["face_id"] == face_id),
            "gender": next(result["gender_result"]["predicted_gender"] for result in gender_detection_result["gender_results"] if result["face_id"] == face_id),
            "gender_confidence": next(result["gender_result"]["gender_confidence"] for result in gender_detection_result["gender_results"] if result["face_id"] == face_id),
            "emotion": next(result["emotion_result"]["predicted_emotion"] for result in emotion_detection_result["emotion_results"] if result["face_id"] == face_id),
            "emotion_confidence": next(result["emotion_result"]["emotion_confidence"] for result in emotion_detection_result["emotion_results"] if result["face_id"] == face_id),
            "emotion_probabilities": next(result["emotion_result"]["emotion_probabilities"] for result in emotion_detection_result["emotion_results"] if result["face_id"] == face_id),
            "face_image": next(face["face_image"] for face in face_detection_result["faces"] if face["face_id"] == face_id)
        }
        combined_results["faces"].append(face_info)
    
    logger.info("Face analysis workflow completed successfully")
    return combined_results

def handle(req: bytes) -> bytes:
    try:
        request_counter.inc()
        with processing_time.time():
            if not req:
                return json.dumps({"error": "Empty request"}).encode('utf-8')
            
            result = face_analysis_workflow(req)
            
            return json.dumps(result).encode('utf-8')
    
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