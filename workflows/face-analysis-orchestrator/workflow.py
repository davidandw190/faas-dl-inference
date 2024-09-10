import requests
from typing import Dict, Any
from logger import logger
from cache import generate_cache_key, get_cached_result, set_cached_result
from image_processing import process_image
from config import FACE_DETECTION_FUNCTION, GENDER_DETECTION_FUNCTION, EMOTION_DETECTION_FUNCTION, GATEWAY_URL

def call_function(function_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GATEWAY_URL}/function/{function_name}"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()

def face_analysis_workflow(image_data: bytes) -> Dict[str, Any]:
    logger.info(f"face_analysis_workflow received data of length: {len(image_data)} bytes")
    
    cache_key = generate_cache_key(image_data)
    cached_result = get_cached_result(cache_key)
    if cached_result:
        logger.info("Returning cached result")
        return cached_result
    
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
        
    set_cached_result(cache_key, combined_results)    
    
    logger.info("Face analysis workflow completed successfully")
    return combined_results