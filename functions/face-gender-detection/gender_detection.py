from typing import Dict, Any
from config import GENDER_LABELS
from model_loader import gender_model
from image_processing import preprocess_image
from logger import logger

def predict_gender(image_data: bytes) -> Dict[str, Any]:
    blob = preprocess_image(image_data)
    gender_model.setInput(blob)
    output = gender_model.forward()
    gender_index = output[0].argmax()
    gender = GENDER_LABELS[gender_index]
    confidence = float(output[0][gender_index])
    return {"predicted_gender": gender, "gender_confidence": confidence}

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
                "face_detection_confidence": face["confidence"]
            })
        except Exception as e:
            logger.error(f"Error processing face {face_id}: {str(e)}")
    
    return {
        "num_faces_processed": len(results),
        "gender_results": results
    }