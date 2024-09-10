import numpy as np
from typing import Dict, Any
from config import EMOTION_TABLE
from model_loader import emotion_model
from image_processing import preprocess

def softmax(scores):
    exp = np.exp(scores - np.max(scores))
    return exp / exp.sum()

def predict_emotion(image_data: bytes) -> Dict[str, Any]:
    input_data = preprocess(image_data)
    input_name = emotion_model.get_inputs()[0].name
    output_name = emotion_model.get_outputs()[0].name
    
    scores = emotion_model.run([output_name], {input_name: input_data})[0]
    probabilities = softmax(scores[0])
    
    emotion_index = np.argmax(probabilities)
    emotion = EMOTION_TABLE[emotion_index]
    confidence = float(probabilities[emotion_index])
    
    return {
        "predicted_emotion": emotion,
        "emotion_confidence": confidence,
        "emotion_probabilities": {EMOTION_TABLE[i]: float(prob) for i, prob in enumerate(probabilities)}
    }

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
            emotion_result = predict_emotion(face_image_bytes)
            
            results.append({
                "face_id": face_id,
                "emotion_result": emotion_result,
                "face_detection_confidence": face["confidence"]
            })
        except Exception as e:
            logger.error(f"Error processing face {face_id}: {str(e)}")
    
    return {
        "num_faces_processed": len(results),
        "emotion_results": results
    }