import json
import logging
import sys
from typing import Dict, Any

import onnxruntime as ort
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "emotion-ferplus-8.onnx"

try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    logger.info("Emotion recognition model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading the emotion recognition model: {str(e)}")
    raise

emotion_table = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                 4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'}

def preprocess(image_data: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img_data = np.expand_dims(img, axis=(0, 1))
    return img_data.astype(np.float32)

def softmax(scores):
    exp = np.exp(scores - np.max(scores))
    return exp / exp.sum()

def predict_emotion(image_data: bytes) -> Dict[str, Any]:
    input_data = preprocess(image_data)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    scores = session.run([output_name], {input_name: input_data})[0]
    probabilities = softmax(scores[0])
    
    emotion_index = np.argmax(probabilities)
    emotion = emotion_table[emotion_index]
    confidence = round(float(probabilities[emotion_index]), 2)
    
    return {
        "predicted_emotion": emotion,
        "emotion_confidence": confidence,
        "emotion_probabilities": {emotion_table[i]: round(float(prob), 3) for i, prob in enumerate(probabilities)}
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
                "face_detection_confidence": round(face["confidence"], 2)
            })
        except Exception as e:
            logger.error(f"Error processing face {face_id}: {str(e)}")
    
    return {
        "num_faces_processed": len(results),
        "emotion_results": results
    }

def handle(req: bytes) -> bytes:
    try:
        logger.info("Received request for emotion detection")
        
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