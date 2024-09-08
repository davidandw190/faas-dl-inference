import json
import sys
import logging
from typing import Dict, Any

import cv2
import onnxruntime as ort
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "version-RFB-320.onnx"

try:
    face_detector = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    logger.info("Face detection model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load face detection model: {str(e)}")
    face_detector = None
    
def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes

def compute_iou(box, boxes):
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]))
    intersection_area = np.prod(intersection, axis=1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = intersection_area / (area1 + area2 - intersection_area)
    return iou

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes, confidences = boxes[0], confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_probs[nms(box_probs[:, :4], box_probs[:, 4], iou_threshold)]
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, :4] *= np.array([width, height, width, height])
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def faceDetector(orig_image, threshold=0.5):
    if face_detector is None:
        logger.error("Face detection model not initialized")
        return [], [], []

    logger.info(f"faceDetector input image shape: {orig_image.shape if orig_image is not None else 'None'}")
    if orig_image is None or orig_image.size == 0:
        logger.error("Input image is None or empty")
        return [], [], []

    try:
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        logger.error(f"Error in cvtColor: {str(e)}")
        return [], [], []

    try:
        image = cv2.resize(image, (320, 240))
        image = (image - 127.5) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = face_detector.get_inputs()[0].name
        confidences, boxes = face_detector.run(None, {input_name: image})
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
        return boxes, labels, probs
    except Exception as e:
        logger.error(f"Error during face detection: {str(e)}")
        return [], [], []

def process_image(image_data: bytes) -> Dict[str, Any]:
    if not image_data:
        logger.error("Received empty image data")
        return {"error": "Empty image data"}

    logger.info(f"process_image input data length: {len(image_data)}")
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        logger.info(f"Numpy array shape: {nparr.shape}")
        
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
                    "confidence": round(float(probs[i]), 3),
                    "bounding_box": box.tolist(),
                    "face_image": face_bytes.tobytes()
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

def handle(req: bytes) -> bytes:
    try:
        logger.info("Received request for face detection")
        logger.info(f"Input data length: {len(req)}")

        if not req:
            return json.dumps({"error": "Empty request"}).encode('utf-8')

        result = process_image(req)
        
        if "error" in result:
            return json.dumps(result).encode('utf-8')
        
        serialized_result = {
            "num_faces_detected": result["num_faces_detected"],
            "faces": [
                {
                    "face_id": face["face_id"],
                    "confidence": face["confidence"],
                    "bounding_box": face["bounding_box"],
                    "face_image": face["face_image"]
                }
                for face in result["faces"]
            ]
        }
        
        return json.dumps(serialized_result, default=lambda x: x.hex() if isinstance(x, bytes) else x).encode('utf-8')
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"}).encode('utf-8')

if __name__ == "__main__":
    try:
        image_data = sys.stdin.buffer.read()
        ret = handle(image_data)
        sys.stdout.buffer.write(ret)
        sys.stdout.buffer.flush()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        error_response = json.dumps({"error": f"Main execution failed: {str(e)}"}).encode('utf-8')
        sys.stdout.buffer.write(error_response)
        sys.stdout.buffer.flush()