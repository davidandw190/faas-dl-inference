import json
import sys
import logging
import base64
from typing import Dict, Any

import cv2
import onnxruntime as ort
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "version-RFB-320.onnx"
face_detector = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

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


def faceDetector(orig_image, threshold=0.7):
    logger.info(f"faceDetector input image shape: {orig_image.shape if orig_image is not None else 'None'}")
    if orig_image is None or orig_image.size == 0:
        logger.error("Input image is None or empty")
        return [], [], []

    try:
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        logger.error(f"Error in cvtColor: {str(e)}")
        return [], [], []

    image = cv2.resize(image, (320, 240))
    image = (image - 127.5) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs

def process_image(image_data: bytes) -> Dict[str, Any]:
    logger.info(f"process_image input data length: {len(image_data)}")
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
        face = orig_image[box[1]:box[3], box[0]:box[2]]
        _, face_bytes = cv2.imencode('.jpg', face)
        face_base64 = base64.b64encode(face_bytes).decode('utf-8')
        
        results.append({
            "face_id": i + 1,
            "confidence": float(probs[i]),
            "bounding_box": box.tolist(),
            "face_image": face_base64
        })
    
    for box in boxes:
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 128, 0), 2)
    
    _, processed_img_bytes = cv2.imencode('.jpg', orig_image)
    processed_img_base64 = base64.b64encode(processed_img_bytes).decode('utf-8')
    
    return {
        "num_faces_detected": len(boxes),
        "processed_image": processed_img_base64,
        "faces": results
    }
    
    
def handle(req):
    try:
        logger.info(f"Received input type: {type(req)}")
        
        if isinstance(req, str):
            try:
                input_data = json.loads(req)
                image_base64 = input_data.get("image")
                
                if not image_base64:
                    return json.dumps({"error": "No image provided in JSON input"})
                
                image_base64 = image_base64.strip()
                
                if image_base64.startswith('data:image'):
                    image_base64 = image_base64.split(',')[1]
                
                image_data = image_base64
                logger.info("Successfully decoded base64 image from JSON")
            except json.JSONDecodeError:
                try:
                    req = req.strip()
                    if req.startswith('data:image'):
                        req = req.split(',')[1]
                    
                    image_data = base64.b64decode(req)
                    logger.info("Successfully decoded base64 image from string")
                except:
                    return json.dumps({"error": "Failed to decode base64 image"})
        else:
            image_data = req
            logger.info("Received raw image data")
        
        logger.info(f"Image data length: {len(image_data)}")

        result = process_image(image_data)
        return json.dumps(result)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})
    
if __name__ == "__main__":
    for line in sys.stdin:
        ret = handle(line)
        print(ret, flush=True)