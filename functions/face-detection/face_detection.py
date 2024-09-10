import cv2
import numpy as np
from logger import logger
from model_loader import face_detector
from utils import nms
from config import DETECTION_THRESHOLD

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

def faceDetector(orig_image, threshold=DETECTION_THRESHOLD):
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