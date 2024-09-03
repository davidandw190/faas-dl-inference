import json
import sys
import logging
from typing import Dict, Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "text_classifier_int8.onnx"
TOKENIZER_NAME = "microsoft/xtremedistil-l6-h256-uncased"
MAX_LENGTH = 128
LABELS = ["World", "Sports", "Business", "Sci/Tech"]  # the AG News dataset labels

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def classify_text(text: str) -> Dict[str, Any]:
    inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"]
    }
    
    logits = session.run(None, ort_inputs)[0]
    probabilities = softmax(logits)[0]
    
    predicted_class_id = np.argmax(probabilities)
    predicted_class = LABELS[predicted_class_id]
    
    return {
        "class": predicted_class,
        "class_id": int(predicted_class_id),
        "probability": float(probabilities[predicted_class_id]),
        "probabilities": {label: float(prob) for label, prob in zip(LABELS, probabilities)}
    }

def handle(req: str) -> str:
    try:
        input_data = json.loads(req)
        text = input_data.get("text", "")
        
        if not text:
            return json.dumps({"error": "No text provided"})
        
        result = classify_text(text)
        return json.dumps(result)
    
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

if __name__ == "__main__":
    for line in sys.stdin:
        ret = handle(line)
        print(ret, flush=True)