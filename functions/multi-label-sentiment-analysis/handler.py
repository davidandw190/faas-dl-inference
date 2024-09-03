import json
import sys
import logging
from typing import Dict, List, Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "improved_sentiment_classifier-int8.onnx"
TOKENIZER_NAME = "microsoft/xtremedistil-l6-h384-uncased"
EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
MAX_LENGTH = 64

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def process_text(text: str) -> Dict[str, List[Dict[str, Any]]]:
    inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=MAX_LENGTH)
    
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    
    logits = session.run(None, ort_inputs)[0]
    probabilities = sigmoid(logits)[0]
    
    top_emotions = np.argsort(probabilities)[-3:][::-1]
    
    return {
        "result": [
            {"emotion": EMOTIONS[i], "probability": float(probabilities[i])}
            for i in top_emotions
        ]
    }

def handle(req: str) -> str:
    try:
        input_data = json.loads(req)
        text = input_data.get("text", "")
        
        if not text:
            return json.dumps({"error": "No text provided"})
        
        result = process_text(text)
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