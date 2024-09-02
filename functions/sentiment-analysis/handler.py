import json
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import sys
import logging

logging.basicConfig(level=logging.INFO)

model_path = "classifier_int8.onnx"
tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
session = ort.InferenceSession(model_path)

emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def handle(req):
    logging.info(f"Received request: {req}")
    try:
        input_data = json.loads(req)
        text = input_data.get("text", "")
        
        logging.info(f"Processed input text: {text}")
        
        if not text:
            return json.dumps({"error": "No text provided"})
        
        inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
        logging.info("Tokenization completed")
        
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"]
        }
        logging.info("Running ONNX inference")
        ort_outputs = session.run(None, ort_inputs)
        
        logits = ort_outputs[0]
        probabilities = softmax(logits[0])
        
        top_emotions = np.argsort(probabilities)[-3:][::-1]
        
        result = [
            {"emotion": emotions[i], "probability": float(probabilities[i])}
            for i in top_emotions
        ]
        
        logging.info(f"Processed result: {result}")
        return json.dumps({"result": result})
    
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {str(e)}")
        return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(f"Exception type: {type(e).__name__}")
        logging.error(f"Exception traceback: {sys.exc_info()[2]}")
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == "__main__":
    st = sys.stdin.read()
    ret = handle(st)
    print(ret, flush=True)