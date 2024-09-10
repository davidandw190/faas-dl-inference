import os

MODEL_PATH = os.getenv("MODEL_PATH", "emotion-ferplus-8.onnx")

EMOTION_TABLE = {
    0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
    4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'
}