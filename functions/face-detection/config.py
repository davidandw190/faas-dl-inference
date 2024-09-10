import os

MODEL_PATH = os.getenv("MODEL_PATH", "version-RFB-320.onnx")
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.8))