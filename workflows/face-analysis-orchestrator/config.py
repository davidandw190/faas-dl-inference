import os

REDIS_HOST = os.getenv("REDIS_HOST", "redis-master.openfaas.svc.cluster.local")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_TTL = int(os.getenv("REDIS_TTL", 300))

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway.openfaas:8080")
FACE_DETECTION_FUNCTION = os.getenv("FACE_DETECTION_FUNCTION", "face-detection")
GENDER_DETECTION_FUNCTION = os.getenv("GENDER_DETECTION_FUNCTION", "face-gender-detection")
EMOTION_DETECTION_FUNCTION = os.getenv("EMOTION_DETECTION_FUNCTION", "face-emotion-detection")