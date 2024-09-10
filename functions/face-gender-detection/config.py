import os

MODEL_PATH = os.getenv("MODEL_PATH", "gender_googlenet.caffemodel")
CONFIG_PATH = os.getenv("CONFIG_PATH", "gender_googlenet.prototxt")

GENDER_LABELS = ['Male', 'Female']