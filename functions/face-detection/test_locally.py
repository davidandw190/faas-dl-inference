import sys
import json
import base64
from handler import handle

def test_with_file(image_path):
    print(f"Testing with image file: {image_path}")
    result = handle(image_path)
    print_result(result)

def test_with_base64(image_path):
    print(f"Testing with base64 encoded image: {image_path}")
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    input_data = json.dumps({"image": image_base64})
    result = handle(input_data)
    print_result(result)

def print_result(result):
    result_dict = json.loads(result)
    print(f"Number of faces detected: {result_dict['num_faces_detected']}")
    for face in result_dict['faces']:
        print(f"Face ID: {face['face_id']}, Confidence: {face['confidence']:.2f}")
    print("Full result:")
    print(json.dumps(result_dict, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_locally.py <image_path> [--base64]")
        sys.exit(1)

    image_path = sys.argv[1]
    use_base64 = "--base64" in sys.argv

    if use_base64:
        test_with_base64(image_path)
    else:
        test_with_file(image_path)