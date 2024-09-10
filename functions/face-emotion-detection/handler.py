import json
import sys
from logger import logger
from emotion_detection import process_faces

def handle(req: bytes) -> bytes:
    try:
        logger.info("Received request for emotion detection")
        
        if not req:
            return json.dumps({"error": "Empty request"}).encode('utf-8')
        
        input_data = json.loads(req.decode('utf-8'))
        
        if "faces" not in input_data or not input_data["faces"]:
            return json.dumps({"error": "No faces provided in the input"}).encode('utf-8')
        
        result = process_faces(input_data["faces"])
        
        return json.dumps(result).encode('utf-8')
    
    except json.JSONDecodeError:
        logger.error("Invalid JSON input")
        return json.dumps({"error": "Invalid JSON input"}).encode('utf-8')
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"}).encode('utf-8')

if __name__ == "__main__":
    try:
        input_data = sys.stdin.buffer.read()
        ret = handle(input_data)
        sys.stdout.buffer.write(ret)
        sys.stdout.buffer.flush()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        error_response = json.dumps({"error": f"Main execution failed: {str(e)}"}).encode('utf-8')
        sys.stdout.buffer.write(error_response)
        sys.stdout.buffer.flush()