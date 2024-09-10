import json
import sys
from logger import logger
from workflow import face_analysis_workflow

def handle(req: bytes) -> bytes:
    try:
        if not req:
            return json.dumps({"error": "Empty request"}).encode('utf-8')
        
        result = face_analysis_workflow(req)
        
        return json.dumps(result).encode('utf-8')
    
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