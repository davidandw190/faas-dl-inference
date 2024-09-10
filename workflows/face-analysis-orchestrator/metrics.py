from prometheus_client import Counter, Histogram

request_counter = Counter('face_analysis_requests_total', 'Total number of face analysis requests')
processing_time = Histogram('face_analysis_processing_seconds', 'Time spent processing face analysis')