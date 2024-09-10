import json
import hashlib
import redis
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_TTL

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def generate_cache_key(image_data: bytes) -> str:
    return hashlib.md5(image_data).hexdigest()

def get_cached_result(cache_key: str) -> dict | None:
    cached_data = redis_client.get(cache_key)
    if cached_data:
        return json.loads(cached_data)
    return None

def set_cached_result(cache_key: str, result: dict):
    redis_client.setex(cache_key, REDIS_TTL, json.dumps(result))