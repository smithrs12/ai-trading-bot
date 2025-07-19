# redis_cache.py

import redis
import json
import hashlib
from config import config
from urllib.parse import urlparse

# === Redis URL ===
REDIS_URL = config.REDIS_URL

# === Redis Client ===
def get_redis_client():
    if not REDIS_URL:
        print("⚠️ REDIS_URL not set.")
        return None
    try:
        parsed_url = urlparse(REDIS_URL)
        client = redis.Redis(
            host=parsed_url.hostname,
            port=parsed_url.port,
            password=parsed_url.password,
            ssl=parsed_url.scheme == 'rediss',
            decode_responses=True
        )
        client.ping()
        print("✅ Redis connected.")
        return client
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return None

client = get_redis_client()

# === RedisCache ===
class RedisCache:
    def __init__(self, redis_client):
        self.client = redis_client
        self.enabled = redis_client is not None

    def make_key(self, prefix, payload):
        raw = json.dumps(payload, sort_keys=True)
        return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()}"

    def get(self, key):
        if not self.enabled:
            return None
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except:
            return None

    def set(self, key, value, ttl_seconds=3600):
        if not self.enabled:
            return
        try:
            self.client.setex(key, ttl_seconds, json.dumps(value))
        except:
            pass

# === Instance ===
redis_cache = RedisCache(client)

# === Key Namespacing Helper ===
def redis_key(*parts):
    return f"{config.USER_ID}:" + ":".join(parts)
