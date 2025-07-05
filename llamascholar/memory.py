# llamascholar/memory.py  (final version)
import os, dotenv
from functools import lru_cache
dotenv.load_dotenv()

@lru_cache
def get_memory():
    url = os.getenv("REDIS_URL")
    if url:
        from langgraph.checkpoint.redis import RedisSaver
        # 👇 give the URL string directly; RedisSaver will open its own connection
        return RedisSaver(url, ttl=60 * 60 * 24)   # 24-h expiry; drop ttl if not wanted
    else:
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()
