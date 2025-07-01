"""
Redis-backed checkpoint saver (falls back to in-memory if Redis absent).
"""

from functools import lru_cache
import os

try:
    # LangGraph ≥ 0.3.3
    from langgraph.checkpoint.redis import RedisSaver
except ImportError:  # old package or missing redis extras
    RedisSaver = None

@lru_cache
def get_memory():
    if RedisSaver is None or "REDIS_URL" not in os.environ:
        # graceful fallback
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    return RedisSaver(
        redis_url=os.environ["REDIS_URL"],
        namespace="llamascholar-chat",
        ttl=60 * 60 * 24,  # 24 h
    )
