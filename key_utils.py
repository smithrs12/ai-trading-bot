import os
from typing import Optional

def redis_key(suffix: str, *parts: str, user_id: Optional[str] = None) -> str:
    """
    Build a namespaced Redis key that is consistent across the codebase.
    Usage examples:
        redis_key("mode")  -> "user:<UID>:mode"
        redis_key("COOLDOWN", "AAPL", user_id="bob") -> "user:bob:COOLDOWN:AAPL"
    """
    uid = user_id or os.getenv("USER_SESSION_ID") or os.getenv("USER_ID") or "default_user"
    tail = ":".join([suffix, *[str(p) for p in parts if p is not None and str(p) != ""]])
    return f"user:{uid}:{tail}"
