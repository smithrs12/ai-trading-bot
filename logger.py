# logger.py

import logging
import sys
import time
from datetime import datetime

# Track recently logged messages to avoid spam
_log_cache = {}

def setup_logger(name="BotLogger", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

# Global logger instance
logger = setup_logger()

def deduped_log(level: str, message: str, cooldown_sec: int = 60):
    """
    Logs a message only if it hasn't been printed in the last `cooldown_sec` seconds.
    """
    now = time.time()
    key = f"{level}:{message}"
    last_time = _log_cache.get(key, 0)
    if now - last_time >= cooldown_sec:
        _log_cache[key] = now
        getattr(logger, level, logger.info)(message)

# Direct log aliases for convenience
def debug(msg): logger.debug(msg)
def info(msg): logger.info(msg)
def warn(msg): logger.warning(msg)
def error(msg): logger.error(msg)
