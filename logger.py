import logging
import sys
import time
from datetime import datetime

# === Deduplicated log memory ===
_log_cache = {}

# === Logger Setup ===
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

# === Global Logger Instance ===
logger = setup_logger()

# === Deduplicated Logging ===
def deduped_log(level: str, message: str, cooldown_sec: int = 60):
    """
    Logs a message only if it hasn't been logged in the last cooldown_sec.
    Prevents log flooding during loops or frequent events.
    """
    now = time.time()
    key = f"{level}:{message}"
    last_time = _log_cache.get(key, 0)

    if now - last_time >= cooldown_sec:
        _log_cache[key] = now
        getattr(logger, level, logger.info)(message)

# === Direct Shorthand Logging Aliases ===
def debug(msg): logger.debug(msg)
def info(msg): logger.info(msg)
def warn(msg): logger.warning(msg)
def error(msg): logger.error(msg)

# === High-Frequency Metrics Log (Optional Future Use) ===
def log_metric(name: str, value, unit: str = ""):
    """
    Logs quantitative metrics in a standardized way.
    Example: log_metric("Sharpe Ratio", 2.1)
    """
    logger.info(f"[METRIC] {name}: {value} {unit}")
