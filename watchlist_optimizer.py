# watchlist_optimizer.py

import random
from trading_state import trading_state
from config import config
import logger

def optimize_watchlist():
    """
    Returns a dynamically selected watchlist of top tickers.
    Placeholder: random sample from current watchlist.
    """
    full_list = trading_state.current_watchlist

    if not full_list:
        logger.deduped_log("warn", "⚠️ No tickers in current_watchlist to optimize")
        return []

    limit = config.WATCHLIST_LIMIT
    optimized = random.sample(full_list, min(len(full_list), limit))
    logger.deduped_log("info", f"📋 Optimized watchlist: {len(optimized)} tickers")
    return optimized
