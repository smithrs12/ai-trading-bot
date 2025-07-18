from collections import defaultdict
import random
from trading_state import trading_state
from config import config
import logger
from execution_manager import get_sector_for_ticker
from datetime import datetime

def optimize_watchlist():
    '''
    Dynamically selects a sector-balanced, performance-prioritized watchlist.
    Adds scoring from sentiment and momentum (if available).
    '''
    full_list = trading_state.current_watchlist
    if not full_list:
        logger.deduped_log("warn", "\u26a0\ufe0f No tickers in current_watchlist to optimize")
        trading_state.qualified_watchlist = []
        return []

    # Placeholder for scoring (add real logic later)
    sentiment_scores = trading_state.sentiment_cache  # {ticker: {score, timestamp}}
    momentum_scores = trading_state.watchlist_performance  # {ticker: score}

    def score(ticker):
        sentiment = sentiment_scores.get(ticker, {}).get("score", 0)
        momentum = momentum_scores.get(ticker, 0)
        return round(0.6 * momentum + 0.4 * sentiment, 4)

    sector_map = defaultdict(list)
    for ticker in full_list:
        sector = get_sector_for_ticker(ticker)
        sector_map[sector].append(ticker)

    per_sector_limit = config.MAX_PER_SECTOR_WATCHLIST
    total_limit = config.WATCHLIST_LIMIT

    ranked = []
    for sector, tickers in sector_map.items():
        scored = sorted(tickers, key=score, reverse=True)
        ranked.extend(scored[:per_sector_limit])

    final = ranked[:total_limit]

    trading_state.qualified_watchlist = final
    trading_state.last_watchlist_optimization = datetime.now()

    logger.deduped_log("info", f"\ud83d\udccb Optimized watchlist: {len(final)} tickers using sentiment/momentum scoring")
    return final
