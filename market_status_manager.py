# market_status_manager.py

import pytz
from datetime import datetime, timedelta
from logger import logger
import api_manager

class MarketStatusManager:
    def __init__(self):
        self.market_timezone = pytz.timezone("US/Eastern")
        self.last_market_check = None
        self.cached_market_status = False

    def is_market_open(self) -> bool:
        try:
            now = datetime.now(self.market_timezone)
            if self.last_market_check and (now - self.last_market_check).total_seconds() < 60:
                return self.cached_market_status

            if api_manager.api:
                clock = api_manager.safe_api_call(api_manager.api.get_clock)
                if clock:
                    self.cached_market_status = clock.is_open
                    self.last_market_check = now
                    return clock.is_open

            # Fallback (weekend/off hours)
            if now.weekday() >= 5:
                self.cached_market_status = False
                self.last_market_check = now
                return False

            market_open = now.replace(hour=9, minute=30)
            market_close = now.replace(hour=16, minute=0)
            self.cached_market_status = market_open <= now <= market_close
            self.last_market_check = now
            return self.cached_market_status

        except Exception as e:
            logger.error(f"Market open check failed: {e}")
            return False

    def is_in_trading_window(self) -> bool:
        try:
            if not self.is_market_open():
                return False
            now = datetime.now(self.market_timezone)
            start = now.replace(hour=10, minute=0)
            end = now.replace(hour=15, minute=45)
            return start <= now <= end
        except Exception as e:
            logger.error(f"Trading window check failed: {e}")
            return False

    def is_near_eod(self) -> bool:
        try:
            now = datetime.now(self.market_timezone)
            close = now.replace(hour=16, minute=0)
            return self.is_market_open() and (close - now <= timedelta(minutes=15))
        except Exception as e:
            logger.error(f"EOD check failed: {e}")
            return False

    def get_time_until_market_open(self) -> timedelta:
        try:
            now = datetime.now(self.market_timezone)
            if self.is_market_open():
                return timedelta(0)
            next_open = now.replace(hour=9, minute=30)
            if now >= next_open:
                next_open += timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
            return next_open - now
        except Exception as e:
            logger.error(f"Market open timing failed: {e}")
            return timedelta(hours=1)
