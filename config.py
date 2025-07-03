"""
Configuration management for the AI Trading Bot
"""
import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    
    # API Configuration
    ALPACA_API_KEY: str = os.getenv("APCA_API_KEY_ID", "")
    ALPACA_SECRET_KEY: str = os.getenv("APCA_API_SECRET_KEY", "")
    ALPACA_BASE_URL: str = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    
    # External APIs
    DISCORD_WEBHOOK_URL: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    GSHEET_ID: str = os.getenv("GSHEET_ID", "")
    GSPREAD_JSON_PATH: str = os.getenv("GSPREAD_JSON_PATH", "")
    
    # Trading Parameters
    SHORT_BUY_THRESHOLD: float = 0.53
    SHORT_SELL_AVOID_THRESHOLD: float = 0.45
    PRICE_MOMENTUM_MIN: float = 0.005  # 0.5%
    VOLUME_SPIKE_MIN: float = 1.2
    SENTIMENT_HOLD_OVERRIDE: float = -0.5
    
    # Portfolio Limits
    MAX_POSITIONS: int = 10
    MAX_PER_SECTOR_WATCHLIST: int = 12
    MAX_PER_SECTOR_PORTFOLIO: float = 0.3  # 30%
    WATCHLIST_LIMIT: int = 20
    
    # Risk Management
    ATR_STOP_MULTIPLIER: float = 1.2
    ATR_PROFIT_MULTIPLIER: float = 2.5
    MAX_PORTFOLIO_RISK: float = 0.02  # 2%
    MAX_DAILY_DRAWDOWN: float = 0.05  # 5%
    VOLATILITY_GATE_THRESHOLD: float = 0.05  # 5%
    
    # Timing
    LOOP_INTERVAL: int = 300  # 5 minutes
    WATCHLIST_REFRESH_INTERVAL: int = 1800  # 30 minutes
    COOLDOWN_DURATION: int = 300  # 5 minutes
    
    def validate(self) -> bool:
        """Validate required configuration"""
        required_fields = [
            'ALPACA_API_KEY',
            'ALPACA_SECRET_KEY'
        ]
        
        for field in required_fields:
            if not getattr(self, field):
                print(f"❌ Missing required configuration: {field}")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

# Global config instance
config = TradingConfig()
