"""
Logging utilities for the AI Trading Bot
"""
import os
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional
import json

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class TradingLogger:
    """Custom logger for trading bot with multiple handlers"""
    
    def __init__(self, name: str = "TradingBot", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # Create logs directory
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for general logs
        file_handler = RotatingFileHandler(
            os.path.join(logs_dir, "trading_bot.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Separate handler for trades
        trade_handler = RotatingFileHandler(
            os.path.join(logs_dir, "trades.log"),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10
        )
        trade_formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        trade_handler.setFormatter(trade_formatter)
        trade_handler.addFilter(lambda record: 'TRADE' in record.getMessage())
        self.logger.addHandler(trade_handler)
        
        # Error handler
        error_handler = RotatingFileHandler(
            os.path.join(logs_dir, "errors.log"),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def get_logger(self):
        return self.logger

# Global logger instance
trading_logger = TradingLogger()
logger = trading_logger.get_logger()

def log_trade(action: str, symbol: str, quantity: int, price: float, 
              reason: str = "", additional_data: Optional[dict] = None):
    """Log trade information in structured format"""
    
    trade_data = {
        'timestamp': datetime.now().isoformat(),
        'action': action.upper(),
        'symbol': symbol,
        'quantity': quantity,
        'price': price,
        'value': quantity * price,
        'reason': reason
    }
    
    if additional_data:
        trade_data.update(additional_data)
    
    # Log as structured JSON for easy parsing
    logger.info(f"TRADE | {json.dumps(trade_data)}")

def log_model_performance(model_name: str, accuracy: float, 
                         precision: float, recall: float, f1: float):
    """Log model performance metrics"""
    
    perf_data = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    logger.info(f"MODEL_PERFORMANCE | {json.dumps(perf_data)}")

def log_portfolio_status(total_value: float, buying_power: float, 
                        positions: int, daily_pnl: float):
    """Log portfolio status"""
    
    portfolio_data = {
        'timestamp': datetime.now().isoformat(),
        'total_value': total_value,
        'buying_power': buying_power,
        'positions': positions,
        'daily_pnl': daily_pnl
    }
    
    logger.info(f"PORTFOLIO | {json.dumps(portfolio_data)}")

def log_error_with_context(error: Exception, context: dict):
    """Log error with additional context"""
    
    error_data = {
        'timestamp': datetime.now().isoformat(),
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context
    }
    
    logger.error(f"ERROR_CONTEXT | {json.dumps(error_data)}")

# Convenience functions
def debug(message: str):
    logger.debug(message)

def info(message: str):
    logger.info(message)

def warning(message: str):
    logger.warning(message)

def error(message: str):
    logger.error(message)

def critical(message: str):
    logger.critical(message)
