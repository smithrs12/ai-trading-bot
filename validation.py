"""
Validation utilities for the AI Trading Bot
"""
import os
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
import requests

def validate_environment_variables() -> Dict[str, Any]:
    """Validate all required and optional environment variables"""
    
    required_vars = {
        'APCA_API_KEY_ID': os.getenv('APCA_API_KEY_ID'),
        'APCA_API_SECRET_KEY': os.getenv('APCA_API_SECRET_KEY'),
    }
    
    optional_vars = {
        'APCA_API_BASE_URL': os.getenv('APCA_API_BASE_URL'),
        'DISCORD_WEBHOOK_URL': os.getenv('DISCORD_WEBHOOK_URL'),
        'NEWS_API_KEY': os.getenv('NEWS_API_KEY'),
        'GSHEET_ID': os.getenv('GSHEET_ID'),
        'GSPREAD_JSON_PATH': os.getenv('GSPREAD_JSON_PATH'),
    }
    
    # Check required variables
    required_status = {}
    for var, value in required_vars.items():
        required_status[var] = bool(value and value.strip())
    
    # Check optional variables
    optional_status = {}
    for var, value in optional_vars.items():
        optional_status[var] = bool(value and value.strip())
    
    all_required_present = all(required_status.values())
    
    return {
        'required': required_status,
        'optional': optional_status,
        'all_required_present': all_required_present
    }

def validate_alpaca_credentials(api_key: str, secret_key: str, base_url: str) -> Tuple[bool, str]:
    """Validate Alpaca API credentials"""
    
    try:
        from alpaca_trade_api.rest import REST
        
        api = REST(api_key, secret_key, base_url=base_url)
        account = api.get_account()
        
        if account.status != 'ACTIVE':
            return False, f"Account status is {account.status}, not ACTIVE"
        
        return True, "Credentials valid"
        
    except Exception as e:
        return False, f"Credential validation failed: {str(e)}"

def validate_discord_webhook(webhook_url: str) -> Tuple[bool, str]:
    """Validate Discord webhook URL"""
    
    if not webhook_url:
        return True, "Discord webhook not configured (optional)"
    
    # Check URL format
    discord_pattern = r'https://discord(?:app)?\.com/api/webhooks/\d+/[\w-]+'
    if not re.match(discord_pattern, webhook_url):
        return False, "Invalid Discord webhook URL format"
    
    try:
        # Test webhook with a simple message
        test_payload = {
            "content": "🧪 Trading Bot Webhook Test"
        }
        
        response = requests.post(webhook_url, json=test_payload, timeout=10)
        
        if response.status_code == 200:
            return True, "Discord webhook valid"
        else:
            return False, f"Discord webhook test failed: HTTP {response.status_code}"
            
    except Exception as e:
        return False, f"Discord webhook test error: {str(e)}"

def validate_trading_parameters(config) -> List[str]:
    """Validate trading configuration parameters"""
    
    errors = []
    
    # Risk parameters
    if config.MAX_PORTFOLIO_RISK <= 0 or config.MAX_PORTFOLIO_RISK > 0.1:
        errors.append("MAX_PORTFOLIO_RISK must be between 0 and 0.1 (10%)")
    
    if config.MAX_DAILY_TRADES <= 0 or config.MAX_DAILY_TRADES > 100:
        errors.append("MAX_DAILY_TRADES must be between 1 and 100")
    
    if config.MAX_POSITIONS <= 0 or config.MAX_POSITIONS > 20:
        errors.append("MAX_POSITIONS must be between 1 and 20")
    
    # Position sizing
    if config.MIN_POSITION_SIZE <= 0:
        errors.append("MIN_POSITION_SIZE must be positive")
    
    if config.MAX_POSITION_SIZE <= config.MIN_POSITION_SIZE:
        errors.append("MAX_POSITION_SIZE must be greater than MIN_POSITION_SIZE")
    
    # Kelly multiplier
    if config.KELLY_MULTIPLIER <= 0 or config.KELLY_MULTIPLIER > 1:
        errors.append("KELLY_MULTIPLIER must be between 0 and 1")
    
    # Stop loss and profit targets
    if config.STOP_LOSS_MULTIPLIER <= 0:
        errors.append("STOP_LOSS_MULTIPLIER must be positive")
    
    if config.PROFIT_TARGET_MULTIPLIER <= config.STOP_LOSS_MULTIPLIER:
        errors.append("PROFIT_TARGET_MULTIPLIER should be greater than STOP_LOSS_MULTIPLIER")
    
    # Thresholds
    if config.MIN_PREDICTION_CONFIDENCE < 0.5 or config.MIN_PREDICTION_CONFIDENCE > 1.0:
        errors.append("MIN_PREDICTION_CONFIDENCE must be between 0.5 and 1.0")
    
    if config.SENTIMENT_OVERRIDE_THRESHOLD < -1.0 or config.SENTIMENT_OVERRIDE_THRESHOLD > 1.0:
        errors.append("SENTIMENT_OVERRIDE_THRESHOLD must be between -1.0 and 1.0")
    
    # Volume and momentum
    if config.MIN_VOLUME_MULTIPLIER <= 1.0:
        errors.append("MIN_VOLUME_MULTIPLIER must be greater than 1.0")
    
    if config.MIN_MOMENTUM_THRESHOLD <= 0:
        errors.append("MIN_MOMENTUM_THRESHOLD must be positive")
    
    # Sector limits
    if config.MAX_SECTOR_ALLOCATION <= 0 or config.MAX_SECTOR_ALLOCATION > 1.0:
        errors.append("MAX_SECTOR_ALLOCATION must be between 0 and 1.0")
    
    if config.MAX_POSITIONS_PER_SECTOR <= 0:
        errors.append("MAX_POSITIONS_PER_SECTOR must be positive")
    
    return errors

def validate_watchlist_symbols(symbols: List[str]) -> Tuple[List[str], List[str]]:
    """Validate watchlist symbols and return valid/invalid lists"""
    
    valid_symbols = []
    invalid_symbols = []
    
    # Basic symbol format validation
    symbol_pattern = r'^[A-Z]{1,5}$'
    
    for symbol in symbols:
        symbol = symbol.upper().strip()
        
        if re.match(symbol_pattern, symbol):
            valid_symbols.append(symbol)
        else:
            invalid_symbols.append(symbol)
    
    return valid_symbols, invalid_symbols

def validate_market_hours() -> Tuple[bool, str]:
    """Check if market is open or will be open soon"""
    
    try:
        import pytz
        from datetime import datetime, time
        
        # Eastern Time (market timezone)
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        current_time = now_et.time()
        is_weekday = now_et.weekday() < 5  # Monday = 0, Friday = 4
        
        if not is_weekday:
            return False, "Market is closed (weekend)"
        
        if current_time < market_open:
            return False, f"Market opens at {market_open.strftime('%H:%M')} ET"
        
        if current_time > market_close:
            return False, "Market is closed for the day"
        
        return True, "Market is open"
        
    except Exception as e:
        return False, f"Unable to determine market hours: {str(e)}"

def validate_model_files() -> Dict[str, bool]:
    """Check if required model files exist"""
    
    model_files = {
        'short_term_model': 'models/short/model.joblib',
        'medium_term_model': 'models/medium/model.joblib',
        'meta_model': 'models/meta/model.joblib',
        'q_learning_model': 'models/q_learning/model.pkl'
    }
    
    file_status = {}
    
    for model_name, file_path in model_files.items():
        file_status[model_name] = os.path.exists(file_path)
    
    return file_status

def validate_data_quality(data) -> List[str]:
    """Validate data quality for model training"""
    
    issues = []
    
    if data is None or len(data) == 0:
        issues.append("No data provided")
        return issues
    
    # Check for missing values
    if hasattr(data, 'isnull'):
        null_counts = data.isnull().sum()
        high_null_cols = null_counts[null_counts > len(data) * 0.1].index.tolist()
        if high_null_cols:
            issues.append(f"High null values in columns: {high_null_cols}")
    
    # Check data size
    if len(data) < 100:
        issues.append(f"Insufficient data: {len(data)} rows (minimum 100 required)")
    
    # Check for duplicate timestamps if present
    if 'timestamp' in data.columns:
        duplicates = data['timestamp'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate timestamps")
    
    return issues

def run_comprehensive_validation() -> Dict[str, Any]:
    """Run all validation checks and return comprehensive report"""
    
    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'UNKNOWN',
        'checks': {}
    }
    
    # Environment variables
    env_check = validate_environment_variables()
    validation_report['checks']['environment'] = {
        'status': 'PASS' if env_check['all_required_present'] else 'FAIL',
        'details': env_check
    }
    
    # Market hours
    market_open, market_msg = validate_market_hours()
    validation_report['checks']['market_hours'] = {
        'status': 'PASS' if market_open else 'INFO',
        'message': market_msg
    }
    
    # Model files
    model_files = validate_model_files()
    all_models_exist = all(model_files.values())
    validation_report['checks']['model_files'] = {
        'status': 'PASS' if all_models_exist else 'WARNING',
        'details': model_files
    }
    
    # Determine overall status
    critical_failures = [
        validation_report['checks']['environment']['status'] == 'FAIL'
    ]
    
    if any(critical_failures):
        validation_report['overall_status'] = 'FAIL'
    elif not all_models_exist:
        validation_report['overall_status'] = 'WARNING'
    else:
        validation_report['overall_status'] = 'PASS'
    
    return validation_report
