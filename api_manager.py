# api_manager.py

from alpaca_trade_api.rest import REST
from config import config

# === Global Alpaca API Client ===
api = None

def initialize_api():
    """Initialize the Alpaca API object using the current config"""
    global api
    try:
        api = REST(
            key_id=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            base_url=config.ALPACA_BASE_URL
        )
        print("✅ Alpaca API initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize Alpaca API: {e}")
        api = None

# Call once on module load
initialize_api()

def reinitialize_api():
    """Call this after switching between Paper/Live mode to refresh credentials."""
    initialize_api()

def safe_api_call(func, *args, **kwargs):
    """Wrapper to safely call Alpaca API methods without crashing, and detect SIP subscription errors"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_msg = str(e)
        # Detect SIP subscription error (Alpaca returns 403 or specific error message)
        if ("sip" in error_msg.lower() and ("permission" in error_msg.lower() or "access" in error_msg.lower() or "403" in error_msg)) or ("403" in error_msg and "subscription" in error_msg.lower()):
            print("❌ SIP subscription error detected.")
            return "SIP_SUBSCRIPTION_ERROR"
        print(f"❌ API call failed: {e}")
        return None
