"""
Setup validation script to check if everything is configured correctly
"""
import os
import sys
from utils.validation import validate_environment_variables
from utils.logging_utils import logger

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        logger.error(f"Python 3.8+ required. Current version: {sys.version}")
        return False
    logger.info(f"✅ Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'alpaca_trade_api',
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'ta',
        'torch',
        'transformers',
        'vaderSentiment',
        'newsapi',
        'gspread',
        'requests',
        'pytz'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} not installed")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_directories():
    """Check and create required directories"""
    required_dirs = [
        'models/short',
        'models/medium', 
        'models/meta',
        'models/q_learning',
        'logs',
        'performance',
        'utils'
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        else:
            logger.info(f"✅ Directory exists: {directory}")
    
    return True

def check_environment_variables():
    """Check environment variables"""
    env_status = validate_environment_variables()
    
    logger.info("Environment Variables Status:")
    
    # Required variables
    for var, status in env_status['required'].items():
        if status:
            logger.info(f"✅ {var}: Set")
        else:
            logger.error(f"❌ {var}: Missing (REQUIRED)")
    
    # Optional variables
    for var, status in env_status['optional'].items():
        if status:
            logger.info(f"✅ {var}: Set")
        else:
            logger.warning(f"⚠️ {var}: Missing (Optional)")
    
    return env_status['all_required_present']

def check_alpaca_connection():
    """Test Alpaca API connection"""
    try:
        from alpaca_trade_api.rest import REST
        
        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")
        base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        
        if not api_key or not secret_key:
            logger.error("❌ Alpaca API credentials not set")
            return False
        
        api = REST(api_key, secret_key, base_url=base_url)
        account = api.get_account()
        
        logger.info(f"✅ Alpaca connection successful")
        logger.info(f"   Account Status: {account.status}")
        logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"   Portfolio Value: ${float(account.equity):,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Alpaca connection failed: {e}")
        return False

def check_discord_webhook():
    """Test Discord webhook"""
    try:
        import requests
        
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            logger.warning("⚠️ Discord webhook not configured")
            return True  # Optional
        
        test_payload = {
            "content": "🧪 Trading Bot Setup Test - Discord Integration Working!"
        }
        
        response = requests.post(webhook_url, json=test_payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Discord webhook test successful")
            return True
        else:
            logger.warning(f"⚠️ Discord webhook test failed: {response.status_code}")
            return True  # Don't fail setup for optional feature
            
    except Exception as e:
        logger.warning(f"⚠️ Discord webhook test error: {e}")
        return True  # Don't fail setup for optional feature

def run_full_setup_check():
    """Run complete setup validation"""
    logger.info("🔧 Running Trading Bot Setup Check...")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Environment Variables", check_environment_variables),
        ("Alpaca Connection", check_alpaca_connection),
        ("Discord Webhook", check_discord_webhook),
    ]
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} Check ---")
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            logger.error(f"❌ {check_name} check failed with exception: {e}")
            results[check_name] = False
            all_passed = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("SETUP CHECK SUMMARY")
    logger.info("="*50)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{check_name}: {status}")
    
    if all_passed:
        logger.info("\n🎉 All checks passed! Bot is ready to run.")
        logger.info("Start the bot with: python main.py")
    else:
        logger.error("\n❌ Some checks failed. Please fix the issues above.")
        logger.info("Required fixes:")
        for check_name, result in results.items():
            if not result and check_name in ["Python Version", "Dependencies", "Environment Variables", "Alpaca Connection"]:
                logger.error(f"  - Fix {check_name}")
    
    return all_passed

if __name__ == "__main__":
    run_full_setup_check()
