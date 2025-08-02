# subscription_manager.py

import streamlit as st
import redis
import os
from datetime import datetime
from alpaca_trade_api.rest import REST

from urllib.parse import urlparse  # Make sure this is at the top of your file

class SubscriptionManager:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        try:
            parsed_url = urlparse(self.redis_url)
            use_ssl = parsed_url.scheme == "rediss"

            if use_ssl:
                self.redis_client = redis.Redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    ssl_cert_reqs=None  # ✅ Needed for Upstash
                )
            else:
                self.redis_client = redis.Redis.from_url(
                    self.redis_url,
                    decode_responses=True
                )
            self.redis_client.ping()
            print("✅ Redis connected in SubscriptionManager")
        except Exception as e:
            print(f"❌ Redis connection error in SubscriptionManager: {e}")
            self.redis_client = None
    
    def check_user_subscription(self, user_id):
        """Check if user has an active subscription in Redis"""
        if self.redis_client is None:
            return True  # Allow access if Redis is not available
        active = self.redis_client.get(f"user:{user_id}:active")
        return active == "true"
    
    def check_alpaca_subscription(self, api):
        """Check if Alpaca account has required SIP subscription"""
        try:
            if not api:
                return False, "No API credentials provided"
            
            # Try to get account info
            account = api.get_account()
            
            # Try to access SIP data by getting a quote
            try:
                api.get_latest_quote('AAPL')
                return True, "SIP subscription active"
            except Exception as e:
                error_msg = str(e)
                if "403" in error_msg or "permission" in error_msg.lower() or "subscription" in error_msg.lower():
                    return False, "SIP subscription required"
                else:
                    return True, "API access available"
                    
        except Exception as e:
            return False, f"API connection error: {str(e)}"
    
    def get_subscription_status(self, user_id, api):
        """Get comprehensive subscription status"""
        user_subscribed = self.check_user_subscription(user_id)
        alpaca_subscribed, alpaca_msg = self.check_alpaca_subscription(api)
        
        return {
            'user_subscribed': user_subscribed,
            'alpaca_subscribed': alpaca_subscribed,
            'alpaca_message': alpaca_msg,
            'all_subscribed': user_subscribed and alpaca_subscribed
        }
    
    def display_subscription_status(self, user_id, api):
        """Display subscription status in Streamlit UI"""
        status = self.get_subscription_status(user_id, api)
        
        if not status['all_subscribed']:
            st.error("🚫 **Subscription Required**")
            
            if not status['user_subscribed']:
                st.error("**User Subscription:** Inactive")
            if not status['alpaca_subscribed']:
                st.error(f"**Alpaca Subscription:** {status['alpaca_message']}")
            
            st.info("""
            **To use the trading bot, you need:**
            1. **Alpaca SIP Subscription** - Required for real-time market data
            2. **Active Account** - Your Alpaca account must be approved for trading
            3. **Valid API Credentials** - Ensure your API keys are correct
            
            **What you can do:**
            - Browse the dashboard and view historical data
            - Configure your settings
            - Monitor market conditions
            - **Trading functionality will be disabled until requirements are met**
            """)
            
            return False
        else:
            st.success("✅ **All subscriptions active**")
            return True
    
    def get_subscription_requirements(self):
        """Return a list of subscription requirements"""
        return [
            {
                'name': 'Alpaca SIP Subscription',
                'description': 'Required for real-time market data access',
                'url': 'https://alpaca.markets/docs/market-data/getting-started/',
                'cost': 'Free tier available with limitations'
            },
            {
                'name': 'Alpaca Trading Account',
                'description': 'Paper or live trading account',
                'url': 'https://alpaca.markets/',
                'cost': 'Free for paper trading'
            },
            {
                'name': 'Valid API Credentials',
                'description': 'API key and secret for Alpaca',
                'url': 'https://alpaca.markets/docs/get-started-with-alpaca/',
                'cost': 'Free'
            }
        ]

# Global instance
subscription_manager = SubscriptionManager()
