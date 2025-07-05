
import os
from alpaca_trade_api.rest import REST, TimeInForce, OrderSide, OrderType
from dotenv import load_dotenv

load_dotenv()

class AlpacaBroker:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.api = REST(self.api_key, self.api_secret, self.base_url)

    def is_available(self):
        try:
            account = self.api.get_account()
            return account.status == "ACTIVE"
        except Exception as e:
            print(f"❌ Alpaca not available: {e}")
            return False

    def place_order(self, ticker, qty, side, price=None):
        try:
            side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_type = OrderType.LIMIT if price else OrderType.MARKET

            order = self.api.submit_order(
                symbol=ticker,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=TimeInForce.DAY,
                limit_price=price if price else None
            )

            print(f"✅ Alpaca order placed: {side} {qty} {ticker} @ {price or 'MARKET'}")
            return True

        except Exception as e:
            print(f"❌ Alpaca order failed for {ticker}: {e}")
            return False

    def get_position(self, ticker):
        try:
            position = self.api.get_position(ticker)
            return {
                "symbol": position.symbol,
                "qty": int(position.qty),
                "avg_entry_price": float(position.avg_entry_price),
                "unrealized_pl": float(position.unrealized_pl)
            }
        except:
            return None
