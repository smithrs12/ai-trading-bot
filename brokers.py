import os
from alpaca_trade_api import REST

class AlpacaBroker:
    def __init__(self):
        self.api = REST(
            key_id=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            base_url=os.getenv("ALPACA_API_BASE")
        )

    def place_order(self, ticker, quantity):
        try:
            order = self.api.submit_order(
                symbol=ticker,
                qty=abs(quantity),
                side="buy" if quantity > 0 else "sell",
                type="market",
                time_in_force="gtc"
            )
            print(f"✅ Order submitted for {ticker}: {order.id}")
            return True
        except Exception as e:
            print(f"❌ Failed to place order for {ticker}: {e}")
            return False


class SimulatedBroker:
    def __init__(self):
        self.positions = {}

    def place_order(self, ticker, quantity):
        print(f"💡 Simulated order: {'BUY' if quantity > 0 else 'SELL'} {abs(quantity)} of {ticker}")
        self.positions[ticker] = self.positions.get(ticker, 0) + quantity
        return True


class InteractiveBrokersBroker:
    def __init__(self):
        print("🔌 IB Broker initialized (not yet implemented)")

    def place_order(self, ticker, quantity):
        print(f"⚠️ IB order logic not implemented for {ticker}")
        return False

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
