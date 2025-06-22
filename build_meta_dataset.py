import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    trades = pd.read_csv("trade_history.csv")
    pnl = pd.read_csv("pnl_tracker.csv")
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    pnl["timestamp"] = pd.to_datetime(pnl["timestamp"])
    return trades, pnl

def build_meta_dataset(trades, pnl):
    meta_rows = []

    # Pair BUY and SELL per ticker
    tickers = trades["ticker"].unique()
    for ticker in tickers:
        buy_rows = trades[(trades["ticker"] == ticker) & (trades["action"] == "BUY")]
        sell_rows = trades[(trades["ticker"] == ticker) & (trades["action"] == "SELL")]

        for _, buy in buy_rows.iterrows():
            # Match the closest SELL after this BUY
            sell_match = sell_rows[sell_rows["timestamp"] > buy["timestamp"]]
            if sell_match.empty:
                continue
            sell = sell_match.iloc[0]

            # Match PnL row
            pnl_match = pnl[(pnl["ticker"] == ticker) & 
                            (pnl["timestamp"] >= sell["timestamp"])].head(1)
            if pnl_match.empty:
                continue

            pnl_row = pnl_match.iloc[0]
            pnl_value = pnl_row["pnl"]
            model_type = pnl_row["model_type"]

            # Meta-features (you can expand this later)
            meta_row = {
                "ticker": ticker,
                "entry_time": buy["timestamp"],
                "exit_time": sell["timestamp"],
                "buy_price": buy["price"],
                "sell_price": sell["price"],
                "qty": buy["qty"],
                "pnl": pnl_value,
                "model_type": model_type,
                "profitable": 1 if pnl_value > 0 else 0,
            }

            # ⬇️ Placeholder values (to be replaced from logs or strategy later)
            meta_row.update({
                "proba_short": np.nan,
                "proba_mid": np.nan,
                "sentiment_score": np.nan,
                "volume_ratio": np.nan,
                "vwap_diff": np.nan,
                "regime": np.nan,
                "q_value_hold": np.nan
            })

            meta_rows.append(meta_row)

    df_meta = pd.DataFrame(meta_rows)
    df_meta.to_csv("meta_dataset.csv", index=False)
    print(f"✅ Meta-dataset saved to meta_dataset.csv with {len(df_meta)} rows.")

if __name__ == "__main__":
    trades, pnl = load_data()
    build_meta_dataset(trades, pnl)
