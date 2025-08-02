import sqlite3
from datetime import datetime

DB_NAME = "trading_logs.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS trade_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            ticker TEXT,
            action TEXT,
            confidence REAL,
            price REAL
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS model_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_type TEXT,
            accuracy REAL,
            sample_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_trade(ticker, action, confidence, price):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO trade_log (timestamp, ticker, action, confidence, price)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), ticker, action, confidence, price))
    conn.commit()
    conn.close()

def log_model(model_type, accuracy, sample_count):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO model_log (timestamp, model_type, accuracy, sample_count)
        VALUES (?, ?, ?, ?)
    ''', (datetime.now().isoformat(), model_type, accuracy, sample_count))
    conn.commit()
    conn.close()

def log_pnl(ticker, entry_price, exit_price, pnl, duration):
    try:
        print(f"💰 PnL Log — {ticker}: Entry=${entry_price:.2f}, Exit=${exit_price:.2f}, PnL={pnl:.2%}, Duration={duration}")
        # You can optionally log this to Google Sheets, a database, or a file
    except Exception as e:
        print(f"❌ Failed to log PnL for {ticker}: {e}")
