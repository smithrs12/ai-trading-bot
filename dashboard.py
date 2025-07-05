import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
import numpy as np
import time

# === AUTHENTICATION ===
PASSWORD = os.getenv("DASHBOARD_PASSWORD", "admin")
if "auth_passed" not in st.session_state:
    st.session_state["auth_passed"] = False

if not st.session_state["auth_passed"]:
    pw = st.text_input("Enter dashboard password", type="password")
    if pw == PASSWORD:
        st.session_state["auth_passed"] = True
        st.experimental_rerun()
    else:
        st.stop()

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Ultra Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .profit {
        color: #00ff00;
    }
    .loss {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)

# === LOADERS ===
def load_trading_state():
    try:
        if os.path.exists('state/trading_state.json'):
            with open('state/trading_state.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Failed to load trading state: {e}")
    return {}

def load_trade_history():
    try:
        trades = []
        log_files = [f for f in os.listdir('logs') if f.startswith('trades_')]
        for log_file in sorted(log_files)[-7:]:
            with open(f'logs/{log_file}', 'r') as f:
                for line in f:
                    if 'BUY EXECUTED' in line or 'Position closed' in line:
                        trades.append(line.strip())
        return trades[-100:]
    except Exception as e:
        st.error(f"Failed to load trade history: {e}")
    return []

# === MAIN ===
def main():
    st.title("🤖 Ultra-Advanced Trading Bot Dashboard")
    st.markdown("---")

    trading_state = load_trading_state()
    trade_history = load_trade_history()

    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)

    # Metrics
    perf = trading_state.get('performance_metrics', {})
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Current Equity", f"${perf.get('current_equity', 100000):,.2f}", f"${perf.get('daily_pnl', 0):+,.2f}")
    total_pnl = perf.get('total_pnl', 0)
    col2.metric("📊 Total P&L", f"${total_pnl:+,.2f}", f"{(total_pnl/100000)*100:+.2f}%")
    col3.metric("🎯 Win Rate", f"{perf.get('win_rate', 0):.1%}", f"{perf.get('total_trades', 0)} trades")
    col4.metric("📈 Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}", f"Max DD: {perf.get('max_drawdown', 0):.1%}")

    # Equity Curve + Sector Allocation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Equity Curve")
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        equity_values = np.cumsum(np.random.randn(100) * 100) + 100000
        fig = go.Figure(go.Scatter(x=dates, y=equity_values, mode='lines', name='Equity'))
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Equity ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🥧 Sector Allocation")
        alloc = trading_state.get('sector_allocations', {})
        if alloc:
            fig = px.pie(values=list(alloc.values()), names=list(alloc.keys()), title="Current Sector Allocation")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions")

    # Open Positions
    st.subheader("📋 Open Positions")
    positions = trading_state.get('open_positions', {})
    if positions:
        df = pd.DataFrame([{
            'Ticker': k,
            'Quantity': v['quantity'],
            'Entry Price': f"${v['entry_price']:.2f}",
            'Current Price': f"${v.get('current_price', v['entry_price']):.2f}",
            'Unrealized P&L': f"${v.get('unrealized_pnl', 0):+.2f}",
            'Sector': v.get('sector', 'Unknown'),
            'Confidence': f"{v.get('confidence', 0.5):.1%}",
            'Entry Time': v.get('entry_time', 'Unknown')
        } for k, v in positions.items()])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No open positions")

    # Trade Log
    st.subheader("📊 Recent Trades")
    if trade_history:
        st.text_area("Trade Log", "\n".join(trade_history[-10:]), height=200)
    else:
        st.info("No recent trades")

    # Model Accuracy
    st.subheader("🤖 Model Performance")
    model_acc = trading_state.get('model_accuracy', {})
    if model_acc:
        df = pd.DataFrame([{ 'Model': k, 'Accuracy': v } for k, v in model_acc.items()])
        fig = px.bar(df, x='Model', y='Accuracy', title="Model Accuracy")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model accuracy data")

    # Risk Metrics
    st.markdown("### 🛡️ Risk Metrics")
    st.markdown(f"**Current Drawdown:** {perf.get('max_drawdown', 0):.2%}")
    st.markdown(f"**Daily P&L:** ${perf.get('daily_pnl', 0):+.2f}")
    st.markdown(f"**Position Count:** {len(positions)}")
    st.markdown(f"**Trading Status:** {'🟢 Active' if not trading_state.get('trading_halted', False) else '🔴 Halted'}")

    st.markdown("---")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
