import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="Ultra Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

def load_trading_state():
    """Load current trading state"""
    try:
        if os.path.exists('state/trading_state.json'):
            with open('state/trading_state.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Failed to load trading state: {e}")
    return None

def load_trade_history():
    """Load trade history from logs"""
    try:
        trades = []
        log_files = [f for f in os.listdir('logs') if f.startswith('trades_')]
        
        for log_file in sorted(log_files)[-7:]:  # Last 7 days
            with open(f'logs/{log_file}', 'r') as f:
                for line in f:
                    if 'BUY EXECUTED' in line or 'Position closed' in line:
                        trades.append(line.strip())
        
        return trades[-100:]  # Last 100 trades
    except Exception as e:
        st.error(f"Failed to load trade history: {e}")
    return []

def main():
    st.title("🤖 Ultra-Advanced Trading Bot Dashboard")
    st.markdown("---")
    
    # Load data
    trading_state = load_trading_state()
    trade_history = load_trade_history()
    
    if not trading_state:
        st.error("❌ No trading state data available")
        return
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    
    if auto_refresh:
        st.rerun()
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    perf_metrics = trading_state.get('performance_metrics', {})
    
    with col1:
        current_equity = perf_metrics.get('current_equity', 100000)
        st.metric(
            "💰 Current Equity",
            f"${current_equity:,.2f}",
            delta=f"${perf_metrics.get('daily_pnl', 0):+,.2f}"
        )
    
    with col2:
        total_pnl = perf_metrics.get('total_pnl', 0)
        st.metric(
            "📊 Total P&L",
            f"${total_pnl:+,.2f}",
            delta=f"{(total_pnl/100000)*100:+.2f}%"
        )
    
    with col3:
        win_rate = perf_metrics.get('win_rate', 0)
        st.metric(
            "🎯 Win Rate",
            f"{win_rate:.1%}",
            delta=f"{perf_metrics.get('total_trades', 0)} trades"
        )
    
    with col4:
        sharpe_ratio = perf_metrics.get('sharpe_ratio', 0)
        st.metric(
            "📈 Sharpe Ratio",
            f"{sharpe_ratio:.2f}",
            delta=f"Max DD: {perf_metrics.get('max_drawdown', 0):.1%}"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Equity Curve")
        # Simulate equity curve data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        equity_values = np.cumsum(np.random.randn(100) * 100) + 100000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_values,
            mode='lines',
            name='Equity',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Date",
            yaxis_title="Equity ($)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🥧 Sector Allocation")
        sector_allocations = trading_state.get('sector_allocations', {})
        
        if sector_allocations:
            fig = px.pie(
                values=list(sector_allocations.values()),
                names=list(sector_allocations.keys()),
                title="Current Sector Allocation"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions")
    
    # Positions table
    st.subheader("📋 Open Positions")
    open_positions = trading_state.get('open_positions', {})
    
    if open_positions:
        positions_data = []
        for ticker, pos in open_positions.items():
            positions_data.append({
                'Ticker': ticker,
                'Quantity': pos['quantity'],
                'Entry Price': f"${pos['entry_price']:.2f}",
                'Current Price': f"${pos.get('current_price', pos['entry_price']):.2f}",
                'Unrealized P&L': f"${pos.get('unrealized_pnl', 0):+.2f}",
                'Sector': pos.get('sector', 'Unknown'),
                'Confidence': f"{pos.get('confidence', 0.5):.1%}",
                'Entry Time': pos.get('entry_time', 'Unknown')
            })
        
        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions, use_container_width=True)
    else:
        st.info("No open positions")
    
    # Recent trades
    st.subheader("📊 Recent Trades")
    if trade_history:
        st.text_area("Trade Log", "\n".join(trade_history[-10:]), height=200)
    else:
        st.info("No recent trades")
    
    # Model performance
    st.subheader("🤖 Model Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        model_accuracy = trading_state.get('model_accuracy', {})
        if model_accuracy:
            accuracy_df = pd.DataFrame([
                {'Model': k, 'Accuracy': v} 
                for k, v in model_accuracy.items()
            ])
            fig = px.bar(accuracy_df, x='Model', y='Accuracy', title="Model Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model performance data")
    
    with col2:
        # Risk metrics
        st.markdown("### 🛡️ Risk Metrics")
        risk_data = {
            'Current Drawdown': f"{perf_metrics.get('max_drawdown', 0):.2%}",
            'Daily P&L': f"${perf_metrics.get('daily_pnl', 0):+.2f}",
            'Position Count': len(open_positions),
            'Trading Status': "🟢 Active" if not trading_state.get('trading_halted', False) else "🔴 Halted"
        }
        
        for metric, value in risk_data.items():
            st.markdown(f"**{metric}:** {value}")
    
    # Footer
    st.markdown("---")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto refresh
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
