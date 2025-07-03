# 🤖 AI Trading Bot - Maximum Profitability Edition

An advanced algorithmic trading bot powered by machine learning, designed for maximum profitability with comprehensive risk management.

## 🚀 Features

### **Core Trading Engine**
- **Dual-Horizon Prediction**: Short-term (2-day) + Medium-term (15-day) models
- **Ensemble Learning**: XGBoost + Random Forest + Logistic Regression voting system
- **Meta Model Approval**: 16+ feature validation system for trade confirmation
- **Q-Learning Reinforcement**: Neural network with experience replay

### **Advanced Risk Management**
- **Kelly Criterion Position Sizing**: Optimal position sizing with confidence adjustment
- **Dynamic ATR-based Stops**: Adaptive stop-loss and profit targets
- **Sector Diversification**: Maximum 30% allocation per sector
- **Profit Decay Logic**: Intelligent exit when momentum fades
- **End-of-Day Liquidation**: Automatic position closure

### **Market Intelligence**
- **Sentiment Analysis**: FinBERT + VADER ensemble for news sentiment
- **Volume Spike Detection**: 1.2x average volume requirement
- **Support/Resistance Analysis**: Technical level confirmation
- **VWAP Filtering**: Price-volume relationship validation
- **Market Regime Detection**: Bull/Bear/Neutral market adaptation

### **Monitoring & Alerts**
- **Discord Integration**: Real-time trade alerts and notifications
- **Google Sheets Logging**: Comprehensive trade and performance tracking
- **Live Accuracy Tracking**: Per-ticker model performance monitoring
- **Daily Performance Reports**: Automated P&L and analytics

## 📋 Requirements

- Python 3.9-3.11
- Alpaca Trading Account (Paper or Live)
- 2GB+ RAM recommended
- Stable internet connection

## 🛠️ Quick Setup

### 1. Clone Repository
\`\`\`bash
git clone https://github.com/smithrs12/ai-trading-bot.git
cd ai-trading-bot
\`\`\`

### 2. Install Dependencies
\`\`\`bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install requirements
pip install -r requirements.txt
\`\`\`

### 3. Configure Environment
\`\`\`bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
\`\`\`

### 4. Run Setup Check
\`\`\`bash
# Make executable (Linux/Mac)
chmod +x run_setup_check.sh
./run_setup_check.sh

# Or run directly
python setup_check.py
\`\`\`

### 5. Start Trading
\`\`\`bash
python main.py
\`\`\`

## 🔑 Required API Keys

### Alpaca Trading (Required)
1. Sign up at [Alpaca Markets](https://alpaca.markets/)
2. Get API keys from dashboard
3. **Start with Paper Trading** (free)

\`\`\`env
APCA_API_KEY_ID=your_key_here
APCA_API_SECRET_KEY=your_secret_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets
\`\`\`

### Discord Alerts (Optional)
1. Create Discord server
2. Server Settings → Integrations → Webhooks
3. Copy webhook URL

\`\`\`env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
\`\`\`

### News API (Optional)
1. Sign up at [NewsAPI](https://newsapi.org/)
2. Get free API key (500 requests/day)

\`\`\`env
NEWS_API_KEY=your_news_api_key
\`\`\`

## 🚀 Deployment Options

### Option 1: Render.com (Recommended)
1. Fork this repository
2. Connect to [Render.com](https://render.com)
3. Create new Web Service
4. Set environment variables
5. Deploy automatically

### Option 2: Docker
\`\`\`bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
\`\`\`

### Option 3: VPS/Cloud Server
\`\`\`bash
# Ubuntu/Debian setup
sudo apt update
sudo apt install python3 python3-pip git
git clone <repo-url>
cd ai-trading-bot
pip3 install -r requirements.txt
nohup python3 main.py &
\`\`\`

## 📊 Performance Monitoring

### Real-time Monitoring
- **Discord Alerts**: Instant trade notifications
- **Console Logs**: Detailed real-time information
- **Google Sheets**: Automated logging and tracking

### Key Metrics Tracked
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Daily P&L**: Profit and loss tracking
- **Model Accuracy**: Live prediction accuracy per ticker

### Log Files
- \`logs/trading_bot.log\`: General application logs
- \`logs/trades.log\`: All trade executions
- \`logs/errors.log\`: Error tracking
- \`performance/\`: Daily performance reports

## ⚙️ Configuration

### Trading Parameters
\`\`\`python
# config.py
MAX_PORTFOLIO_RISK = 0.02      # 2% max risk per trade
MAX_DAILY_TRADES = 10          # Maximum trades per day
MAX_POSITIONS = 5              # Maximum concurrent positions
KELLY_MULTIPLIER = 0.25        # Conservative Kelly fraction
\`\`\`

### Model Thresholds
\`\`\`python
MIN_PREDICTION_CONFIDENCE = 0.6    # 60% minimum confidence
META_MODEL_THRESHOLD = 0.5         # Meta model approval threshold
SENTIMENT_OVERRIDE_THRESHOLD = -0.5 # Sentiment veto threshold
\`\`\`

### Risk Management
\`\`\`python
STOP_LOSS_MULTIPLIER = 2.0         # 2x ATR stop loss
PROFIT_TARGET_MULTIPLIER = 3.0     # 3x ATR profit target
TRAILING_STOP_THRESHOLD = 0.03     # 3% profit before trailing
\`\`\`

## 🧠 Machine Learning Models

### Short-Term Model (2-day horizon)
- **Data**: 5-minute candles
- **Features**: Technical indicators, volume, momentum
- **Retraining**: Every 5 minutes
- **Purpose**: Entry timing optimization

### Medium-Term Model (15-day horizon)
- **Data**: Daily bars
- **Features**: Fundamental + technical analysis
- **Retraining**: Every hour
- **Purpose**: Trend direction confirmation

### Meta Model
- **Features**: 16+ combined signals
- **Purpose**: Final trade approval/rejection
- **Retraining**: Daily with outcome data

### Q-Learning System
- **Neural Network**: Experience replay
- **Reward Shaping**: Sentiment + volatility adjusted
- **Purpose**: Dynamic strategy optimization

## 📈 Trading Strategy

### Entry Criteria (ALL must be met)
1. **Dual Model Agreement**: Both short & medium models bullish
2. **Meta Model Approval**: Meta model score > threshold
3. **Volume Confirmation**: Volume > 1.2x recent average
4. **VWAP Filter**: Price above VWAP
5. **Momentum Check**: Price momentum > 0.5%
6. **Sentiment Override**: News sentiment > -0.5
7. **Support/Resistance**: No major resistance overhead

### Position Sizing
- **Kelly Criterion**: Optimal fraction based on win rate & odds
- **Confidence Adjustment**: Size scaled by prediction confidence
- **Risk Limits**: Maximum 2% portfolio risk per trade
- **Sector Limits**: Maximum 30% per sector

### Exit Strategy
- **Dynamic Stops**: 2x ATR-based stop loss
- **Profit Targets**: 3x ATR-based profit target
- **Trailing Stops**: Activated at 3% profit
- **Profit Decay**: Exit when momentum fades 50%
- **EOD Liquidation**: All positions closed at 3:45 PM ET

## 🔧 Troubleshooting

### Common Issues

**Build Failures**
\`\`\`bash
# Update pip and try again
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
\`\`\`

**API Connection Issues**
\`\`\`bash
# Test API connection
python setup_check.py
\`\`\`

**Memory Issues**
- Reduce watchlist size in \`config.py\`
- Increase server memory allocation
- Use lighter model parameters

**Rate Limiting**
- Increase delays between API calls
- Check API usage limits
- Consider upgrading API plan

### Debug Mode
\`\`\`python
# In main.py
DEBUG = True
LOG_LEVEL = "DEBUG"
\`\`\`

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/smithrs12/ai-trading-bot/issues)
- **Documentation**: This README + inline code comments
- **Community**: Discord server (link in repo)

## ⚠️ Disclaimer

**This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always start with paper trading and never risk more than you can afford to lose.**

## 📄 License

MIT License - see LICENSE file for details.

---

**🚀 Ready to maximize your trading profitability? Start with paper trading and let the AI do the work!**
