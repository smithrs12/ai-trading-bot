FROM python:3.9.18-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Update pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy and install requirements (NO TA-Lib)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models/short models/medium models/meta models/q_learning \
    performance backtests support_resistance volume_profiles sector_analysis \
    fibonacci_analysis elliott_waves harmonic_patterns ichimoku_analysis \
    microstructure volatility_models regime_analysis

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start command
CMD ["./start.sh"]
