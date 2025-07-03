FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including ta-lib and other libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    build-essential \
    libffi-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libta-lib0 \
    libta-lib-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python wrapper for TA-Lib (no source build needed)
RUN pip install --no-cache-dir TA-Lib

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create necessary directories
RUN mkdir -p logs models/short models/medium models/meta models/q_learning \
    models/garch models/regime performance backtests support_resistance \
    volume_profiles sector_analysis fibonacci_analysis elliott_waves \
    harmonic_patterns ichimoku_analysis microstructure volatility_models \
    regime_analysis

# Optional: expose health check port
EXPOSE 5000

# Run your bot
CMD ["python", "main.py"]
