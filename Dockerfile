FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p logs models/short models/medium models/meta models/q_learning \
    models/garch models/regime performance backtests support_resistance \
    volume_profiles sector_analysis fibonacci_analysis elliott_waves \
    harmonic_patterns ichimoku_analysis microstructure volatility_models \
    regime_analysis

EXPOSE 5000

CMD ["python", "main.py"]
