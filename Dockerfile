FROM python:3.9-slim

WORKDIR /app

# Install system dependencies and TA-Lib build tools
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Build TA-Lib from source and install it to /usr/local
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Fix: update linker cache so it finds /usr/local/lib
RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/ta-lib.conf && ldconfig

# Install TA-Lib Python wrapper
RUN pip install --no-cache-dir TA-Lib

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models/short models/medium models/meta models/q_learning \
    models/garch models/regime performance backtests support_resistance \
    volume_profiles sector_analysis fibonacci_analysis elliott_waves \
    harmonic_patterns ichimoku_analysis microstructure volatility_models \
    regime_analysis

# Expose port for health check or UI
EXPOSE 5000

# Start the bot
CMD ["python", "main.py"]
