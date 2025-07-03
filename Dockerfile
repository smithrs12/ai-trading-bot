FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for TA-Lib
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Build TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Tell linker where TA-Lib is
ENV LD_LIBRARY_PATH=/usr/local/lib

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

# Optional port expose
EXPOSE 5000

# Run your bot
CMD ["python", "main.py"]
