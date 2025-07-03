FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for TA-Lib and other libraries
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
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source (C library)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# ✅ Tell the linker where to find libta_lib.so
ENV LD_LIBRARY_PATH="/usr/lib:$LD_LIBRARY_PATH"
ENV CFLAGS="-I/usr/include"
ENV LDFLAGS="-L/usr/lib"

# Install Python wrapper for TA-Lib
RUN pip install --no-cache-dir TA-Lib

# Copy requirements first for better caching
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

# Expose port for health checks
EXPOSE 5000

# Run the application
CMD ["python", "main.py"]
