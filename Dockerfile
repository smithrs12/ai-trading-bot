FROM python:3.9-slim

WORKDIR /app

# --- Install system dependencies required to build TA-Lib ---
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    gcc \
    g++ \
    make \
    libffi-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Build TA-Lib from source ---
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# --- Make sure linker can find the library ---
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

# --- Install TA-Lib Python wrapper ---
RUN pip install --no-cache-dir TA-Lib

# --- Copy requirements first for Docker caching ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy all source code ---
COPY . .

# --- Create required folders ---
RUN mkdir -p logs models/short models/medium models/meta models/q_learning \
    models/garch models/regime performance backtests support_resistance \
    volume_profiles sector_analysis fibonacci_analysis elliott_waves \
    harmonic_patterns ichimoku_analysis microstructure volatility_models \
    regime_analysis

# --- Expose port for health checks (optional) ---
EXPOSE 5000

# --- Run the main bot ---
CMD ["python", "main.py"]
