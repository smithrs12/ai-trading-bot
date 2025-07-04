FROM python:3.11-slim

WORKDIR /app

# --- Install system build tools ---
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# --- Install TA-Lib from source ---
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# --- Ensure linker can find TA-Lib ---
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV TA_LIBRARY_PATH=/usr/local/lib

# --- Install Python dependencies ---
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir ta-lib
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy your app code ---
COPY . .

# --- Run your app ---
CMD ["python", "main.py"]
