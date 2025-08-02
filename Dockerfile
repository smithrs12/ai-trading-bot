# === Base Python Image with Slim Footprint ===
FROM python:3.10-slim

# === Set Working Directory ===
WORKDIR /app

# === System-Level Dependencies ===
# These are essential for ta-lib, matplotlib, transformers, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# === Python Dependencies ===
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout 60 -r requirements.txt

# === Copy Application Code ===
COPY . .

# === Executable Entrypoint ===
RUN chmod +x start.sh

# === Prevent Output Buffering for Logs ===
ENV PYTHONUNBUFFERED=1

# === Default Entrypoint ===
CMD ["./start.sh"]
