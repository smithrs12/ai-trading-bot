# Use slim Python base with build tools
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements separately for caching
COPY requirements.txt .

# Install system dependencies (needed for ta-lib, transformers, matplotlib, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --timeout 60 -r requirements.txt

# Copy full project
COPY . .

# Make shell script executable
RUN chmod +x start.sh

# Prevents output buffering
ENV PYTHONUNBUFFERED=1

# Entrypoint
CMD ["./start.sh"]
