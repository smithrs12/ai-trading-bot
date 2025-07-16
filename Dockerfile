# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Make start.sh executable
RUN chmod +x start.sh

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --timeout 60 -r requirements.txt

# Environment variable to show logs in real time
ENV PYTHONUNBUFFERED=1

# Entry point
CMD ["./start.sh"]
