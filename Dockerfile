FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev git wget && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements and install them
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

ENV PYTHONUNBUFFERED=1

# Default command - runs the main.py script
CMD ["python3", "main.py"]
