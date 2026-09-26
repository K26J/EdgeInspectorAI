# Use a lightweight Python base image to minimize container size
FROM python:3.10-slim

WORKDIR /workspace

# Install system dependencies required by Pillow and ONNX Runtime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configurations
COPY ./src ./src
COPY ./app ./app
COPY ./config ./config

# Expose the API port
EXPOSE 8000

# Start the Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]