FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for vl-convert
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for vl-convert backend)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Create logs directory
RUN mkdir -p logs

# Install Python dependencies
RUN pip install --no-cache-dir vl-convert-python

RUN pip install pydantic

# Copy project files
COPY . .

# Optional: If you still want to use pyproject.toml-based setup
# COPY pyproject.toml ./
# RUN pip install .

# Set PYTHONPATH if needed (optional now that you're in /app)
ENV PYTHONPATH=/app

# Run your standalone tool-based server 
CMD ["python", "-u", "server.py"]
