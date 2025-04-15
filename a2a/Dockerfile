# Use a lightweight Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
COPY a2a_adapter.py .
COPY .well-known .well-known

RUN pip install --no-cache-dir -r requirements.txt

# Expose the A2A port (default 10000)
EXPOSE 10000

# Start FastAPI using uvicorn
CMD ["uvicorn", "a2a_adapter:app", "--host", "0.0.0.0", "--port", "10000"]
