# Dockerfile - Place this in the root directory (FRONCORT/)
# This handles the FastAPI backend
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for MySQL connector
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file from sql_agent directory
COPY sql_agent/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application (main.py, sql_agent/, etc.)
COPY main.py .
COPY sql_agent/ ./sql_agent/

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI application (main.py is in root)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]