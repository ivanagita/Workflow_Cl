FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (optional tapi aman)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command (sesuaikan mau basic / advanced)
CMD ["python", "modelling_tuning.py"]
