FROM python:3.8-slim

RUN useradd -m -s /bin/bash shustea

WORKDIR /workspace

RUN chown -R shustea:shustea /workspace

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER shustea

# Copy the project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --user --no-cache-dir -r requirements.txt

# Default command to run the application
CMD ["python", "./confidence_localization/confidence_localization/train.py"]