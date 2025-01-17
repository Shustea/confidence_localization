FROM python:3.8-slim

WORKDIR /workspace
COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", ".\confidence_localization\train.py"]