FROM python:3.9-slim

WORKDIR /workspace
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", ".\confidence_localization\train.py"]