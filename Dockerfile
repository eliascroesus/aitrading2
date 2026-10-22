FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    HIVE_DATA_DIR=/data \
    HIVE_WORKERS=2

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY hive ./hive

# /data holds the database and the market-data cache; mount a volume there so it survives redeploys
VOLUME ["/data"]
EXPOSE 8000
CMD ["python", "-m", "hive"]
