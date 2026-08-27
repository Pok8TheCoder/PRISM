FROM python:3.11-slim

WORKDIR /app
COPY benign_traffic.py /app/benign_traffic.py

ENV TARGET_HOST=target-server
ENV BENIGN_INTERVAL=3.0

CMD ["python3", "-u", "/app/benign_traffic.py"]
