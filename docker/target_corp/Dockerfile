FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    python3 \
    python3-pip \
    tcpdump \
    procps \
    iproute2 \
    iptables \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY webapp/ /app/webapp/
COPY decoys.py /app/decoys.py
COPY dns_stub.py /app/dns_stub.py
RUN pip3 install --break-system-packages --no-cache-dir -r /app/webapp/requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh /app/decoys.py /app/dns_stub.py && chmod +x /entrypoint.sh

CMD ["/bin/bash", "/entrypoint.sh"]
