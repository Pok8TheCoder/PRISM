FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    python3 \
    python3-pip \
    tcpdump \
    vsftpd \
    procps \
    iproute2 \
    iptables \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY webapp/ /app/webapp/
RUN pip3 install --break-system-packages --no-cache-dir -r /app/webapp/requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

CMD ["/bin/bash", "/entrypoint.sh"]
