#!/bin/bash
set -e

mkdir -p /var/run/sshd /app/data /app/config
echo "root:prism-lab" | chpasswd
sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

python3 /app/webapp/seed_db.py
service ssh start

python3 /app/decoys.py &
python3 /app/dns_stub.py &

cd /app/webapp
WORKERS="${GUNICORN_WORKERS:-4}"
gunicorn --bind 0.0.0.0:80 --workers "$WORKERS" --access-logfile - --error-logfile - app:app &

echo "Harborline target ready (HTTP:80, SSH:22, DNS:53, decoy TCP ports)"
exec tail -f /dev/null
