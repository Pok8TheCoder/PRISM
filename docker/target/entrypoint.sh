#!/bin/bash
set -e

mkdir -p /var/run/sshd
echo "root:prism-lab" | chpasswd
sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Fresh per-boot session-signing secret + seeded sqlite DB/fixtures for the
# vulnerable webapp (idempotent -- safe if the container restarts without a
# full `lab_ctl.py reset`).
rm -f /app/config/secret.key
python3 /app/webapp/seed_db.py

service ssh start

if [ -f /etc/vsftpd.conf ]; then
  service vsftpd start || true
fi

cd /app/webapp
gunicorn --bind 0.0.0.0:80 --workers 2 --access-logfile - --error-logfile - app:app &

echo "PRISM target-server ready (SSH:22, HTTP:80 -> Flask lab webapp)"
exec tail -f /dev/null
