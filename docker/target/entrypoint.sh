#!/bin/bash
set -e

mkdir -p /var/run/sshd
echo "root:prism-lab" | chpasswd
sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

echo "<html><body><h1>PRISM Target Server</h1></body></html>" > /var/www/html/index.html
mkdir -p /var/www/html/admin /var/www/html/login

service ssh start
service apache2 start

if [ -f /etc/vsftpd.conf ]; then
  service vsftpd start || true
fi

echo "PRISM target-server ready (SSH:22, HTTP:80)"
exec tail -f /dev/null
