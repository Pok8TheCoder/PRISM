FROM python:3.11-slim

WORKDIR /app

# HTTP client for the requests-based objective bots (lab_defacement,
# lab_key_theft, lab_cred_theft) -- raw-socket recon bots need nothing extra.
RUN pip install --no-cache-dir requests

# Source tree is bind-mounted from host at runtime (/app)
CMD ["sleep", "infinity"]
