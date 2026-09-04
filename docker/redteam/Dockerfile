FROM python:3.11-slim

WORKDIR /agent
RUN pip install --no-cache-dir requests==2.32.3

COPY redteam_agent.py /agent/redteam_agent.py
COPY redteam_agent_slow.py /agent/redteam_agent_slow.py

CMD ["sleep", "infinity"]
