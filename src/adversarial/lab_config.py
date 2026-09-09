"""Configuration for the isolated PRISM adversarial Docker lab."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Ensure Docker CLI is in PATH on Windows
_DOCKER_BIN_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "DockerDesktop" / "resources" / "bin",
    Path(r"C:\Program Files\Docker\Docker\resources\bin"),
]
for _p in _DOCKER_BIN_CANDIDATES:
    if _p.is_dir() and str(_p) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(_p) + os.pathsep + os.environ.get("PATH", "")

COMPOSE_FILE = ROOT / "docker" / "docker-compose.yml"
COMPOSE_PROJECT = "prism"

TARGET_HOST = "target-server"
ATTACKER_CONTAINER = "attacker-bot"
REDTEAM_CONTAINER = "redteam"
TARGET_CONTAINER = "target-server"
BENIGN_CONTAINER = "benign-client"

REDTEAM_EVENTS_CONTAINER_PATH = "/events/redteam.jsonl"
REDTEAM_EVENTS_HOST_PATH = ROOT / "data" / "lab_events" / "redteam.jsonl"
IPS_OUT_ROOT = ROOT / "results" / "ips_redteam"

SAVE_DIR = ROOT / "data" / "raw" / "adversarial"
MISSED_DIR = SAVE_DIR / "missed"
CHECKPOINT_PATH = ROOT / "models" / "checkpoints" / "world_model_multiclass.pth"

EPOCHS_PER_CYCLE = 10
BATCH_SIZE = 64
MAX_EVASION_ATTEMPTS = 4
DETECTION_THRESHOLD = 0.55
RETRAIN_BATCHES = 3

# Subset of catalog bots used in the default adversarial rotation
DEFAULT_ATTACK_CLASSES = [
    "T1110_ssh_bruteforce",
    "T1046_service_scan",
    "T1499_http_flood",
    "T1071_http_beacon",
    "T1041_exfil_c2",
    "T1595_active_scan",
]
