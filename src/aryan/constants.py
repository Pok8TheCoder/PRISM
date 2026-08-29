"""Trimmed constants ported from Aryan's `origin/aryan:src/utils/constants.py`.

Only the pieces `src/aryan/world_model.py` and the forecast-player session
builder actually need -- MITRE stage count/names for the classification head
and label bookkeeping.
"""

MITRE_STAGES = {
    "Benign": 0,
    "Reconnaissance": 1,
    "Initial Access": 2,
    "Lateral Movement": 3,
    "Command & Control": 4,
    "Exfiltration": 5,
    "Impact": 6,
}

MITRE_STAGES_INV = {v: k for k, v in MITRE_STAGES.items()}

NUM_MITRE_STAGES = len(MITRE_STAGES)
