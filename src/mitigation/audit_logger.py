"""
SOC Compliance & Incident Audit Logger for PRISM P-IPS.
Maintains an immutable, append-only JSONL log of every preemptive action,
approval, lease expiration, and blast-radius rejection, with CEF (Common Event Format) support.
"""

import os
import json
import time
import uuid
import datetime
from typing import Dict, List, Optional
from src.utils.logger import setup_logger

logger = setup_logger("MitigationAuditLogger")


class MitigationAuditLogger:
    """
    Append-only audit trail logger for SOC compliance and SIEM integration.
    """

    def __init__(self, log_path: str = "logs/mitigation_audit.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_event(
        self,
        event_type: str,
        target_ip: str,
        action_type: str = "DROP",
        risk_score: float = 1.0,
        lead_time_seconds: int = 45,
        ttl_seconds: int = 300,
        strike_count: int = 1,
        driving_feature: str = "Unknown",
        reason: str = "",
        driver_used: str = "SANDBOX",
        extra_metadata: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        Appends an audit event entry with high-precision timestamping and CEF mapping.
        """
        now_epoch = time.time()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event_id = str(uuid.uuid4())[:12]

        # Common Event Format (CEF) standard for SIEM integration (Splunk, Elastic, Sentinel)
        severity = 10 if risk_score >= 0.90 else (7 if risk_score >= 0.75 else 4)
        cef_string = (
            f"CEF:0|PRISM|P-IPS|2.0|{event_type}|Preemptive Threat Neutralization|{severity}|"
            f"src={target_ip} act={action_type} cn1={lead_time_seconds} cn1Label=LeadAdvanceSec "
            f"cn2={ttl_seconds} cn2Label=LeaseTTL cs1={driving_feature} cs1Label=DrivingFeature "
            f"reason={reason}"
        )

        record = {
            "event_id": event_id,
            "timestamp_iso": now_iso,
            "timestamp_epoch": now_epoch,
            "event_type": event_type,
            "target_ip": target_ip,
            "action_type": action_type,
            "risk_score": float(risk_score),
            "lead_time_seconds": int(lead_time_seconds),
            "ttl_seconds": int(ttl_seconds),
            "strike_count": int(strike_count),
            "driving_feature": driving_feature,
            "reason": reason,
            "driver_used": driver_used,
            "cef_format": cef_string,
            "metadata": extra_metadata or {}
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to append mitigation audit record: {e}")

        return record

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, any]]:
        """Reads the most recent audit records from disk for dashboard visualization."""
        if not os.path.exists(self.log_path):
            return []

        events = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in reversed(lines[-limit:]):
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read mitigation audit log: {e}")

        return events
