#!/usr/bin/env python3
"""Lab Console BFF + UI integration smoke test."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

API = "http://127.0.0.1:8790/api"
FAILURES: list[str] = []


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=10) as r:
        return json.loads(r.read().decode())


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def patch(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="PATCH",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        msg = f"  FAIL  {name}" + (f" — {detail}" if detail else "")
        print(msg, file=sys.stderr)
        FAILURES.append(name)


def main() -> int:
    print("=== Lab Console API smoke test ===")

    try:
        h = get("/health")
        check("health", h.get("ok") is True)
    except Exception as exc:
        check("health", False, str(exc))
        return 1

    try:
        dash = get("/dashboard")
        check("dashboard.containers", isinstance(dash.get("containers"), list) and len(dash["containers"]) > 0)
        check("dashboard.models", isinstance(dash.get("models"), list))
        check("dashboard.metrics", isinstance(dash.get("metricsTimeseries"), list))
    except Exception as exc:
        check("dashboard", False, str(exc))

    try:
        rec = get("/sessions/default/state?mode=recorded")
        check("session.recorded.models", len(rec.get("models", [])) >= 2, f"{len(rec.get('models', []))} models")
        check("session.recorded.actual", len(rec.get("actual", [])) > 0)
        check("session.recorded.duration", rec.get("durationSec", 0) > 0)
    except Exception as exc:
        check("session.recorded", False, str(exc))

    try:
        cfg = get("/lab-config")
        check("lab-config.horizon", cfg.get("horizonSec") == 60 or cfg.get("horizonSec") == 60.0)
        cfg2 = patch("/lab-config", {"horizonSec": 60})
        check("lab-config.patch", "horizonSec" in cfg2)
    except Exception as exc:
        check("lab-config", False, str(exc))

    try:
        live = get("/sessions/default/state?mode=live")
        check("session.live", "mode" in live, f"mode={live.get('mode')}")
    except Exception as exc:
        check("session.live", False, str(exc))

    try:
        pol = patch("/sessions/default/policy", {"policyMode": "ips"})
        check("policy.ips", pol.get("policyMode") == "ips")
        pol2 = patch("/sessions/default/policy", {"policyMode": "ids"})
        check("policy.ids", pol2.get("policyMode") == "ids")
    except Exception as exc:
        check("policy", False, str(exc))

    try:
        scripts = get("/scripts")
        check("scripts.list", len(scripts.get("scripts", [])) >= 8)
    except Exception as exc:
        check("scripts.list", False, str(exc))

    try:
        job = post("/scripts/run", {"script_id": "killchain-recon"})
        check("scripts.run", job.get("status") == "queued", job.get("job_id", ""))
    except Exception as exc:
        check("scripts.run", False, str(exc))

    import time

    time.sleep(2)
    try:
        logs = get("/logs?since=0")
        lines = logs.get("lines", [])
        check("logs.stream", len(lines) >= 1, f"{len(lines)} lines")
        started = any("killchain-recon" in ln.get("text", "") or "starting" in ln.get("text", "") for ln in lines)
        check("logs.script_queued", started)
    except Exception as exc:
        check("logs", False, str(exc))

    try:
        models = get("/models")
        check("models.registry", len(models.get("models", [])) >= 3)
    except Exception as exc:
        check("models.registry", False, str(exc))

    try:
        recs = get("/recordings")
        check("recordings.list", isinstance(recs.get("recordings"), list))
    except Exception as exc:
        check("recordings.list", False, str(exc))

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}", file=sys.stderr)
        return 1
    print("ALL API CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
