"""IPS controller — block attacker source IP on target-server via iptables.

Used by ``scripts/live_ips_redteam.py`` when RAMX P(attack) >= threshold.
Only drops traffic from the red-team (or specified) container IP; benign-client
is unaffected.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from src.adversarial.lab_config import REDTEAM_CONTAINER, TARGET_CONTAINER


@dataclass
class BlockResult:
    blocked: bool
    source_ip: str
    message: str = ""


def get_container_ip(container_name: str) -> str:
    """Return the first IPv4 address on the container's default network."""
    r = subprocess.run(
        [
            "docker", "inspect", "-f",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            container_name,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"docker inspect failed for {container_name}: {r.stderr.strip()}")
    ip = (r.stdout or "").strip()
    if not ip:
        raise RuntimeError(f"No IP found for container {container_name}")
    return ip


def _exec_target_iptables(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", TARGET_CONTAINER, "iptables", *args],
        capture_output=True,
        text=True,
        timeout=15,
    )


def clear_blocks(target_container: str = TARGET_CONTAINER) -> None:
    """Remove all PRISM IPS DROP rules (comment match via chain flush of custom rules)."""
    # List and delete INPUT rules that drop (best-effort; lab container is ephemeral)
    r = _exec_target_iptables(["-L", "INPUT", "-n", "--line-numbers"])
    if r.returncode != 0:
        return
    to_delete: list[int] = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "DROP":
            try:
                to_delete.append(int(parts[0]))
            except ValueError:
                continue
    for num in sorted(to_delete, reverse=True):
        _exec_target_iptables(["-D", "INPUT", str(num)])


def block_ip(
    source_ip: str,
    target_container: str = TARGET_CONTAINER,
) -> BlockResult:
    """Insert DROP rule for source_ip on target INPUT chain."""
    if not source_ip:
        return BlockResult(False, source_ip, "empty ip")
    check = _exec_target_iptables(["-C", "INPUT", "-s", source_ip, "-j", "DROP"])
    if check.returncode == 0:
        return BlockResult(True, source_ip, "already blocked")
    r = _exec_target_iptables(["-I", "INPUT", "1", "-s", source_ip, "-j", "DROP"])
    if r.returncode != 0:
        return BlockResult(False, source_ip, r.stderr.strip() or "iptables failed")
    return BlockResult(True, source_ip, "blocked")


def get_bridge_gateway(target_container: str = TARGET_CONTAINER) -> str:
    """IPv4 default gateway inside the target — host-published traffic sources from here."""
    r = subprocess.run(
        ["docker", "exec", target_container, "ip", "route", "show", "default"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if r.returncode != 0:
        return ""
    parts = (r.stdout or "").split()
    if "via" in parts:
        i = parts.index("via")
        if i + 1 < len(parts):
            return parts[i + 1].strip()
    return ""


def block_demo_attackers(
    extra_ips: list[str] | None = None,
    *,
    include_redteam: bool = True,
    include_host_gateway: bool = True,
    target_container: str = TARGET_CONTAINER,
) -> list[BlockResult]:
    """Block red-team container and/or host-browser NAT IP. Benign clients stay up."""
    results: list[BlockResult] = []
    ips: list[str] = []
    if include_redteam:
        try:
            ips.append(get_container_ip(REDTEAM_CONTAINER))
        except RuntimeError as exc:
            results.append(BlockResult(False, "", str(exc)))
    if include_host_gateway:
        gw = get_bridge_gateway(target_container)
        if gw:
            ips.append(gw)
    for ip in extra_ips or []:
        if ip:
            ips.append(ip)
    seen: set[str] = set()
    for ip in ips:
        if ip in seen:
            continue
        seen.add(ip)
        results.append(block_ip(ip, target_container=target_container))
    return results


def block_container(
    attacker_container: str,
    target_container: str = TARGET_CONTAINER,
) -> BlockResult:
    ip = get_container_ip(attacker_container)
    return block_ip(ip, target_container=target_container)
