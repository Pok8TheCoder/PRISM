#!/usr/bin/env python3
"""Attack dispatcher — runs inside attacker-bot container or from host.

Usage:
  python attack_script.py <target_ip> <class_id> [evasion]

Examples:
  python attack_script.py 172.17.0.2 T1046_service_scan none
  python attack_script.py 172.17.0.2 ssh_bruteforce random_timing   # legacy alias
"""

from __future__ import annotations

import sys
from pathlib import Path


def _setup_import_path() -> None:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent,  # repo root (src/adversarial/attack_script.py)
        Path("/app"),
        Path.cwd(),
    ]
    for root in candidates:
        if (root / "src" / "adversarial" / "bots").is_dir():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return


_setup_import_path()

from src.adversarial.bots import list_bots, resolve_class_id, run_bot  # noqa: E402


def main() -> int:
    target_ip = sys.argv[1] if len(sys.argv) > 1 else "172.17.0.2"
    class_arg = sys.argv[2] if len(sys.argv) > 2 else "T1046_service_scan"
    evasion = sys.argv[3] if len(sys.argv) > 3 else "none"

    if class_arg in ("--list", "-l"):
        for bot_id in list_bots():
            print(bot_id)
        return 0

    class_id = resolve_class_id(class_arg)
    flows = run_bot(class_id, target_ip, evasion)
    print(f"DONE|class={class_id}|flows={flows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
