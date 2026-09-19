import hashlib
import random
import subprocess
from collections import Counter
from pathlib import Path

root = Path("/pok8/prism/PRISM/data/raw/bulk_5s")
pcaps = list(root.rglob("*prism-p*.pcap"))
random.seed(42)
sample = random.sample(pcaps, min(500, len(pcaps)))
hashes = []
for p in sample:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    hashes.append(h)
counts = Counter(hashes)
dup_groups = sum(1 for c in counts.values() if c > 1)
print(f"sampled={len(sample)} unique={len(counts)} dup_groups={dup_groups}")
