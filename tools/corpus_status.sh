#!/bin/bash
ROOT=/pok8/prism/PRISM/data/raw/bulk_5s
CKPT=/pok8/prism/PRISM/data/raw/bulk_5s_parallel/checkpoint.json
CAP=107374182400  # 100 GiB

echo "=== CHECKPOINT (progress cap source) ==="
cat "$CKPT"
echo
echo "=== CORPUS ON DISK ==="
du -sb "$ROOT"
du -sh "$ROOT"
echo
echo "=== MANIFEST ==="
wc -l "$ROOT/manifest.jsonl"
python3 <<'PY'
import json
from pathlib import Path
ok=fail=0
mb=0
parallel=0
for line in Path("/pok8/prism/PRISM/data/raw/bulk_5s/manifest.jsonl").read_text().splitlines():
    r=json.loads(line)
    if r.get("ok"):
        ok+=1
        mb+=r.get("bytes",0)
        if r.get("project"):
            parallel+=1
    else:
        fail+=1
print(f"ok_rows={ok} fail_rows={fail} parallel_ok={parallel}")
print(f"manifest_sum_bytes_ok={mb} ({mb/1e9:.2f} GB decimal)")
PY
echo
echo "=== PCAP FILES ==="
find "$ROOT" -name 'bulk_*.pcap' | wc -l
echo
echo "=== PROGRESS % (binary 100 GiB cap) ==="
BYTES=$(du -sb "$ROOT" | awk '{print $1}')
python3 -c "b=$BYTES; cap=$CAP; print(f'bytes={b} cap={cap} pct={100*b/cap:.1f}% remaining_gib={(cap-b)/1024**3:.2f}')"
echo
echo "=== PROCESS ==="
pgrep -af parallel_internet_corpus | grep -v pgrep || echo stopped
echo
echo "=== LOG ==="
tail -4 /pok8/prism/data/raw/bulk_5s_parallel/nohup.out
