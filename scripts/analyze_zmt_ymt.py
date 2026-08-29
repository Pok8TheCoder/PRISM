"""Post-hoc analysis of the ZMT.01 vs YMT.01 run.

Breaks the headline numbers into the parts that actually matter: per-class
deltas, the benign false-positive decomposition, and the scan/flood family that
the live adversarial loop was collapsing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RES = ROOT / "results" / "zmt_ymt"
MODELS = ("ZMT.01", "YMT.01", "AMT.01")

# Classes the 20-minute adversarial loop kept confusing with one another.
SCAN_FAMILY = [
    "T1046_service_scan", "T1595_active_scan", "T1498_network_dos",
    "T1499_http_flood", "T1049_connections_discovery", "T1018_remote_discovery",
]


def main() -> None:
    m = json.load(open(RES / "metrics.json"))
    cn = m["class_names"]
    idx = {c: i for i, c in enumerate(cn)}
    sup = m["test_support"]
    benign = idx["Benign"]
    infil = idx["T1021_remote_services"]

    out: dict = {}

    models = [n for n in MODELS if n in m["summary"]]

    print("=" * 82)
    print("1. HEADLINE (mean +/- std over 3 seeds)")
    print("=" * 82)
    hdr = "".join(f"{n:>20}" for n in models)
    print(f"{'Metric':<22}{hdr}")
    print("-" * 82)
    keys = ["accuracy", "f1_macro", "f1_weighted", "binary_f1",
            "binary_recall", "binary_precision", "binary_fpr", "binary_auc"]
    for k in keys:
        cells = "".join(
            f"{m['summary'][n][k]['mean']:>13.4f} ±{m['summary'][n][k]['std']:.4f}"
            for n in models
        )
        print(f"{k:<22}{cells}")
    out["headline"] = {k: {n: m["summary"][n][k] for n in models} for k in keys}

    print("\n" + "=" * 78)
    print("2. FALSE POSITIVES: benign windows and where they go")
    print("=" * 78)
    fp_break = {}
    for n in models:
        C = np.array(m["confusion"][n])
        row = C[benign]
        total = row.sum()
        to_infil = int(row[infil])
        other_fp = int(total - row[benign] - to_infil)
        fp_break[n] = {
            "benign_support": int(total),
            "correct": int(row[benign]),
            "to_infiltration": to_infil,
            "to_other_attacks": other_fp,
            "fpr_all": float((total - row[benign]) / total),
            "fpr_excl_infiltration": float(other_fp / total),
        }
        print(f"\n  {n}: {total} benign test windows")
        print(f"    correctly benign          {row[benign]:>5}")
        print(f"    -> T1021 (CIC Infilt.)    {to_infil:>5}   "
              f"(the known-ambiguous CIC pair)")
        print(f"    -> any other attack       {other_fp:>5}")
        print(f"    FPR all attacks           {fp_break[n]['fpr_all']:.4f}")
        print(f"    FPR excl. Infiltration    {fp_break[n]['fpr_excl_infiltration']:.4f}")
    out["false_positives"] = fp_break

    print("\n" + "=" * 78)
    print("3. SCAN / FLOOD FAMILY  (the classes the live loop collapsed)")
    print("=" * 78)
    print(f"{'Class':<32}{'support':>8}" + "".join(f"{n:>10}" for n in models))
    print("-" * 78)
    fam, acc = {}, {n: [] for n in models}
    for c in SCAN_FAMILY:
        if c not in sup:
            continue
        vals = {n: m["per_class_f1"][n].get(c, 0.0) for n in models}
        for n in models:
            acc[n].append(vals[n])
        fam[c] = {"support": sup[c], **vals}
        print(f"{c:<32}{sup[c]:>8}" + "".join(f"{vals[n]:>10.4f}" for n in models))
    print("-" * 78)
    print(f"{'FAMILY MEAN':<32}{'':>8}"
          + "".join(f"{np.mean(acc[n]):>10.4f}" for n in models))
    out["scan_family"] = fam
    out["scan_family_mean"] = {n: float(np.mean(acc[n])) for n in models}

    print("\n" + "=" * 78)
    print("4. PER-CLASS F1 (best seed), sorted by YMT.01 gain over ZMT.01")
    print("=" * 78)
    rows = []
    for c in sup:
        vals = {n: m["per_class_f1"][n].get(c, 0.0) for n in models}
        rows.append((c, sup[c], vals, vals["YMT.01"] - vals["ZMT.01"]))
    rows.sort(key=lambda r: -r[3])
    print(f"{'Class':<32}{'sup':>6}" + "".join(f"{n:>9}" for n in models)
          + f"{'Y-Z':>9}")
    print("-" * 78)
    for c, s, vals, d in rows:
        print(f"{c:<32}{s:>6}"
              + "".join(f"{vals[n]:>9.3f}" for n in models) + f"{d:>+9.3f}")
    win = sum(1 for r in rows if r[3] > 0.01)
    loss = sum(1 for r in rows if r[3] < -0.01)
    print("-" * 78)
    print(f"YMT.01 better than ZMT.01 on {win} classes, worse on {loss}, "
          f"tied on {len(rows) - win - loss}")
    out["per_class"] = [
        {"class": c, "support": s, **vals, "delta": d} for c, s, vals, d in rows
    ]
    out["class_win_loss"] = {"ymt_better": win, "ymt_worse": loss,
                             "tied": len(rows) - win - loss}

    print("\n" + "=" * 78)
    print("5. LAB-CAPTURE-ONLY SUBSET (PCAP windows, excludes CIC CSV)")
    print("=" * 78)
    for n in models:
        p = m["pcap_only"][n]
        print(f"  {n}: n={p['n']}  acc={p['accuracy']:.4f}  macroF1={p['f1_macro']:.4f}")
    out["pcap_only"] = {n: m["pcap_only"][n] for n in models}

    with open(RES / "analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {RES / 'analysis.json'}")


if __name__ == "__main__":
    main()
