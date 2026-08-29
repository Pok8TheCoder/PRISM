"""Quick check: how separable is Infiltration from Benign, at flow level?"""

import pandas as pd

df = pd.read_csv("data/raw/wednesday_28_02_2018.csv", low_memory=False)
df = df[df["Label"] != "Label"]
cols = ["Flow Duration", "Flow Byts/s", "Flow Pkts/s", "Tot Fwd Pkts",
        "SYN Flag Cnt", "Fwd Pkt Len Mean"]
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

b = df[df["Label"].str.lower() == "benign"]
a = df[df["Label"].str.lower() == "infilteration"]

print(f"n_benign={len(b)}  n_infiltration={len(a)}\n")
print(f"{'feature':<18}{'benign mean':>14}{'infil mean':>14}"
      f"{'benign std':>14}{'infil std':>14}")
for c in cols:
    print(f"{c:<18}{b[c].mean():>14.1f}{a[c].mean():>14.1f}"
          f"{b[c].std():>14.1f}{a[c].std():>14.1f}")
