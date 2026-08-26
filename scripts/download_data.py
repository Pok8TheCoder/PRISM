"""Download verified CIC-IDS2018 daily flow subsets for PoC."""
import os
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

data_links = {
    "thursday_01_03_2018": "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed+Traffic+Data+for+ML+Algorithms/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "wednesday_28_02_2018": "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed+Traffic+Data+for+ML+Algorithms/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "thursday_15_02_2018": "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed+Traffic+Data+for+ML+Algorithms/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
}

def download_file(dataset_name, url, dest_path):
    print(f"Downloading {dataset_name}")
    response = urllib.request.urlopen(url)
    with open(dest_path, "wb") as f:
        f.write(response.read())
    print(f"Saved to {dest_path.name}")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in data_links.items():
        dest = DATA_DIR / f"{name}.csv"
        if dest.exists():
            print(f"Dataset {name} already exists at {dest}.")
        else:
            try:
                download_file(name, url, dest)
            except Exception as e:
                print(f"Failed to download dataset {name}: {e}")

if __name__ == "__main__":
    main()