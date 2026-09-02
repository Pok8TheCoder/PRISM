"""
Downloads external attack datasets (UNSW-NB15 & CTU-13 samples)
and places them in data/raw/
"""

import os
import sys
import ssl
import urllib.request

# Append root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logger import setup_logger

logger = setup_logger("Downloader")

DATA_DIR = os.path.join("data", "raw")
UNSW_DIR = os.path.join(DATA_DIR, "unsw_nb15")
CTU_DIR = os.path.join(DATA_DIR, "ctu13")

os.makedirs(UNSW_DIR, exist_ok=True)
os.makedirs(CTU_DIR, exist_ok=True)

UNSW_URLS = [
    "https://raw.githubusercontent.com/tariq-nas/Intrusion-Detection-System-using-UNSW-NB15-Dataset/master/UNSW_NB15_testing-set.csv",
    "https://raw.githubusercontent.com/ML-Projects/UNSW-Network_Packet_Classification/master/UNSW_NB15_testing-set.csv",
    "https://media.githubusercontent.com/media/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/UNSW_NB15_training-set.csv"
]


def download_file(url: str, dest_path: str):
    ctx = ssl._create_unverified_context()
    logger.info(f"Downloading from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as response, open(dest_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)
    logger.info(f"Successfully downloaded {os.path.basename(dest_path)} ({len(data) / (1024*1024):.2f} MB)")


def main():
    dest_unsw = os.path.join(UNSW_DIR, "UNSW_NB15_sample.csv")
    if not os.path.exists(dest_unsw) or os.path.getsize(dest_unsw) < 1000:
        downloaded = False
        for url in UNSW_URLS:
            try:
                download_file(url, dest_unsw)
                downloaded = True
                break
            except Exception as e:
                logger.warning(f"Failed {url}: {e}")
        if not downloaded:
            logger.error("Could not download UNSW-NB15 from mirrors.")
    else:
        logger.info(f"UNSW-NB15 already present at {dest_unsw}")


if __name__ == "__main__":
    main()
