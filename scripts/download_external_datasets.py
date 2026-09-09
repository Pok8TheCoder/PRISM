"""
Downloads external attack datasets (UNSW-NB15 & CTU-13 samples)
and places them in data/raw/
"""

import os
import sys
import ssl
import urllib.request
import time

# Append root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logger import setup_logger

logger = setup_logger("Downloader")

DATA_DIR = os.path.join("data", "raw")
UNSW_DIR = os.path.join(DATA_DIR, "unsw_nb15")
CTU_DIR = os.path.join(DATA_DIR, "ctu13")

os.makedirs(UNSW_DIR, exist_ok=True)
os.makedirs(CTU_DIR, exist_ok=True)

UNSW_TRAIN_URLS = [
    "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
    "https://media.githubusercontent.com/media/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/UNSW_NB15_training-set.csv"
]

CTU_URLS = [
    "https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/detailed-bidirectional-flow-labels/capture20110810.binetflow"
]


def download_file(url: str, dest_path: str, chunk_size: int = 1024 * 1024):
    ctx = ssl._create_unverified_context()
    logger.info(f"Downloading from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    start_time = time.time()
    downloaded_bytes = 0
    temp_path = dest_path + ".tmp"
    
    with urllib.request.urlopen(req, context=ctx, timeout=60) as response, open(temp_path, "wb") as out_file:
        total_size = int(response.headers.get("content-length", 0))
        logger.info(f"Content length: {total_size / (1024*1024):.2f} MB")
        
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded_bytes += len(chunk)
            elapsed = time.time() - start_time
            if total_size > 0 and downloaded_bytes % (10 * 1024 * 1024) < chunk_size:
                pct = (downloaded_bytes / total_size) * 100
                speed = (downloaded_bytes / (1024 * 1024)) / max(elapsed, 1e-3)
                logger.info(f"Progress: {downloaded_bytes / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) - {speed:.2f} MB/s")

    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
        except OSError:
            pass
    os.replace(temp_path, dest_path)
    elapsed = time.time() - start_time
    logger.info(f"Successfully downloaded {os.path.basename(dest_path)} ({downloaded_bytes / (1024*1024):.2f} MB in {elapsed:.1f}s)")


def main():
    # 1. Download UNSW-NB15 Training Set (rich in Exfiltration, Backdoors, Exploits)
    dest_unsw = os.path.join(UNSW_DIR, "UNSW_NB15_training-set.csv")
    if not os.path.exists(dest_unsw) or os.path.getsize(dest_unsw) < 1_000_000:
        downloaded = False
        for url in UNSW_TRAIN_URLS:
            try:
                download_file(url, dest_unsw)
                downloaded = True
                break
            except Exception as e:
                logger.warning(f"Failed {url}: {e}")
        if not downloaded:
            logger.error("Could not download UNSW_NB15_training-set.csv from mirrors.")
    else:
        logger.info(f"UNSW_NB15_training-set.csv already present ({os.path.getsize(dest_unsw) / (1024*1024):.2f} MB)")

    # 2. Download CTU-13 Scenario 1 Botnet 42 (Pure Command & Control)
    dest_ctu = os.path.join(CTU_DIR, "capture20110810.binetflow")
    if not os.path.exists(dest_ctu) or os.path.getsize(dest_ctu) < 1_000_000:
        downloaded = False
        for url in CTU_URLS:
            try:
                download_file(url, dest_ctu)
                downloaded = True
                break
            except Exception as e:
                logger.warning(f"Failed {url}: {e}")
        if not downloaded:
            logger.error("Could not download CTU-13 dataset.")
    else:
        logger.info(f"CTU-13 capture20110810.binetflow already present ({os.path.getsize(dest_ctu) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()

