"""
PRISM Dataset Registry and Factory.
Central registry mapping dataset names to metadata, download URLs, and extractor instances.
Supports CIC-IDS-2018, CTU-13, UNSW-NB15, CICIoT2023, LANL, and DARPA datasets.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
import pandas as pd

from src.utils.constants import (
    CICIDS_LABEL_TO_MITRE,
    CTU13_LABEL_TO_MITRE,
    UNSWNB15_LABEL_TO_MITRE,
    CICIOT2023_LABEL_TO_MITRE,
    LANL_LABEL_TO_MITRE,
    DARPA_LABEL_TO_MITRE,
    SUPPORTED_DATASETS,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    name: str
    display_name: str
    description: str
    format: str  # flow_csv, auth_csv, pcap, tcpdump
    source_url: str
    citation: str
    label_mapping: Dict[str, str]
    num_raw_features: int
    default_filename: str


DATASET_CATALOG: Dict[str, DatasetMetadata] = {
    "cicids2018": DatasetMetadata(
        name="cicids2018",
        display_name="CIC-IDS-2018",
        description="CSE-CIC-IDS2018 Intrusion Detection Dataset from Canadian Institute for Cybersecurity.",
        format="flow_csv",
        source_url="https://www.unb.ca/cic/datasets/ids-2018.html",
        citation="Sharafaldin et al. (2018), Toward Generating a New Dataset for Cybersecurity Analysis.",
        label_mapping=CICIDS_LABEL_TO_MITRE,
        num_raw_features=80,
        default_filename="cicids2018_sample.csv",
    ),
    "ctu13": DatasetMetadata(
        name="ctu13",
        display_name="CTU-13",
        description="CTU-13 Botnet Traffic Dataset from Stratosphere IPS.",
        format="flow_csv",
        source_url="https://www.stratosphereips.org/datasets-ctu13",
        citation="Garcia et al. (2014), An Empirical Comparison of Botnet Detection Methods.",
        label_mapping=CTU13_LABEL_TO_MITRE,
        num_raw_features=14,
        default_filename="ctu13_sample.csv",
    ),
    "unsw-nb15": DatasetMetadata(
        name="unsw-nb15",
        display_name="UNSW-NB15",
        description="UNSW-NB15 Network Intrusion Dataset from Cyber Range Lab of UNSW Canberra.",
        format="flow_csv",
        source_url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        citation="Moustafa & Slay (2015), UNSW-NB15: a comprehensive data set for network intrusion detection systems.",
        label_mapping=UNSWNB15_LABEL_TO_MITRE,
        num_raw_features=49,
        default_filename="unsw_nb15_sample.csv",
    ),
    "ciciot2023": DatasetMetadata(
        name="ciciot2023",
        display_name="CICIoT2023",
        description="CICIoT2023 Real-time Internet of Things (IoT) Security Dataset.",
        format="flow_csv",
        source_url="https://www.unb.ca/cic/datasets/iot-2023.html",
        citation="Neto et al. (2023), CICIoT2023: A Real-Time Dataset for Profiling Attacks in IoT Devices.",
        label_mapping=CICIOT2023_LABEL_TO_MITRE,
        num_raw_features=46,
        default_filename="ciciot2023_sample.csv",
    ),
    "lanl": DatasetMetadata(
        name="lanl",
        display_name="LANL Unified Host and Network Authentication Dataset",
        description="Los Alamos National Laboratory Comprehensive Multi-Source Cyber Event Dataset.",
        format="auth_csv",
        source_url="https://csr.lanl.gov/repository/data/",
        citation="Kent (2015), Cybersecurity Data Sources for Dynamic Network Research.",
        label_mapping=LANL_LABEL_TO_MITRE,
        num_raw_features=9,
        default_filename="lanl_auth_sample.csv",
    ),
    "darpa": DatasetMetadata(
        name="darpa",
        display_name="DARPA Intrusion Detection Dataset",
        description="DARPA 1998/1999/2000 Intrusion Detection Evaluation Datasets.",
        format="tcpdump",
        source_url="https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-data-set",
        citation="Lippmann et al. (2000), Evaluating Intrusion Detection Systems: The 1998 DARPA Off-line Intrusion Detection Evaluation.",
        label_mapping=DARPA_LABEL_TO_MITRE,
        num_raw_features=41,
        default_filename="darpa_sample.csv",
    ),
}


def get_dataset_metadata(dataset_name: str) -> DatasetMetadata:
    """Retrieve metadata for a registered dataset."""
    key = dataset_name.lower().replace("_", "-")
    if key not in DATASET_CATALOG:
        raise ValueError(
            f"Dataset '{dataset_name}' not registered. Supported: {list(DATASET_CATALOG.keys())}"
        )
    return DATASET_CATALOG[key]


def list_supported_datasets() -> List[Dict[str, Any]]:
    """Return a summary list of all supported datasets."""
    return [
        {
            "id": meta.name,
            "display_name": meta.display_name,
            "format": meta.format,
            "num_raw_features": meta.num_raw_features,
            "source_url": meta.source_url,
            "description": meta.description,
        }
        for meta in DATASET_CATALOG.values()
    ]


def detect_dataset_type(file_path: str) -> str:
    """
    Auto-detect dataset type from CSV header columns.
    Returns dataset key string.
    """
    try:
        sample_df = pd.read_csv(file_path, nrows=5)
        cols_lower = [str(c).strip().lower() for c in sample_df.columns]

        if any(c in cols_lower for c in ["dur", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "attack_cat"]):
            return "unsw-nb15"
        elif any(c in cols_lower for c in ["flow_duration", "tot fwd pkts", "dst port", "bwd packet length max"]):
            return "cicids2018"
        elif any(c in cols_lower for c in ["flow duration", "header_length", "syn_count", "urg_count"]):
            return "ciciot2023"
        elif any(c in cols_lower for c in ["time", "source_user", "destination_user", "auth_type", "logon_type"]):
            return "lanl"
        elif any(c in cols_lower for c in ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"]):
            return "darpa"
        elif "botnet" in cols_lower or "background" in cols_lower:
            return "ctu13"
        else:
            logger.warning(f"Could not auto-detect dataset type for {file_path}. Defaulting to cicids2018.")
            return "cicids2018"
    except Exception as e:
        logger.error(f"Error reading header from {file_path}: {e}")
        return "cicids2018"


def get_dataset_extractor(dataset_name: str, raw_dir: str = "data/raw") -> Any:
    """
    Factory method to instantiate the appropriate extractor for a dataset.
    """
    key = dataset_name.lower().replace("_", "-")
    if key == "cicids2018":
        from src.data.flow_extractor import FlowExtractor
        return FlowExtractor(dataset_type=key, raw_dir=raw_dir)
    elif key == "ctu13":
        from src.data.ctu13_extractor import CTU13Extractor
        return CTU13Extractor(raw_dir=raw_dir)
    elif key == "unsw-nb15":
        from src.data.unswnb15_extractor import UNSWNB15Extractor
        return UNSWNB15Extractor(raw_dir=raw_dir)
    elif key == "ciciot2023":
        from src.data.ciciot2023_extractor import CICIoT2023Extractor
        return CICIoT2023Extractor(raw_dir=raw_dir)
    elif key == "lanl":
        from src.data.lanl_extractor import LANLExtractor
        return LANLExtractor(raw_dir=raw_dir)
    elif key == "darpa":
        from src.data.darpa_extractor import DARPAExtractor
        return DARPAExtractor(raw_dir=raw_dir)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_name}. Supported: {list(DATASET_CATALOG.keys())}")
