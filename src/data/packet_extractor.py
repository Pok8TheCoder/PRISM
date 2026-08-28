"""
PRISM -- Packet-Level Feature Extraction

Parses PCAP files using Scapy and extracts packet-level features aggregated
per flow (5-tuple) per time window.  The output DataFrame is designed to be
merged with the flow-level features produced by :mod:`src.data.flow_extractor`.

Usage:
    extractor = PacketExtractor(window_size_seconds=30)
    df = extractor.extract_from_pcap("data/raw/capture.pcap")
"""

from __future__ import annotations

import logging
import math
import os
import pathlib
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.constants import PACKET_FEATURES, TCP_FLAG_BITS, WINDOW_SIZE_SECONDS

logger = logging.getLogger("prism.data.packet_extractor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_sequential(ports: list, min_length: int = 5) -> bool:
    """Return ``True`` if *ports* contain a consecutive integer run of at
    least *min_length* elements.

    Parameters
    ----------
    ports : list
        Collection of port numbers (need not be sorted or unique).
    min_length : int
        Minimum run length to qualify as a sequential port scan.
    """
    if len(ports) < min_length:
        return False

    unique_sorted = sorted(set(ports))
    if len(unique_sorted) < min_length:
        return False

    run = 1
    for i in range(1, len(unique_sorted)):
        if unique_sorted[i] == unique_sorted[i - 1] + 1:
            run += 1
            if run >= min_length:
                return True
        else:
            run = 1
    return False


# ---------------------------------------------------------------------------
# PacketExtractor
# ---------------------------------------------------------------------------


class PacketExtractor:
    """Extract packet-level features from PCAP files.

    Features are aggregated per flow (5-tuple) per time window and returned
    as a :class:`pandas.DataFrame` whose columns match
    :data:`src.utils.constants.PACKET_FEATURES`.
    """

    def __init__(self, window_size_seconds: int = WINDOW_SIZE_SECONDS) -> None:
        """
        Parameters
        ----------
        window_size_seconds : int
            Duration of each aggregation window in seconds.
        """
        self.window_size: int = window_size_seconds
        logger.info(
            "PacketExtractor initialised  ·  window_size=%ds",
            self.window_size,
        )

    # ------------------------------------------------------------------
    # Packet-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_flow_key(pkt: Any) -> Optional[Tuple[str, str, int, int, int]]:
        """Extract the 5-tuple from a Scapy packet.

        Returns ``(src_ip, dst_ip, src_port, dst_port, protocol)`` or
        ``None`` if the packet lacks an IP layer.
        """
        # Lazy import to avoid import-time dependency on scapy
        from scapy.layers.inet import IP, TCP, UDP

        if not pkt.haslayer(IP):
            return None

        ip_layer = pkt[IP]
        src_ip: str = ip_layer.src
        dst_ip: str = ip_layer.dst
        protocol: int = int(ip_layer.proto)

        src_port: int = 0
        dst_port: int = 0

        try:
            if pkt.haslayer(TCP):
                src_port = int(pkt[TCP].sport)
                dst_port = int(pkt[TCP].dport)
            elif pkt.haslayer(UDP):
                src_port = int(pkt[UDP].sport)
                dst_port = int(pkt[UDP].dport)
        except Exception:
            pass

        return (src_ip, dst_ip, src_port, dst_port, protocol)

    @staticmethod
    def compute_port_entropy(ports: list) -> float:
        """Compute Shannon entropy (base 2) of the port distribution.

        Parameters
        ----------
        ports : list
            List of port numbers (may contain duplicates).

        Returns
        -------
        float
            Entropy in bits.  Returns 0.0 for empty input.
        """
        if not ports:
            return 0.0

        total = len(ports)
        counts = Counter(ports)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_from_pcap(self, pcap_path: str) -> pd.DataFrame:
        """Read a PCAP file and return aggregated packet features.

        Packets are iterated via :class:`scapy.utils.PcapReader` so that
        arbitrarily large captures can be processed without loading the
        entire file into memory.

        Parameters
        ----------
        pcap_path : str
            Path to a ``.pcap`` / ``.pcapng`` file.

        Returns
        -------
        pd.DataFrame
            One row per flow per time window.  Columns include the flow
            identifiers (``src_ip``, ``dst_ip``, ``src_port``,
            ``dst_port``, ``protocol``), window metadata (``window_id``,
            ``window_start``), and the 12 packet-level feature columns
            from :data:`PACKET_FEATURES`.
        """
        from scapy.layers.inet import IP, TCP, UDP
        from scapy.packet import Raw
        from scapy.utils import PcapReader

        logger.info("Reading PCAP: %s", pcap_path)

        if not os.path.isfile(pcap_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

        # Per-flow per-window raw packet records
        # Key: (flow_key, window_id)
        # Value: dict of lists
        flow_windows: Dict[
            Tuple[Tuple[str, str, int, int, int], int], Dict[str, list]
        ] = defaultdict(
            lambda: {
                "timestamps": [],
                "ttl": [],
                "tcp_window": [],
                "ip_frag": [],
                "payload_size": [],
                "syn": [],
                "ack": [],
                "fin": [],
                "rst": [],
                "psh": [],
                "urg": [],
                "seq": [],
                "dst_port": [],
            }
        )

        # Determine the capture start time from the first packet so that
        # window IDs are relative to the capture beginning.
        capture_start: Optional[float] = None
        packet_count = 0

        try:
            with PcapReader(pcap_path) as reader:
                for pkt in reader:
                    try:
                        flow_key = self.get_flow_key(pkt)
                        if flow_key is None:
                            continue

                        ts = float(pkt.time)

                        if capture_start is None:
                            capture_start = ts

                        window_id = int((ts - capture_start) // self.window_size)
                        bucket_key = (flow_key, window_id)
                        rec = flow_windows[bucket_key]

                        rec["timestamps"].append(ts)

                        # --- IP-layer features ---
                        ip_layer = pkt[IP]
                        rec["ttl"].append(int(ip_layer.ttl))

                        # Fragmentation: MF flag set or fragment offset > 0
                        try:
                            is_frag = bool(ip_layer.flags.MF) or int(ip_layer.frag) > 0
                        except Exception:
                            is_frag = False
                        rec["ip_frag"].append(is_frag)

                        # --- TCP-layer features ---
                        if pkt.haslayer(TCP):
                            tcp_layer = pkt[TCP]
                            rec["tcp_window"].append(int(tcp_layer.window))

                            flags_int = int(tcp_layer.flags)
                            rec["syn"].append(
                                bool(flags_int & TCP_FLAG_BITS["SYN"])
                            )
                            rec["ack"].append(
                                bool(flags_int & TCP_FLAG_BITS["ACK"])
                            )
                            rec["fin"].append(
                                bool(flags_int & TCP_FLAG_BITS["FIN"])
                            )
                            rec["rst"].append(
                                bool(flags_int & TCP_FLAG_BITS["RST"])
                            )
                            rec["psh"].append(
                                bool(flags_int & TCP_FLAG_BITS["PSH"])
                            )
                            rec["urg"].append(
                                bool(flags_int & TCP_FLAG_BITS["URG"])
                            )

                            rec["seq"].append(int(tcp_layer.seq))
                        else:
                            rec["tcp_window"].append(0)
                            rec["syn"].append(False)
                            rec["ack"].append(False)
                            rec["fin"].append(False)
                            rec["rst"].append(False)
                            rec["psh"].append(False)
                            rec["urg"].append(False)
                            rec["seq"].append(-1)  # sentinel for non-TCP

                        # --- Payload size ---
                        try:
                            rec["payload_size"].append(
                                len(pkt[Raw].load) if pkt.haslayer(Raw) else 0
                            )
                        except Exception:
                            rec["payload_size"].append(0)

                        # Destination port (for port-scan detection)
                        rec["dst_port"].append(flow_key[3])

                        packet_count += 1

                    except Exception:
                        logger.debug(
                            "Skipping malformed packet #%d", packet_count,
                            exc_info=True,
                        )
                        continue

        except Exception:
            logger.exception("Error reading PCAP file: %s", pcap_path)
            raise

        logger.info(
            "Parsed %d IP packets into %d flow-window buckets from %s",
            packet_count,
            len(flow_windows),
            pcap_path,
        )

        if not flow_windows:
            logger.warning("No IP packets found in %s — returning empty DataFrame.", pcap_path)
            return self._empty_dataframe()

        # ------------------------------------------------------------------
        # Aggregate per flow per window
        # ------------------------------------------------------------------
        rows: List[Dict[str, Any]] = []

        for (flow_key, window_id), rec in flow_windows.items():
            src_ip, dst_ip, src_port, dst_port, protocol = flow_key

            timestamps = rec["timestamps"]
            window_start = (
                capture_start + window_id * self.window_size
                if capture_start is not None
                else 0.0
            )

            ttl_arr = np.array(rec["ttl"], dtype=np.float64)
            tcp_win_arr = np.array(rec["tcp_window"], dtype=np.float64)
            payload_arr = np.array(rec["payload_size"], dtype=np.float64)

            # TTL statistics
            ttl_mean = float(np.mean(ttl_arr)) if len(ttl_arr) > 0 else 0.0
            ttl_std = float(np.std(ttl_arr, ddof=0)) if len(ttl_arr) > 0 else 0.0

            # TCP window statistics
            tcp_window_mean = float(np.mean(tcp_win_arr)) if len(tcp_win_arr) > 0 else 0.0
            tcp_window_std = float(np.std(tcp_win_arr, ddof=0)) if len(tcp_win_arr) > 0 else 0.0

            # IP fragmentation flag (any in window)
            ip_frag_flag = int(any(rec["ip_frag"]))

            # Payload size statistics
            payload_size_mean = float(np.mean(payload_arr)) if len(payload_arr) > 0 else 0.0
            payload_size_std = float(np.std(payload_arr, ddof=0)) if len(payload_arr) > 0 else 0.0

            # Port scan — sequential
            port_scan_sequential = int(
                _is_sequential(rec["dst_port"], min_length=6)  # > 5 ⟹ length ≥ 6
            )

            # Port scan — random (high unique count AND high entropy)
            unique_dst_ports = list(set(rec["dst_port"]))
            port_entropy = self.compute_port_entropy(rec["dst_port"])
            port_scan_random = int(
                len(unique_dst_ports) > 20 and port_entropy > 3.0
            )

            # Retransmission count (duplicate TCP sequence numbers)
            tcp_seqs = [s for s in rec["seq"] if s >= 0]  # exclude non-TCP sentinels
            if tcp_seqs:
                seq_counts = Counter(tcp_seqs)
                retransmission_count = sum(c - 1 for c in seq_counts.values() if c > 1)
            else:
                retransmission_count = 0

            # SYN / ACK ratio
            syn_count = sum(rec["syn"])
            ack_count = sum(rec["ack"])
            syn_ack_ratio = syn_count / (ack_count + 1)

            # RST rate
            rst_count = sum(rec["rst"])
            if len(timestamps) >= 2:
                flow_duration = max(timestamps) - min(timestamps)
            else:
                flow_duration = 0.0
            rst_rate = rst_count / (flow_duration + 1)

            rows.append(
                {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": protocol,
                    "window_id": window_id,
                    "window_start": window_start,
                    # --- PACKET_FEATURES (12 columns) ---
                    "ttl_mean": ttl_mean,
                    "ttl_std": ttl_std,
                    "tcp_window_mean": tcp_window_mean,
                    "tcp_window_std": tcp_window_std,
                    "ip_frag_flag": ip_frag_flag,
                    "payload_size_mean": payload_size_mean,
                    "payload_size_std": payload_size_std,
                    "port_scan_sequential": port_scan_sequential,
                    "port_scan_random": port_scan_random,
                    "retransmission_count": retransmission_count,
                    "syn_ack_ratio": syn_ack_ratio,
                    "rst_rate": rst_rate,
                }
            )

        df = pd.DataFrame(rows)
        df.sort_values(
            by=["window_id", "src_ip", "dst_ip", "src_port", "dst_port"],
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)

        logger.info(
            "Aggregation complete  ·  %d flow-window records  ·  shape %s",
            len(df),
            df.shape,
        )
        return df

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_batch(self, pcap_dir: str) -> pd.DataFrame:
        """Process all PCAP files in *pcap_dir* and concatenate results.

        Parameters
        ----------
        pcap_dir : str
            Directory to search for ``.pcap`` and ``.pcapng`` files
            (non-recursive).

        Returns
        -------
        pd.DataFrame
        """
        pcap_dir_path = pathlib.Path(pcap_dir)
        if not pcap_dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {pcap_dir}")

        pcap_files = sorted(
            str(p)
            for p in pcap_dir_path.iterdir()
            if p.suffix.lower() in (".pcap", ".pcapng")
        )

        if not pcap_files:
            raise FileNotFoundError(
                f"No .pcap / .pcapng files found in '{pcap_dir}'."
            )

        logger.info(
            "Batch extraction: found %d PCAP file(s) in %s",
            len(pcap_files),
            pcap_dir,
        )

        frames: List[pd.DataFrame] = []
        for fpath in pcap_files:
            try:
                frames.append(self.extract_from_pcap(fpath))
            except Exception:
                logger.exception("Failed to process %s — skipping.", fpath)

        if not frames:
            raise RuntimeError(
                "All PCAP files failed to process.  Check data integrity."
            )

        df = pd.concat(frames, ignore_index=True)
        logger.info(
            "Batch extraction complete  ·  %d total records  ·  shape %s",
            len(df),
            df.shape,
        )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        """Return an empty DataFrame with the expected schema."""
        columns = [
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "protocol",
            "window_id",
            "window_start",
        ] + list(PACKET_FEATURES)
        return pd.DataFrame(columns=columns)


# ======================================================================
# Standalone entry point
# ======================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="PRISM Packet-Level Feature Extractor",
    )
    parser.add_argument(
        "--pcap",
        type=str,
        default=None,
        help="Path to a single PCAP file.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing PCAP files.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=WINDOW_SIZE_SECONDS,
        help=f"Window size in seconds (default: {WINDOW_SIZE_SECONDS}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the output DataFrame as parquet.",
    )
    args = parser.parse_args()

    if args.pcap is None and args.dir is None:
        parser.error("Provide at least one of --pcap or --dir.")

    extractor = PacketExtractor(window_size_seconds=args.window)

    if args.pcap is not None:
        result = extractor.extract_from_pcap(args.pcap)
    else:
        result = extractor.extract_batch(args.dir)

    print("\n--- Packet Feature DataFrame ---")
    print(f"Shape : {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print(result.head(20).to_string())

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result.to_parquet(args.output, index=False)
        logger.info("Output written to %s", args.output)
        print(f"\nSaved to {args.output}")
