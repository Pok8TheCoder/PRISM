"""Human-readable feature names for ARY (242-d) and Shaun (292-d) state vectors."""

from __future__ import annotations

from src.aryan.feature_schema242 import FEATURE_BLOCKS_242, FEATURE_COLS_242


def shaun_feature_names_292() -> list[str]:
    """292-d Shaun state layout (matches PRISM-shaun state_builder)."""
    # Canonical CIC feature names — abbreviated for graph/macro tail.
    flow_base = [f"flow_{i}" for i in range(64)]
    names: list[str] = []
    for f in flow_base:
        names.append(f"{f}_mean")
    for f in flow_base:
        names.append(f"{f}_std")
    for f in flow_base:
        names.append(f"{f}_max")
    for f in flow_base:
        names.append(f"{f}_min")
    median_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    for i in median_idx:
        names.append(f"flow_{i}_median")
    names.extend([
        "log_flow_volume",
        "tcp_fraction",
        "udp_fraction",
        "icmp_fraction",
        "other_proto_fraction",
        "dst_port_entropy",
        "syn_ack_ratio",
        "down_up_ratio_mean",
        "burst_factor",
        "log_syn_volume",
        "max_src_out_degree_log",
        "max_dst_in_degree_log",
        "src_ip_entropy",
        "dst_ip_entropy",
        "graph_density_ratio",
        "one_way_edge_ratio",
    ])
    if len(names) != 292:
        names = [f"dim_{i}" for i in range(292)]
    return names


def ary_block_indices() -> dict[str, list[int]]:
    name_to_idx = {n: i for i, n in enumerate(FEATURE_COLS_242)}
    return {block: [name_to_idx[c] for c in cols] for block, cols in FEATURE_BLOCKS_242.items()}


def shaun_block_indices() -> dict[str, list[int]]:
    return {
        "flow_means": list(range(0, 64)),
        "flow_stds": list(range(64, 128)),
        "flow_maxs": list(range(128, 192)),
        "flow_mins": list(range(192, 256)),
        "flow_medians": list(range(256, 276)),
        "macro_graph": list(range(276, 292)),
    }
