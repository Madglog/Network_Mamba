"""Real-dataset loaders: CICIoT2023 (primary), TON_IoT, IoT-23.

This module is intentionally a thin, documented skeleton. The flow-pattern
representation needs *per-packet sequences*, which means the data pipeline has
two possible entry points:

  1. From pcap (preferred for flow patterns):
        pcap --> per-packet feature extraction --> first-K-packet sequences
     Use a tool such as NFStream (`pip install nfstream`) or a Scapy reader to
     walk packets per 5-tuple flow and emit the schema in
     :mod:`flowmamba.data.features`. CICIoT2023 and IoT-23 both ship pcaps.

  2. From the published CSV flow records (aggregated statistics only):
        CSV --> ~80-stat vectors --> classical baselines
     These CSVs are *already aggregated* and therefore cannot reconstruct packet
     order. They are still useful for the XGBoost / Isolation-Forest baselines
     and for the CNN-BiLSTM aggregated-stats control, but NOT for the Mamba arm.

The category mapping below groups CICIoT2023's 33 attack types into the 8-class
scheme (benign + 7 categories) used by the classifier head.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from flowmamba.data.synthetic import CLASS_NAMES

# CICIoT2023 publishes 33 fine-grained labels; the proposal classifies the 7
# coarse categories (+ benign). Extend this map as the fine labels are parsed.
CICIOT_CATEGORY_MAP: Dict[str, str] = {
    "BenignTraffic": "Benign",
    # DDoS family
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    # DoS family
    "DoS-UDP_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    # Recon
    "Recon-PingSweep": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon",
    "Recon-HostDiscovery": "Recon",
    "VulnerabilityScan": "Recon",
    # Web-based
    "SqlInjection": "WebBased",
    "CommandInjection": "WebBased",
    "XSS": "WebBased",
    "Backdoor_Malware": "WebBased",
    "Uploading_Attack": "WebBased",
    "BrowserHijacking": "WebBased",
    # Brute force
    "DictionaryBruteForce": "BruteForce",
    # Spoofing
    "MITM-ArpSpoofing": "Spoofing",
    "DNS_Spoofing": "Spoofing",
    # Mirai
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
}


def category_to_index(category: str) -> int:
    """Map a coarse category name to its classifier index (benign == 0)."""
    return CLASS_NAMES.index(category)


def label_to_index(fine_label: str) -> int:
    """Map a CICIoT2023 fine label straight to a classifier index."""
    category = CICIOT_CATEGORY_MAP.get(fine_label)
    if category is None:
        raise KeyError(f"Unknown CICIoT2023 label: {fine_label!r}")
    return category_to_index(category)


def flows_from_pcap(
    pcap_path: str,
    max_packets: int = 32,
    bidirectional: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Extract first-K-packet flow patterns from a pcap file.

    Returns ``(flows, flow_keys)`` where ``flows`` has shape ``(N, max_packets,
    F)`` matching :mod:`flowmamba.data.features`, and ``flow_keys`` are the 5-tuple
    identifiers (for joining against label files).

    Implementation note
    -------------------
    This is the one piece that must be wired to a packet reader for real
    experiments. NFStream exposes per-packet hooks via a custom plugin, or Scapy
    can be used directly. Pseudocode:

        for flow in group_packets_by_5tuple(pcap_path):
            seq = []
            for pkt in flow.packets[:max_packets]:
                seq.append([
                    pkt.size, direction(pkt), pkt.iat, pkt.ttl, pkt.tcp_window,
                    pkt.syn, pkt.ack, pkt.fin, pkt.rst, pkt.psh, pkt.urg,
                    int(pkt.is_tcp), pkt.l4_header_len,
                ])
            flows.append(pad(seq, max_packets))
    """
    raise NotImplementedError(
        "Wire flows_from_pcap to NFStream or Scapy for real CICIoT2023/IoT-23 "
        "experiments. The synthetic generator in flowmamba.data.synthetic "
        "provides a runnable stand-in until then."
    )


def load_aggregated_csv(csv_path: str, label_column: str = "label"):
    """Load the published aggregated-statistic CSVs for the classical baselines.

    Returns ``(X, y, feature_names)``. Used by the XGBoost / Isolation-Forest
    baselines and the aggregated-stats control -- NOT by the Mamba arm, which
    needs packet order that aggregated CSVs cannot supply.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise KeyError(f"label column {label_column!r} not in {csv_path}")
    raw_labels = df[label_column].astype(str)
    y = np.array(
        [category_to_index(CICIOT_CATEGORY_MAP.get(lbl, lbl)) for lbl in raw_labels],
        dtype=np.int64,
    )
    feature_cols = [c for c in df.columns if c != label_column]
    x = df[feature_cols].to_numpy(dtype=np.float64)
    return x, y, feature_cols
