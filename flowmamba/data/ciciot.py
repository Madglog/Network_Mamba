"""Real-dataset loaders: CICIoT2023 (primary), TON_IoT, IoT-23.

The flow-pattern representation needs *per-packet sequences*, so the data
pipeline has two possible entry points:

  1. From pcap (the path the Mamba arm needs):
        pcap --> per-packet feature extraction --> first-K-packet sequences
     :func:`flows_from_pcap` walks packets, groups them by 5-tuple flow, and
     emits the payload-free schema in :mod:`flowmamba.data.features`. CICIoT2023
     and IoT-23 both ship pcaps.

  2. From aggregated flow records (statistics only, no packet order):
        Zeek conn.log / published CSV --> ~N-stat vectors --> classical baselines
     These are *already aggregated* and cannot reconstruct packet order, so they
     drive the XGBoost / Isolation-Forest baselines (and the aggregated-stats
     control) but NOT the Mamba arm. The IoT-23 "lighter version" ships only
     these ``conn.log.labeled`` files -- :func:`load_zeek_conn_aggregated` reads
     them; :func:`load_zeek_conn_labels` reads just the labels to join onto pcap
     flows.

The category maps group each dataset's fine-grained labels into the 8-class
scheme (benign + 7 categories) used by the classifier head. They are coarse and
deliberately editable -- adjust them to the taxonomy your experiment reports.
"""

from __future__ import annotations

import socket
from typing import Dict, List, Optional, Tuple

import numpy as np

from flowmamba.data.features import PACKET_FEATURES, n_features
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

# IoT-23 (Stratosphere) labels its flows with a coarse ``label`` (Benign /
# Malicious) and an informative ``detailed-label``. The 8-class scheme has no
# native botnet-family split, so the Mirai/Torii/Okiru C&C families all fold into
# the "Mirai" botnet bucket. This is a research-judgement mapping -- edit it to
# match the taxonomy you intend to report.
IOT23_CATEGORY_MAP: Dict[str, str] = {
    "Benign": "Benign",
    "-": "Benign",                              # Zeek's empty detailed-label on benign rows
    "PartOfAHorizontalPortScan": "Recon",
    "PartOfAHorizontalPortScan-Attack": "Recon",
    "DDoS": "DDoS",
    "Attack": "BruteForce",                     # mostly telnet/credential login attempts
    "C&C": "Mirai",
    "C&C-HeartBeat": "Mirai",
    "C&C-HeartBeat-Attack": "Mirai",
    "C&C-HeartBeat-FileDownload": "Mirai",
    "C&C-FileDownload": "Mirai",
    "C&C-Torii": "Mirai",
    "C&C-Mirai": "Mirai",
    "C&C-PartOfAHorizontalPortScan": "Mirai",
    "FileDownload": "Mirai",
    "Okiru": "Mirai",
    "Okiru-Attack": "Mirai",
    "Torii": "Mirai",
    "Mirai": "Mirai",
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


def iot23_label_to_index(detailed_label: str) -> Optional[int]:
    """Map an IoT-23 detailed-label to a classifier index, or ``None`` if unknown."""
    category = IOT23_CATEGORY_MAP.get(detailed_label.strip())
    if category is None:
        return None
    return category_to_index(category)


# --------------------------------------------------------------------------- #
# pcap -> first-K-packet flow patterns (the Mamba-arm entry point)
# --------------------------------------------------------------------------- #
# Indices into the per-packet feature vector (kept in sync with features.py).
_F = {name: i for i, name in enumerate(PACKET_FEATURES)}

# libpcap data-link types we know how to strip down to an IP datagram.
_DLT_NULL = 0
_DLT_EN10MB = 1
_DLT_RAW = 12
_DLT_RAW_ALT = 101
_DLT_LOOP = 108
_DLT_LINUX_SLL = 113


def _open_pcap(path: str):
    """Open a classic-pcap or pcapng file, returning ``(reader, datalink)``."""
    import dpkt

    handle = open(path, "rb")
    magic = handle.read(4)
    handle.seek(0)
    # pcapng section-header block starts with 0x0A0D0D0A.
    if magic == b"\x0a\x0d\x0d\x0a":
        reader = dpkt.pcapng.Reader(handle)
    else:
        reader = dpkt.pcap.Reader(handle)
    return reader, reader.datalink(), handle


def _ip_from_buf(buf: bytes, datalink: int):
    """Strip the link layer and return an IPv4/IPv6 object, or ``None``."""
    import dpkt

    ip_types = (dpkt.ip.IP, dpkt.ip6.IP6)
    try:
        if datalink == _DLT_EN10MB:
            data = dpkt.ethernet.Ethernet(buf).data
        elif datalink == _DLT_LINUX_SLL:
            data = dpkt.sll.SLL(buf).data
        elif datalink in (_DLT_NULL, _DLT_LOOP):
            data = dpkt.loopback.Loopback(buf).data
        elif datalink in (_DLT_RAW, _DLT_RAW_ALT):
            version = buf[0] >> 4
            data = dpkt.ip.IP(buf) if version == 4 else dpkt.ip6.IP6(buf)
        else:  # unknown link type -- best-effort guess at Ethernet framing
            data = dpkt.ethernet.Ethernet(buf).data
        return data if isinstance(data, ip_types) else None
    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, IndexError):
        return None


def _l3_l4(ip):
    """Return ``(src, sport, dst, dport, proto, ttl, size, l4)`` or ``None``.

    ``l4`` is the TCP/UDP object; non-TCP/UDP packets are skipped (the schema is
    transport-flow oriented).
    """
    import dpkt

    if isinstance(ip, dpkt.ip6.IP6):
        ttl = ip.hlim
        size = ip.plen + 40  # payload + fixed IPv6 header (extension headers ignored)
        src = socket.inet_ntop(socket.AF_INET6, ip.src)
        dst = socket.inet_ntop(socket.AF_INET6, ip.dst)
    else:
        ttl = ip.ttl
        size = ip.len
        src = socket.inet_ntoa(ip.src)
        dst = socket.inet_ntoa(ip.dst)

    l4 = ip.data
    if isinstance(l4, dpkt.tcp.TCP):
        return src, l4.sport, dst, l4.dport, "tcp", ttl, size, l4
    if isinstance(l4, dpkt.udp.UDP):
        return src, l4.sport, dst, l4.dport, "udp", ttl, size, l4
    return None


def _packet_row(l4, ttl: int, size: int, is_upstream: bool, iat: float) -> List[float]:
    """Build the 13-feature per-packet row from parsed header fields."""
    import dpkt

    row = [0.0] * n_features()
    row[_F["size"]] = float(size)
    row[_F["direction"]] = 1.0 if is_upstream else -1.0
    row[_F["iat"]] = float(iat)
    row[_F["ttl"]] = float(ttl)
    if isinstance(l4, dpkt.tcp.TCP):
        flags = l4.flags
        row[_F["win"]] = float(l4.win)
        row[_F["syn"]] = float(bool(flags & dpkt.tcp.TH_SYN))
        row[_F["ack"]] = float(bool(flags & dpkt.tcp.TH_ACK))
        row[_F["fin"]] = float(bool(flags & dpkt.tcp.TH_FIN))
        row[_F["rst"]] = float(bool(flags & dpkt.tcp.TH_RST))
        row[_F["psh"]] = float(bool(flags & dpkt.tcp.TH_PUSH))
        row[_F["urg"]] = float(bool(flags & dpkt.tcp.TH_URG))
        row[_F["is_tcp"]] = 1.0
        row[_F["header_len"]] = float(20 + len(l4.opts))
    else:  # UDP
        row[_F["is_tcp"]] = 0.0
        row[_F["header_len"]] = 8.0
    return row


class _Flow:
    __slots__ = ("orig", "resp", "proto", "rows", "last_ts", "n_seen")

    def __init__(self, orig, resp, proto):
        self.orig = orig          # (ip, port) of the endpoint that sent the first packet
        self.resp = resp          # (ip, port) of the other endpoint
        self.proto = proto
        self.rows: List[List[float]] = []
        self.last_ts: float = 0.0
        self.n_seen: int = 0      # total packets observed (may exceed max_packets)


def flows_from_pcap(
    pcap_path: str,
    max_packets: int = 32,
    max_flows: Optional[int] = None,
    max_total_packets: Optional[int] = None,
    min_packets: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str, int, str, int]]]:
    """Extract first-K-packet flow patterns from a pcap / pcapng file.

    Packets are grouped into bidirectional flows by their canonical 5-tuple. The
    endpoint that sent the *first* packet of a flow is the originator, so
    ``direction`` is ``+1`` for originator->responder packets and ``-1`` for the
    reverse -- matching Zeek's orig/resp convention, which lets the keys join
    straight onto ``conn.log`` labels.

    Parameters
    ----------
    pcap_path : path to a ``.pcap`` / ``.pcapng`` capture.
    max_packets : K -- packets kept per flow (the sequence length).
    max_flows : stop registering new flows after this many (bounds memory on
        large captures); ``None`` = unbounded.
    max_total_packets : hard cap on packets read from the file; ``None`` = read
        the whole capture.
    min_packets : drop flows shorter than this many packets.

    Returns
    -------
    flows   : float array ``(N, max_packets, F)`` -- raw, pre-standardisation.
    lengths : int array ``(N,)`` -- true packet count per flow (<= max_packets).
    keys    : list of ``(proto, orig_ip, orig_port, resp_ip, resp_port)`` tuples,
              aligned row-for-row with ``flows`` (for joining against labels).
    """
    reader, datalink, handle = _open_pcap(pcap_path)
    flows: Dict[tuple, _Flow] = {}
    n_read = 0
    try:
        for ts, buf in reader:
            if max_total_packets is not None and n_read >= max_total_packets:
                break
            n_read += 1

            ip = _ip_from_buf(buf, datalink)
            if ip is None:
                continue
            parsed = _l3_l4(ip)
            if parsed is None:
                continue
            src, sport, dst, dport, proto, ttl, size, l4 = parsed

            ep_src = (src, sport)
            ep_dst = (dst, dport)
            lo, hi = sorted((ep_src, ep_dst))
            key = (proto, lo, hi)

            flow = flows.get(key)
            if flow is None:
                if max_flows is not None and len(flows) >= max_flows:
                    continue  # capacity reached: ignore packets from unseen flows
                flow = _Flow(orig=ep_src, resp=ep_dst, proto=proto)
                flows[key] = flow

            flow.n_seen += 1
            if len(flow.rows) >= max_packets:
                continue  # flow already full; keep counting but stop recording

            iat = 0.0 if not flow.rows else float(ts) - flow.last_ts
            flow.last_ts = float(ts)
            is_upstream = ep_src == flow.orig
            flow.rows.append(_packet_row(l4, ttl, size, is_upstream, iat))
    finally:
        handle.close()

    f = n_features()
    seqs: List[np.ndarray] = []
    lengths: List[int] = []
    keys: List[Tuple[str, str, int, str, int]] = []
    for flow in flows.values():
        length = len(flow.rows)
        if length < min_packets:
            continue
        padded = np.zeros((max_packets, f), dtype=np.float64)
        padded[:length] = np.asarray(flow.rows, dtype=np.float64)
        seqs.append(padded)
        lengths.append(length)
        keys.append((flow.proto, flow.orig[0], flow.orig[1], flow.resp[0], flow.resp[1]))

    if not seqs:
        return np.zeros((0, max_packets, f), dtype=np.float64), np.zeros((0,), np.int64), []
    return np.stack(seqs), np.asarray(lengths, dtype=np.int64), keys


# --------------------------------------------------------------------------- #
# Zeek conn.log parsing (IoT-23 labels + aggregated baselines)
# --------------------------------------------------------------------------- #
def _read_zeek_tsv(path: str):
    """Yield ``dict`` rows from a Zeek TSV log, honouring ``#fields`` / ``#separator``.

    Tolerates the IoT-23 quirk where ``label`` / ``detailed-label`` are appended
    space-separated onto the ``tunnel_parents`` field instead of tab-separated.
    """
    sep = "\t"
    fields: Optional[List[str]] = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if line.startswith("#separator"):
                    token = line.split(" ", 1)[1].strip()
                    sep = token.encode().decode("unicode_escape")  # e.g. \x09 -> tab
                elif line.startswith("#fields"):
                    # Split on *any* whitespace, not just `sep`: real IoT-23 files
                    # join the trailing tunnel_parents/label/detailed-label columns
                    # with spaces even when the declared separator is tab. Zeek field
                    # names never contain internal whitespace, so this is safe.
                    fields = line.split()[1:]
                continue
            if fields is None or not line:
                continue
            # conn.log values never contain internal whitespace, so split on *any*
            # whitespace -- this is immune to the IoT-23 files that inconsistently
            # mix tabs and spaces between the trailing label columns. Fall back to
            # the declared separator only if the whitespace split disagrees on count.
            parts = line.split()
            if len(parts) != len(fields):
                alt = line.split(sep)
                if len(alt) == len(fields):
                    parts = alt
            if len(parts) != len(fields):
                continue
            yield dict(zip(fields, parts))


# Different IoT-23 files name the fine-label column differently
# (``detailed-label`` in some, ``det_label`` in others). It is always the final
# column of conn.log.labeled, so fall back to that if no known alias is present.
_DETAILED_ALIASES = ("detailed-label", "detailed_label", "det_label", "detailedlabel")


def _detailed_label(row: Dict[str, str]) -> str:
    for alias in _DETAILED_ALIASES:
        if alias in row:
            return row[alias]
    return list(row.values())[-1] if row else "-"


def load_zeek_conn_labels(
    conn_log_path: str,
) -> Dict[Tuple[str, str, int, str, int], str]:
    """Build a ``5-tuple -> detailed-label`` lookup from a ``conn.log.labeled``.

    Keyed by ``(proto, orig_h, orig_p, resp_h, resp_p)`` to match the keys that
    :func:`flows_from_pcap` emits. *Both* orientations are stored: a label is a
    per-connection property, but a capture that starts mid-flow can make the
    pcap's first-seen sender differ from Zeek's originator, flipping the tuple.
    Storing the reverse direction too makes the join direction-agnostic.
    (5-tuples can repeat across time with port reuse; the last occurrence wins --
    IoT-23 captures are near-homogeneous per file so this is a safe simplification.)
    """
    lut: Dict[Tuple[str, str, int, str, int], str] = {}
    for row in _read_zeek_tsv(conn_log_path):
        try:
            proto = row["proto"]
            oh, op = row["id.orig_h"], int(row["id.orig_p"])
            rh, rp = row["id.resp_h"], int(row["id.resp_p"])
        except (KeyError, ValueError):
            continue
        detailed = _detailed_label(row) or "-"
        lut[(proto, oh, op, rh, rp)] = detailed
        lut[(proto, rh, rp, oh, op)] = detailed  # reverse, for mid-flow capture starts
    return lut


def build_dataset_from_pcap(
    pcap_path: str,
    conn_log_path: Optional[str] = None,
    fixed_label: Optional[str] = None,
    max_packets: int = 32,
    category_map: Optional[Dict[str, str]] = None,
    drop_unlabeled: bool = True,
    **pcap_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn a pcap into the ``(flows, labels, lengths)`` training contract.

    Labels come from one of two sources:

      * ``conn_log_path`` -- a Zeek ``conn.log.labeled`` (IoT-23). Each flow's
        detailed-label is looked up by 5-tuple and mapped through
        ``category_map`` (default :data:`IOT23_CATEGORY_MAP`).
      * ``fixed_label`` -- a single coarse category for the whole file (the
        common CICIoT2023 case, where each pcap is one attack type).

    Flows whose label cannot be resolved are dropped when ``drop_unlabeled`` is
    set (otherwise they raise). Returns arrays matching
    :func:`flowmamba.data.synthetic.make_synthetic_flows`.
    """
    if (conn_log_path is None) == (fixed_label is None):
        raise ValueError("pass exactly one of conn_log_path or fixed_label")

    flows, lengths, keys = flows_from_pcap(pcap_path, max_packets=max_packets, **pcap_kwargs)
    if len(keys) == 0:
        return flows, np.zeros((0,), np.int64), lengths

    if fixed_label is not None:
        idx = category_to_index(fixed_label)  # validates the category name
        labels = np.full(len(keys), idx, dtype=np.int64)
        return flows, labels, lengths

    assert conn_log_path is not None  # guaranteed by the XOR check above
    cmap = category_map or IOT23_CATEGORY_MAP
    lut = load_zeek_conn_labels(conn_log_path)
    labels = np.empty(len(keys), dtype=np.int64)
    keep = np.ones(len(keys), dtype=bool)
    for i, key in enumerate(keys):
        detailed = lut.get(key)
        category = cmap.get(detailed.strip()) if detailed is not None else None
        if category is None:
            if not drop_unlabeled:
                raise KeyError(f"no label for flow {key} (detailed={detailed!r})")
            keep[i] = False
            labels[i] = 0
        else:
            labels[i] = category_to_index(category)
    return flows[keep], labels[keep], lengths[keep]


def _find_conn_log(pcap_path: str) -> Optional[str]:
    """Locate the Zeek ``conn.log.labeled`` that belongs to a pcap.

    IoT-23 scenarios keep the labelled log beside the pcap, or under a ``bro/`` /
    ``zeek/`` subdirectory. Searches the pcap's directory and one level down.
    """
    import glob
    import os

    here = os.path.dirname(os.path.abspath(pcap_path))
    patterns = [
        os.path.join(here, "*conn.log.labeled"),
        os.path.join(here, "bro", "*conn.log.labeled"),
        os.path.join(here, "zeek", "*conn.log.labeled"),
        os.path.join(here, "*", "*conn.log.labeled"),
    ]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


def build_dataset_from_iot23_dir(
    root: str,
    max_packets: int = 32,
    category_map: Optional[Dict[str, str]] = None,
    max_flows_per_file: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one ``(flows, labels, lengths)`` dataset from an IoT-23 directory tree.

    Walks ``root`` for ``*.pcap`` / ``*.pcapng`` files, pairs each with its
    ``conn.log.labeled`` (see :func:`_find_conn_log`), extracts and labels flows,
    and concatenates everything. Files without a discoverable label log are
    skipped with a warning.
    """
    import glob
    import os

    pcaps = sorted(
        glob.glob(os.path.join(root, "**", "*.pcap"), recursive=True)
        + glob.glob(os.path.join(root, "**", "*.pcapng"), recursive=True)
    )
    if not pcaps:
        raise FileNotFoundError(f"no .pcap/.pcapng files under {root!r}")

    all_flows, all_labels, all_lengths = [], [], []
    for pcap in pcaps:
        conn_log = _find_conn_log(pcap)
        if conn_log is None:
            if verbose:
                print(f"  [skip] no conn.log.labeled for {pcap}")
            continue
        flows, labels, lengths = build_dataset_from_pcap(
            pcap, conn_log_path=conn_log, max_packets=max_packets,
            category_map=category_map, max_flows=max_flows_per_file,
        )
        if verbose:
            print(f"  [ok]  {os.path.basename(pcap)}: {len(labels)} labelled flows")
        if len(labels):
            all_flows.append(flows)
            all_labels.append(labels)
            all_lengths.append(lengths)

    if not all_flows:
        raise RuntimeError(f"no labelled flows extracted from {root!r}")
    return (
        np.concatenate(all_flows, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_lengths, axis=0),
    )


# Numeric Zeek conn.log columns usable as aggregated features for the baselines.
_ZEEK_NUMERIC = [
    "duration", "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]
_ZEEK_PROTO_CODE = {"tcp": 0, "udp": 1, "icmp": 2}


def load_zeek_conn_aggregated(
    conn_log_path: str,
    category_map: Optional[Dict[str, str]] = None,
    drop_unlabeled: bool = True,
):
    """Load a Zeek ``conn.log.labeled`` as ``(X, y, feature_names)`` for baselines.

    This is the entry point for the IoT-23 *lighter version* (conn.log only, no
    pcaps): aggregated per-flow statistics feed the XGBoost / Isolation-Forest
    baselines. It deliberately cannot drive the Mamba arm -- packet order is gone.
    """
    cmap = category_map or IOT23_CATEGORY_MAP
    conn_states: Dict[str, int] = {}
    rows: List[List[float]] = []
    ys: List[int] = []
    for row in _read_zeek_tsv(conn_log_path):
        detailed = (_detailed_label(row) or "-").strip()
        category = cmap.get(detailed)
        if category is None:
            if drop_unlabeled:
                continue
            category = "Benign"
        feats: List[float] = []
        for col in _ZEEK_NUMERIC:
            val = row.get(col, "-")
            feats.append(float(val) if val not in ("-", "", None) else 0.0)
        feats.append(float(_ZEEK_PROTO_CODE.get(row.get("proto", ""), -1)))
        state = row.get("conn_state", "-")
        feats.append(float(conn_states.setdefault(state, len(conn_states))))
        rows.append(feats)
        ys.append(category_to_index(category))

    feature_names = list(_ZEEK_NUMERIC) + ["proto_code", "conn_state_code"]
    x = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(feature_names)))
    y = np.asarray(ys, dtype=np.int64)
    return x, y, feature_names


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
