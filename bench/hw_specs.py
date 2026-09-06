"""Best-effort peak memory-bandwidth reference table, used only to draw a
dashed "roofline" guide line on the bandwidth chart — NOT used to compute a
"% of peak" figure anywhere, since that would additionally require the
model's active-parameter count and bit-width (unreliable to infer from a
model id string, especially for MoE models).

Values are vendor-published peak numbers (best case, unlikely to be hit in
practice) gathered from public spec sheets, not independently verified by
this tool. Treat them as an approximate reference line, not ground truth —
edit freely if a number looks off or a chip you use is missing.
"""
from __future__ import annotations

# label substring (lowercased) -> peak memory bandwidth, GB/s
PEAK_MEM_BANDWIDTH_GBPS: dict[str, float] = {
    "m5 ultra": 1200,          # quad-die M5 Ultra, Mac Studio (announced Aug 2026)
    "m5 max": 614,             # 40-core GPU variant; 460 for the 32-core part
    "m3 ultra": 819,
    "m2 ultra": 800,
    "m4 max": 546,
    "m3 max": 400,
    "m4 pro": 273,
    "m3 pro": 150,
    "gb10": 273,              # NVIDIA GB10 / DGX Spark (Grace-Blackwell superchip)
    "rtx pro 6000": 1792,     # Blackwell Workstation Edition, 96GB GDDR7
    "rtx 5090": 1792,         # 32GB GDDR7
    "rtx 4090": 1008,
    "h200": 4800,
    "h100": 3350,
    "a100": 2039,
}


def chip_label(machine_spec: dict | None) -> str | None:
    """Best-effort chip/GPU label from a run's machine_spec.exporter_info."""
    if not machine_spec:
        return None
    info = machine_spec.get("exporter_info") or {}
    for node_info in info.values():
        chip = node_info.get("chip") or node_info.get("gpu_model")
        if chip:
            return str(chip)
    return None


def peak_bandwidth_gbps(label: str | None) -> float | None:
    """Substring-match ``label`` against the table (most specific key wins)."""
    if not label:
        return None
    low = label.lower()
    match = None
    for key, val in PEAK_MEM_BANDWIDTH_GBPS.items():
        if key in low and (match is None or len(key) > len(match[0])):
            match = (key, val)
    return match[1] if match else None
