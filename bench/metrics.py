"""Hardware monitoring during a run.

Two Prometheus exporter families are supported per host (configured in
each scenario under the host's ``metrics`` list):

  * ``macmon``  — https://github.com/DMontgomery40/macmon-prometheus-exporter
                  Apple Silicon power / energy / temp / utilization. Gauges are
                  ``macmon_*``-prefixed (chip carried as a ``chip`` label).
  * ``dcgm``    — NVIDIA dcgm-export (DCGM_FI_* metrics) for GPU hosts.

Samplers run in a background thread per host while the benchmark executes,
writing one row per poll into the stats DB. Efficiency (tokens-per-joule,
tokens/s per watt) is derived by correlating each request's time window with
the power time-series.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ------------------------------------------------------------- prometheus parse

_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-0-9.eE+]+)$")


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse the text exposition format into {name: [(labels, value), ...]}."""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name, raw_labels, raw_val = m.group(1), m.group(2), m.group(3)
        labels: dict[str, str] = {}
        if raw_labels:
            for part in raw_labels.split(","):
                k, _, v = part.partition("=")
                labels[k.strip()] = v.strip().strip('"')
        try:
            val = float(raw_val)
        except ValueError:
            continue
        out.setdefault(name, []).append((labels, val))
    return out


def detect_kind(parsed: dict[str, list]) -> str:
    for name in parsed:
        if name.startswith("mac_") or name.startswith("macmon_"):
            return "macmon"
        if name.startswith("DCGM_FI"):
            return "dcgm"
    return "prometheus"


# ------------------------------------------------------------------ normalize

def _g(name: str, parsed: dict) -> float | None:
    vals = parsed.get(name)
    return vals[0][1] if vals else None


def _glist(name: str, parsed: dict) -> list[float]:
    return [v for _, v in parsed.get(name, [])]


def _first(parsed: dict, *names: str) -> float | None:
    """First present gauge among ``names`` (new ``macmon_*`` first, legacy
    ``mac_*`` fallback)."""
    for name in names:
        v = _g(name, parsed)
        if v is not None:
            return v
    return None


def _pct(parsed: dict, *names: str) -> float | None:
    """First present value expressed as a percentage.

    New ``macmon_*_ratio`` gauges are 0-1 and get scaled x100; legacy
    ``mac_*_percent`` gauges already are percentages and pass through.
    """
    for name in names:
        v = _g(name, parsed)
        if v is None:
            continue
        return v * 100.0 if "percent" not in name else v
    return None


def _chip_label(parsed: dict[str, list]) -> str | None:
    """Any ``chip`` label carried by a macmon series (e.g. "Apple M3 Ultra")."""
    for entries in parsed.values():
        for labels, _ in entries:
            chip = labels.get("chip")
            if chip:
                return chip
    return None


def _filter_by_label(parsed: dict[str, list], label: str, value: str) -> dict[str, list]:
    """Keep only series entries whose ``label`` equals ``value`` (e.g. dcgm
    modelName filtering on multi-GPU hosts)."""
    out: dict[str, list] = {}
    for name, entries in parsed.items():
        kept = [(lab, v) for lab, v in entries if lab.get(label) == value]
        if kept:
            out[name] = kept
    return out


def normalize(kind: str, parsed: dict[str, list], gpu_model: str | None = None) -> dict[str, Any]:
    """Map raw exporter metrics into a common vocabulary.

    Values are floats (single-GPU / whole-machine) or lists (per-GPU). Also
    surfaces a couple of static ``info`` fields (chip / GPU model) useful for
    the machine spec. For ``dcgm``, ``gpu_model`` narrows collection to a
    single device model (per the exporter's ``modelName`` label); multiple
    devices of that model are still aggregated together.
    """
    if kind == "dcgm" and gpu_model:
        parsed = _filter_by_label(parsed, "modelName", gpu_model)
    m: dict[str, Any] = {}
    if kind == "macmon":
        m.update({
            # power
            "total_power_w": _first(parsed, "macmon_sys_power_watts", "mac_system_power_watts"),
            "all_power_w": _first(parsed, "macmon_all_power_watts", "mac_all_power_watts"),
            "cpu_power_w": _first(parsed, "macmon_cpu_power_watts", "mac_cpu_power_watts"),
            "gpu_power_w": _first(parsed, "macmon_gpu_power_watts", "mac_gpu_power_watts"),
            "ane_power_w": _first(parsed, "macmon_ane_power_watts", "mac_ane_power_watts"),
            "ram_power_w": _first(parsed, "macmon_ram_power_watts", "mac_ram_power_watts"),
            "gpu_ram_power_w": _g("macmon_gpu_ram_power_watts", parsed),
            # utilization (0-100). scaled = frequency-adjusted, active = not.
            "cpu_util_pct": _pct(parsed, "macmon_cpu_scaled_ratio", "macmon_cpu_usage_ratio", "mac_cpu_usage_percent"),
            "cpu_active_pct": _pct(parsed, "macmon_cpu_active_ratio"),
            "gpu_util_pct": _pct(parsed, "macmon_gpu_scaled_ratio", "macmon_gpu_usage_ratio", "mac_gpu_usage_percent"),
            "gpu_active_pct": _pct(parsed, "macmon_gpu_active_ratio"),
            "ecpu_util_pct": _pct(parsed, "macmon_ecpu_scaled_ratio", "macmon_ecpu_usage_ratio"),
            "ecpu_active_pct": _pct(parsed, "macmon_ecpu_active_ratio"),
            "pcpu_util_pct": _pct(parsed, "macmon_pcpu_scaled_ratio", "macmon_pcpu_usage_ratio"),
            "pcpu_active_pct": _pct(parsed, "macmon_pcpu_active_ratio"),
            "ane_util_pct": _g("mac_ane_usage_percent", parsed),
            "ram_util_pct": _g("mac_memory_usage_percent", parsed),
            # temperature
            "cpu_temp_c": _first(parsed, "macmon_cpu_temp_celsius", "mac_cpu_temperature_celsius"),
            "gpu_temp_c": _first(parsed, "macmon_gpu_temp_celsius", "mac_gpu_temperature_celsius"),
            # clocks
            "cpu_freq_mhz": _first(parsed, "macmon_pcpu_freq_mhz", "mac_pcpu_frequency_mhz"),
            "ecpu_freq_mhz": _g("macmon_ecpu_freq_mhz", parsed),
            "gpu_freq_mhz": _first(parsed, "macmon_gpu_freq_mhz", "mac_gpu_frequency_mhz"),
            # memory
            "ram_used_gb": (_first(parsed, "macmon_memory_ram_used_bytes", "mac_memory_usage_bytes") or 0) / 1e9,
            "ram_total_gb": (_first(parsed, "macmon_memory_ram_total_bytes", "mac_memory_total_bytes") or 0) / 1e9,
            "swap_used_gb": (_g("macmon_memory_swap_used_bytes", parsed) or 0) / 1e9,
            "swap_total_gb": (_g("macmon_memory_swap_total_bytes", parsed) or 0) / 1e9,
            # fans (one series per fan; reduce() takes the max)
            "fan_speed_rpm": _glist("macmon_fan_speed_rpm", parsed),
        })
        info: dict[str, str] = {}
        infos = parsed.get("mac_chip_info", [])
        if infos:
            info.update(infos[0][0])
        chip = _chip_label(parsed)
        if chip:
            info["chip"] = chip
        if info:
            m["info"] = info
    elif kind == "dcgm":
        m.update({
            "gpu_util_pct": _glist("DCGM_FI_DEV_GPU_UTIL", parsed),
            "gpu_mem_util_pct": _glist("DCGM_FI_DEV_MEM_COPY_UTIL", parsed),
            "gpu_power_w": _glist("DCGM_FI_DEV_POWER_USAGE", parsed),
            "gpu_temp_c": _glist("DCGM_FI_DEV_GPU_TEMP", parsed),
            "gpu_mem_temp_c": _glist("DCGM_FI_DEV_MEMORY_TEMP", parsed),
            "gpu_sm_clock_mhz": _glist("DCGM_FI_DEV_SM_CLOCK", parsed),
            "gpu_energy_ws": [
                v / 1000.0 for v in _glist("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", parsed)
            ],
            "gpu_enc_util_pct": _glist("DCGM_FI_DEV_ENC_UTIL", parsed),
            "gpu_dec_util_pct": _glist("DCGM_FI_DEV_DEC_UTIL", parsed),
        })
        for name, vals in parsed.items():
            if name in ("DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_POWER_USAGE"):
                if vals and "modelName" in vals[0][0]:
                    m["info"] = {
                        "gpu_model": vals[0][0]["modelName"],
                        "gpu_uuids": sorted({v[0].get("UUID", "") for v in vals}),
                        "gpu_count": len(vals),
                    }
    return {k: v for k, v in m.items() if v is not None and v != []}


def reduce(metrics: dict[str, Any]) -> dict[str, float]:
    """Collapse per-GPU lists to scalars (sum power/energy, max the rest)."""
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            continue  # info blobs (chip/gpu model), not time-series
        if isinstance(v, list):
            if not v:
                continue
            out[k] = float(sum(v) if k.endswith("_w") or k.endswith("_ws") else max(v))
        else:
            out[k] = float(v)
    return out


# -------------------------------------------------------------------- fetch

def fetch(
    url: str,
    kind: str = "auto",
    timeout: float = 5.0,
    gpu_model: str | None = None,
) -> dict[str, Any]:
    """Fetch + normalize an exporter. Returns {} on any failure."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.get(url)
            if r.status_code != 200:
                return {}
            parsed = parse_prometheus(r.text)
    except Exception:
        return {}
    if not parsed:
        return {}
    if kind == "auto":
        kind = detect_kind(parsed)
    return normalize(kind, parsed, gpu_model=gpu_model)


# ---------------------------------------------------------------- sampler

@dataclass(eq=False)
class MetricsSampler(threading.Thread):
    """Polls every exporter for one host at ``interval`` seconds.

    Stores ONE row per exporter per poll (tagged with the exporter's node name)
    in the stats DB, timestamped relative to the run start (same clock the
    request samples use) for window correlation. Cluster nodes therefore each
    contribute their own row, which the efficiency code sums.
    """

    exporters: list[Any]
    interval: float
    run_id: str
    host: str
    db: Any
    run_start: float  # time.time() at run start (wall clock; resumes keep the original)
    enabled: bool = True

    def __post_init__(self) -> None:
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self._fetches: int = 0

    def stop(self) -> None:
        self._stop.set()

    @property
    def fetches(self) -> int:
        return self._fetches

    def run(self) -> None:
        while not self._stop.is_set():
            ts = time.time() - self.run_start
            for exp in self.exporters:
                try:
                    kind = exp.type if hasattr(exp, "type") else "auto"
                    url = exp.url if hasattr(exp, "url") else str(exp)
                    node = exp.name if hasattr(exp, "name") and exp.name else str(url)
                    gpu_model = exp.gpu_model if hasattr(exp, "gpu_model") else None
                except AttributeError:
                    kind, url, node, gpu_model = "auto", str(exp), str(exp), None
                m = reduce(fetch(url, kind, gpu_model=gpu_model))
                if m:
                    self._fetches += 1
                    self.db.insert_metric(self.run_id, self.host, node, ts, m)
            self._stop.wait(self.interval)


# -------------------------------------------------------------- efficiency

def build_power_series(
    metric_rows: list[dict],
) -> list[tuple[float, str, dict[str, float]]]:
    """Precompute reduced per-node time-series: [(ts_epoch, node, scalars), ...]."""
    series: list[tuple[float, str, dict[str, float]]] = []
    for row in metric_rows:
        raw = row["metrics"]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
        if not isinstance(raw, dict):
            continue
        series.append((float(row["ts_epoch"]), str(row.get("node", "")), reduce(raw)))
    return series


def cluster_value_for_window(
    series: list[tuple[float, str, dict[str, float]]],
    start: float,
    end: float,
    metric: str,
    tol: float = 120.0,
) -> float:
    """Mean of ``metric`` per node over polls overlapping [start, end], then SUMMED.

    Polls are usually every 1-2s while short requests can be sub-second, so a
    tolerance window (~2 min) is used to always capture at least one poll per
    node. This is what makes cluster configs work: each node contributes its own
    power/utilization and the result is the cluster-wide figure.
    """
    per_node: dict[str, list[float]] = {}
    for ts, node, red in series:
        if start - tol <= ts <= end + tol and metric in red:
            per_node.setdefault(node, []).append(float(red[metric]))
    if not per_node:
        return 0.0
    return sum(sum(vals) / len(vals) for vals in per_node.values())


def pick_power_key(series: list[tuple[float, str, dict[str, float]]]) -> str | None:
    """Pick the total-power metric: macmon total > gpu power > any *_w."""
    if not series:
        return None
    keys = set().union(*(red.keys() for _, _, red in series))
    for pref in ("total_power_w", "all_power_w", "gpu_power_w"):
        if pref in keys:
            return pref
    for k in keys:
        if k.endswith("_w") or k.endswith("_ws"):
            return k
    return None


def sample_power(
    series: list[tuple[float, str, dict[str, float]]],
    start: float,
    end: float,
    metric: str,
) -> float:
    """Cluster-summed power over one sample's window.

    Uses polls strictly inside [start, end]; for short windows with no poll in
    them, each node falls back to its single nearest poll to the window center
    so the value reflects THIS request, not neighboring ones.
    """
    mid = (start + end) / 2.0
    per_node: dict[str, list[float]] = {}
    for ts, node, red in series:
        if metric in red and start <= ts <= end:
            per_node.setdefault(node, []).append(float(red[metric]))
    total = sum(sum(v) / len(v) for v in per_node.values())
    nodes = {n for _, n, _ in series}
    for node in nodes - set(per_node):
        cands = [
            (abs(ts - mid), float(red[metric]))
            for ts, n, red in series if n == node and metric in red
        ]
        if cands:
            total += min(cands)[1]
    return total


def efficiency_for_samples(
    samples: list[dict],
    power_series: list[tuple[float, str, dict[str, float]]],
    power_key: str,
) -> list[float]:
    """tokens/s per watt for each ok sample (empty if no power data)."""
    out: list[float] = []
    for s in samples:
        p = cluster_value_for_window(power_series, s.get("ts_start_epoch", 0.0),
                                      s.get("ts_end_epoch", 0.0), power_key)
        if p <= 0 or not s.get("out_tps"):
            continue
        out.append(s["out_tps"] / p)
    return out


# ------------------------------------------------------- per-run aggregation

def _aggregate_efficiency(
    samples: list[dict],
    metric_rows: list[dict],
) -> dict[str, dict[str, Any]]:
    """Per (phase, step [, concurrency]) median power/energy/tokens-per-watt.

    Power is SUMMED across cluster nodes (each row carries a node label), so a
    multi-node dgx cluster reports cluster-wide watts.
    """
    series = build_power_series(metric_rows)
    if not series:
        return {}
    power_key = pick_power_key(series)
    if not power_key:
        return {}
    nodes = sorted({node for _, node, _ in series})
    groups: dict[tuple, list[dict]] = {}
    for s in samples:
        # concurrent-batch phases keep each level separate (iter = users/concurrency)
        key = (s["phase"], s["step"], s["iter"]) if s["phase"] in ("conc", "mucold", "muwarm") else (s["phase"], s["step"])
        groups.setdefault(key, []).append(s)

    out: dict[str, dict[str, Any]] = {}
    for key, ss in sorted(groups.items()):
        per: list[dict[str, float]] = []
        for s in ss:
            start = s.get("ts_start_epoch") or 0.0
            end = s.get("ts_end_epoch") or 0.0
            p = cluster_value_for_window(series, start, end, power_key)
            if p <= 0:
                continue
            dur = (end - start) or (s.get("total_ms", 0) / 1000.0)
            energy = p * dur
            per.append({
                "power_w": p,
                "energy_ws": energy,
                "tg_tokens": s.get("tg_tokens") or 0,
                "out_tps": s.get("out_tps") or 0.0,
                "j_per_token": (energy / s["tg_tokens"]) if s.get("tg_tokens") else 0.0,
                "tps_per_w": (s["out_tps"] / p) if s.get("out_tps") else 0.0,
            })
        if not per:
            continue
        med = lambda k: float(sorted(x[k] for x in per)[len(per) // 2])
        label = ":".join(str(k) for k in key)
        out[label] = {
            "n": len(per),
            "power_key": power_key,
            "nodes": len(nodes),
            "avg_power_w": med("power_w"),
            "min_power_w": float(min(x["power_w"] for x in per)),
            "max_power_w": float(max(x["power_w"] for x in per)),
            "wh": med("energy_ws") / 3600.0,
            "energy_ws": med("energy_ws"),
            "j_per_token": med("j_per_token"),
            "tps_per_w": med("tps_per_w"),
        }
    return out


def run_aggregates(db: Any, run_id: str) -> dict[str, float]:
    """Run-wide summary stats for console output.

    * ppts  — total prompt tokens / total prefill (TTFT) time
    * tgs   — total generated tokens INCLUDING reasoning / total generation time
    * avg_w — cluster-summed mean watts across the whole run's power series
    * n     — ok sample count
    """
    samples = [s for s in db.samples(run_id) if s["ok"]]
    # new_prompt_tokens excludes the cached, already-processed prefix for
    # warm-cache samples (falls back to the full prompt_tokens for phases/
    # older runs that don't carry it), so ppts reflects tokens actually
    # prefilled rather than the full (possibly cached) context.
    total_prompt = float(sum(
        (s.get("new_prompt_tokens") if s.get("new_prompt_tokens") is not None
         else s.get("prompt_tokens")) or 0
        for s in samples
    ))
    prefill_s = sum(s.get("ttft_ms") or 0.0 for s in samples) / 1000.0
    total_gen = float(sum(
        (s.get("content_tokens") or 0) + (s.get("reasoning_tokens") or 0)
        for s in samples
    ))
    gen_s = sum(s.get("tg_ms") or 0.0 for s in samples) / 1000.0
    ppts = total_prompt / prefill_s if prefill_s > 0 else 0.0
    tgs = total_gen / gen_s if gen_s > 0 else 0.0

    avg_w = 0.0
    series = build_power_series(db.metric_samples(run_id))
    if series:
        pk = pick_power_key(series)
        if pk:
            per_node: dict[str, list[float]] = {}
            for _, node, red in series:
                if pk in red:
                    per_node.setdefault(node, []).append(float(red[pk]))
            avg_w = sum(sum(v) / len(v) for v in per_node.values())
    return {"n": len(samples), "ppts": ppts, "tgs": tgs, "avg_w": avg_w}


def compute_run_efficiency(db: Any, run_id: str) -> dict[str, dict[str, Any]]:
    samples = [s for s in db.samples(run_id) if s["ok"]]
    return _aggregate_efficiency(samples, db.metric_samples(run_id))


def compute_efficiency_all(db: Any) -> dict[str, dict[str, Any]]:
    """Latest-run efficiency per host: {run_id: {host, efficiency}}."""
    out: dict[str, dict[str, Any]] = {}
    for run in db.runs():
        rid = run["run_id"]
        rows = db.metric_samples(rid)
        if not rows:
            continue
        samples = [s for s in db.samples(rid) if s["ok"]]
        eff = _aggregate_efficiency(samples, rows)
        if eff:
            out[rid] = {"host": run["host"], "model": run["model"], "efficiency": eff}
    return out
