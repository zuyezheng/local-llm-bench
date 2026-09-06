"""Best-effort model + machine spec discovery. Nothing here is fatal."""
from __future__ import annotations

import re
import subprocess
from typing import Any

import httpx

_GAUGE_PREFIXES = ("vllm", "nv_", "nvidia", "rocm")


def fetch_model_list(url: str, timeout: float = 15.0) -> list[dict]:
    """GET /v1/models. Returns [] on any failure."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.get(f"{url}/v1/models")
            r.raise_for_status()
            return list(r.json().get("data", []))
    except Exception:
        return []


def discover_model(url: str, models: list[dict]) -> str | None:
    """Pick a model id, preferring a deepseek model when the host serves many."""
    if not models:
        return None
    for m in models:
        mid = str(m.get("id", ""))
        if "deepseek" in mid.lower():
            return mid
    return str(models[0]["id"])


def model_spec(url: str, model_id: str | None, models: list[dict]) -> dict[str, Any]:
    chosen = next((m for m in models if m.get("id") == model_id), models[0] if models else {})
    vendor = "unknown"
    owned = str(chosen.get("owned_by", "")).lower()
    if "vllm" in owned or "vllm" in url:
        vendor = "vllm"
    elif "omlx" in owned or "mlx" in owned or "lmstudio" in owned or "ollama" in owned:
        vendor = "mlx"
    return {
        "url": url,
        "model": model_id,
        "vendor": vendor,
        "max_model_len": chosen.get("max_model_len"),
        "owned_by": chosen.get("owned_by"),
        "all_models": [m.get("id") for m in models],
        "raw": chosen,
    }


def fetch_prometheus_metrics(url: str, timeout: float = 15.0) -> dict[str, float]:
    """Grab the latest value of every vllm/nv prometheus metric. Best effort."""
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"{url}/metrics")
            if r.status_code != 200:
                return {}
            out: dict[str, float] = {}
            for line in r.text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-0-9.eE+]+)$", line)
                if not m:
                    continue
                name, val = m.group(1), m.group(2)
                if name.startswith(_GAUGE_PREFIXES):
                    try:
                        out[name] = float(val)
                    except ValueError:
                        pass
            return out
    except Exception:
        return {}


def fetch_machine_spec(ssh_host: str | None, timeout: float = 10.0) -> dict[str, Any]:
    """Collect server-side hardware specs over ssh. Returns {} if not configured/possible."""
    if not ssh_host:
        return {"source": "unavailable", "reason": "ssh not configured"}
    script = (
        "echo hostname=$(hostname);"
        "echo os=$(uname -s);"
        "echo arch=$(uname -m);"
        "echo kernel=$(uname -r);"
        "echo ncpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null);"
        "echo mem_gb=$(awk '/MemTotal/{printf \"%.0f\", $2/1024/1024}' /proc/meminfo 2>/dev/null || "
        "sysctl -n hw.memsize 2>/dev/null);"
        "nvidia-smi --query-gpu=name,memory_total,driver_version --format=csv,noheader 2>/dev/null "
        "| sed 's/^/gpu=/';"
        "lspci 2>/dev/null | grep -iE 'vga|nvidia|amd.*gpu' | head -4 | sed 's/^/pci=/';"
    )
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_host, script],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError) as e:
        return {"source": "ssh", "error": str(e)}
    if proc.returncode != 0:
        return {"source": "ssh", "error": proc.stderr.strip()[:500]}
    spec: dict[str, Any] = {"source": "ssh"}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        if key in ("gpu", "pci"):
            spec.setdefault(key, []).append(val)
        else:
            spec[key] = val
    return spec
