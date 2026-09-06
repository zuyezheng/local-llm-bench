"""Configuration: hosts, model, benchmark sizing, and storage paths."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENARIO_PATH = PROJECT_ROOT / "config" / "scenarios" / "deepseekv4.yaml"


@dataclass
class MetricsExporter:
    """A single prometheus endpoint (one node). Internal flattened form."""
    url: str
    type: str = "auto"  # macmon | dcgm | auto (auto sniffs metric names)
    name: str = ""      # node label (defaults to the URL hostname); power is summed across nodes
    gpu_model: Optional[str] = None  # dcgm: only collect GPUs with this modelName

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")
        if not self.name:
            self.name = self.url.split("//")[-1].split(":")[0].split(".")[0]


@dataclass
class MetricsGroup:
    """One exporter family on a host: a type plus one-or-more URLs (cluster nodes).

    Multiple URLs of the same type are treated as one cluster whose power and
    utilization are SUMMED (e.g. the two DGX Spark nodes of the gx10 cluster).
    """
    urls: list[str] = field(default_factory=list)
    type: str = "auto"  # macmon | dcgm | auto
    gpu_model: Optional[str] = None  # dcgm: only collect GPUs matching this modelName

    def exporters(self) -> list[MetricsExporter]:
        return [
            MetricsExporter(url=u, type=self.type, gpu_model=self.gpu_model)
            for u in self.urls
        ]


@dataclass
class HostConfig:
    url: str
    name: str = ""
    model: Optional[str] = None       # model id on this host; overrides global model
    ssh: Optional[str] = None  # e.g. "user@<mac-host>" for server-spec collection
    flush_cache_path: Optional[str] = None  # e.g. "/flush_cache" if server exposes it
    max_context: Optional[int] = None  # optional hard cap for this host
    metrics: list[MetricsGroup] = field(default_factory=list)  # exporter families

    def __post_init__(self) -> None:
        self.url = self.url.rstrip("/")
        if not self.name:
            self.name = self.url.split("//")[-1].split(":")[0]
        norm: list[MetricsGroup] = []
        for m in self.metrics:
            if isinstance(m, MetricsGroup):
                norm.append(m)
            elif isinstance(m, str):
                norm.append(MetricsGroup(urls=[m]))
            elif isinstance(m, list):
                norm.append(MetricsGroup(urls=[str(x) for x in m]))
            elif isinstance(m, dict):
                norm.append(MetricsGroup(**{k: v for k, v in m.items() if v is not None}))
        self.metrics = norm

    @property
    def base_url(self) -> str:
        return self.url

    def all_exporters(self) -> list[MetricsExporter]:
        out: list[MetricsExporter] = []
        for g in self.metrics:
            out.extend(g.exporters())
        return out


@dataclass
class Config:
    name: str = ""  # scenario id (e.g. "deepseekv4")
    hosts: list[HostConfig] = field(default_factory=list)
    model: Optional[str] = None  # auto-discovered from /v1/models when None

    # --- benchmark sizing ---
    max_tokens: int = 1024
    temperature: float = 0.0
    context_lengths: list[int] = field(
        default_factory=lambda: [1000, 10000, 50000, 100000, 200000, 500000]
    )
    # warm-cache "coding session": context grows in small 10k steps up to 200k
    warm_sizes: list[int] = field(
        default_factory=lambda: list(range(10000, 200001, 10000))
    )
    # concurrency sweep: concurrent streams per context (capped at 100k context)
    concurrency: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    concurrency_contexts: list[int] = field(default_factory=lambda: [1000, 10000, 50000, 100000])
    conc_batches: int = 1
    # multi-user: N users, each with a DISTINCT (non-shared) context slice
    multi_users: list[int] = field(default_factory=list)            # e.g. [2, 4, 8]
    multi_user_contexts: list[int] = field(default_factory=list)    # cold contexts per user
    multi_user_warm_sizes: list[int] = field(default_factory=list)  # per-user warm steps
    multi_user_batches: int = 1
    cold_iters: int = 3
    warm_sessions: int = 1
    skip_cold: bool = False
    skip_warm: bool = False
    skip_conc: bool = False
    skip_multiuser: bool = False

    # --- hardware monitoring ---
    metrics_interval: float = 2.0   # seconds between exporter polls per host
    metrics_enabled: bool = True

    # --- sampling / cache behavior ---
    encoding: str = "o200k_base"  # tiktoken encoding used to size prompts
    cold_prefix: str = "random"  # "random" busts prefix cache; "deterministic" for A/B
    corpus_kind: str = "mixed"   # code | writing | mixed (mixing weights below)
    code_ratio: float = 0.7
    warm_kind: str = "code"      # warm coding sessions use code corpus
    ignore_eos: bool = True      # force full max_tokens output for stable TG
    seed: int = 0

    # --- storage ---
    corpus_dir: Path = PROJECT_ROOT / "corpora"
    results_dir: Path = PROJECT_ROOT / "results"

    # --- client timeouts ---
    # timeout: HARD total cap per request (across retries). When a server kills
    # a request (omlx aborts long prefills), the client fails within read_timeout
    # instead of waiting forever.
    timeout: float = 1800.0
    # max silence between chunks for one attempt (per-request budget caps it tighter)
    read_timeout: float = 900.0
    connect_timeout: float = 15.0
    max_retries: int = 1
    # per-request read budget: min(read_timeout,
    #   max(min_read_timeout, prompt_tokens / min_pp_tps + ttft_margin))
    min_pp_tps: float = 300.0     # floor on prompt-processing speed used for the budget
    min_read_timeout: float = 300.0
    ttft_margin: float = 240.0

    # --- corpus fetch ---
    corpus_code_urls: list[str] = field(default_factory=list)
    corpus_writing_urls: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        cfg = cls()
        if path and path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
            cfg.apply_dict(raw)
        elif path:
            print(f"[config] {path} not found; using defaults.")
        return cfg

    @classmethod
    def from_scenario(
        cls, scenario_path: Path, base_path: Optional[Path] = None
    ) -> "Config":
        cfg = cls.load(base_path)
        raw = yaml.safe_load(scenario_path.read_text()) or {}
        cfg.apply_dict(raw)
        return cfg

    def apply_dict(self, raw: dict[str, Any]) -> None:
        # Scenario configs are authoritative: a present `hosts` key REPLACES
        # whatever was loaded before (base config or a parent file).
        if "hosts" in raw:
            hosts_raw = raw.get("hosts") or []
            self.hosts = []
            for h in hosts_raw:
                if isinstance(h, str):
                    h = {"url": h}
                self.hosts.append(HostConfig(**{k: v for k, v in h.items() if v is not None}))

        if "name" in raw and raw["name"]:
            self.name = str(raw["name"])

        scalar = [
            "model", "max_tokens", "temperature", "context_lengths", "warm_sizes",
            "concurrency", "concurrency_contexts", "conc_batches",
            "multi_users", "multi_user_contexts", "multi_user_warm_sizes",
            "multi_user_batches",
            "cold_iters", "warm_sessions", "skip_cold", "skip_warm", "skip_conc",
            "skip_multiuser",
            "metrics_interval", "metrics_enabled",
            "encoding", "cold_prefix", "corpus_kind", "code_ratio",
            "warm_kind", "ignore_eos", "seed", "timeout", "max_retries",
            "read_timeout", "connect_timeout", "min_pp_tps", "min_read_timeout",
            "ttft_margin",
        ]
        for k in scalar:
            if k in raw and raw[k] is not None:
                setattr(self, k, raw[k])

        path_keys = ["corpus_dir", "results_dir"]
        for k in path_keys:
            if k in raw and raw[k] is not None:
                setattr(self, k, Path(raw[k]).expanduser().resolve())

        for k in ("corpus_code_urls", "corpus_writing_urls"):
            if k in raw:
                setattr(self, k, list(raw[k]))

    @classmethod
    def from_file(cls, path: Path, overrides: dict[str, Any] | None = None) -> "Config":
        cfg = cls.load(path)
        if overrides:
            cfg.apply_dict(overrides)
        return cfg

    def serializable_config(self) -> dict[str, Any]:
        """Everything needed to reproduce/resume a run (sizing + seed)."""
        return {
            "name": self.name,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "context_lengths": list(self.context_lengths),
            "warm_sizes": list(self.warm_sizes),
            "concurrency": list(self.concurrency),
            "concurrency_contexts": list(self.concurrency_contexts),
            "conc_batches": self.conc_batches,
            "multi_users": list(self.multi_users),
            "multi_user_contexts": list(self.multi_user_contexts),
            "multi_user_warm_sizes": list(self.multi_user_warm_sizes),
            "multi_user_batches": self.multi_user_batches,
            "cold_iters": self.cold_iters,
            "warm_sessions": self.warm_sessions,
            "skip_cold": self.skip_cold,
            "skip_warm": self.skip_warm,
            "skip_conc": self.skip_conc,
            "skip_multiuser": self.skip_multiuser,
            "corpus_kind": self.corpus_kind,
            "code_ratio": self.code_ratio,
            "warm_kind": self.warm_kind,
            "cold_prefix": self.cold_prefix,
            "metrics_interval": self.metrics_interval,
            "metrics_enabled": self.metrics_enabled,
        }


def default_scenario_yaml() -> str:
    return """\
# llm-bench scenario — self-contained (model, machines, workload, tag).
# Usage:  uv run llm-bench run --scenario config/scenarios/example.yaml

name: example
tag: example

max_tokens: 1024
temperature: 0.0
cold_iters: 3
warm_sessions: 1

context_lengths: [1000, 10000, 50000, 100000]
warm_sizes: [10000, 20000, 30000, 40000, 50000]
concurrency: [1, 2, 3, 4]
concurrency_contexts: [1000, 10000, 50000, 100000]

metrics_interval: 2.0
metrics_enabled: true

hosts:
  - name: myhost
    url: http://localhost:8000
    model: null                   # auto-discovered from /v1/models when null
    # exporter families; multiple urls of the same type = cluster nodes whose
    # power/utilization is summed for this host
    metrics:
      - type: macmon              # macmon-prometheus-exporter (Apple Silicon)
        urls:
          - http://localhost:9101/metrics
      # - type: dcgm              # NVIDIA dcgm-export; two nodes = one cluster
      #   urls:
      #     - http://node-a:9400/metrics
      #     - http://node-b:9400/metrics
      #   gpu_model: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
      #             # dcgm only: collect just this device model (exporter modelName label)
"""
