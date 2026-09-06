"""File-backed per-run result storage (no database).

Every run owns a self-contained directory:

    results/<scenario>/<run_id>/
      run.json       metadata (written at start, rewritten at finish)
      samples.jsonl  one JSON object per completed request, appended live
      metrics.jsonl  one JSON object per hardware-exporter poll, appended live

Runs append only to their own directory, so concurrent benchmarks on different
hosts never touch a shared file — no cross-process locking, no DB. Aggregation
happens afterwards in the report phase, which scans these directories.

Reads tolerate partially-written trailing JSONL lines (a live run may be
mid-append), so a report can be rendered while runs are still going.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic write (temp file + rename) so readers never see partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # partial trailing line from a concurrent writer
    return out


def concurrent_batch_aggregates(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one batch of concurrently-fired streams over their OVERLAPPING
    decode window, rather than summing each stream's standalone max out_tps.

    Each stream decodes during ``[ts_start + ttft, ts_end]``. The overlapping
    span is ``[max(decode_start), min(decode_end)]`` — the window in which every
    stream was generating at the same time. Inside that window each stream
    contributes tokens at its own measured rate (its ``tg_tokens`` scaled by the
    fraction of its decode window that lies within the overlap), and those are
    summed and divided by the overlap duration. That yields the throughput the
    host actually sustains while the streams are truly concurrent — as opposed
    to the full union span ``max(ts_end) - min(ts_start)``, which dilutes the
    rate with staggered prefill and any serialized (non-overlapping) time.

    Both views are returned: ``agg_tps`` (overlap-based, the concurrency-honest
    number) and ``agg_union_tps`` (achieved wall-clock rate over the whole union
    span), plus ``overlap_s`` / ``overlap_frac`` so callers can see how much of
    the batch actually overlapped. If the streams never overlap (e.g. the server
    serialized them), ``agg_tps`` falls back to the union rate and
    ``overlap_frac`` is 0.
    """
    wins: list[tuple[float, float, float]] = []  # (decode_start, decode_end, tg_tokens)
    for s in samples:
        start = s.get("ts_start_epoch")
        ttft = (s.get("ttft_ms") or 0.0) / 1000.0
        tok = s.get("tg_tokens") or 0.0
        if start is None or tok <= 0:
            continue
        dec_start = start + ttft
        tg_ms = s.get("tg_ms")
        dec_end = dec_start + (tg_ms / 1000.0) if tg_ms else (s.get("ts_end_epoch") or dec_start)
        if dec_end <= dec_start:
            continue
        wins.append((dec_start, dec_end, tok))

    if not wins:
        return {
            "agg_tps": float("nan"), "agg_union_tps": float("nan"),
            "wall_s": float("nan"), "overlap_s": 0.0, "overlap_frac": 0.0,
            "total_tokens": 0.0, "n": 0,
        }

    ov_start = max(w[0] for w in wins)
    ov_end = min(w[1] for w in wins)
    ov_s = max(ov_end - ov_start, 0.0)
    union_start = min(w[0] for w in wins)
    union_end = max(w[1] for w in wins)
    union_s = max(union_end - union_start, 1e-6)
    total_tok = sum(w[2] for w in wins)

    if ov_s > 0:
        ov_tok = 0.0
        for ds, de, tok in wins:
            frac = max(0.0, min(de, ov_end) - max(ds, ov_start)) / (de - ds)
            ov_tok += tok * frac
        agg = ov_tok / ov_s
    else:
        agg = total_tok / union_s

    return {
        "agg_tps": agg,
        "agg_union_tps": total_tok / union_s,
        "wall_s": union_s,
        "overlap_s": ov_s,
        "overlap_frac": ov_s / union_s,
        "total_tokens": total_tok,
        "n": len(wins),
    }


class ResultsStore:
    """Per-run file store with a StatsDB-compatible surface.

    ``scenario`` names the subdirectory under ``root`` that this store owns.
    When empty, read operations scan every scenario subdirectory (used by the
    cross-run report phase).
    """

    def __init__(self, root: str | Path, scenario: str = "") -> None:
        self.root = Path(root)
        self.scenario = scenario or ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------- layout

    def run_dir(self, run_id: str) -> Path:
        """Directory for a run under THIS store's scenario."""
        return self.root / self.scenario / run_id

    def _find_dir(self, run_id: str) -> Path | None:
        """Locate a run dir, preferring this store's scenario, then any."""
        cand = self.run_dir(run_id)
        if cand.is_dir():
            return cand
        if self.root.is_dir():
            for sc in sorted(self.root.iterdir()):
                d = sc / run_id
                if d.is_dir():
                    return d
        return None

    def _run_dirs(self, run_id: str | None = None) -> list[Path]:
        if run_id:
            d = self._find_dir(run_id)
            return [d] if d else []
        if not self.root.is_dir():
            return []
        out: list[Path] = []
        if self.scenario:
            sc = self.root / self.scenario
            if sc.is_dir():
                out = [d for d in sorted(sc.iterdir()) if d.is_dir()]
        else:
            for sc in sorted(self.root.iterdir()):
                if sc.is_dir():
                    out.extend(d for d in sorted(sc.iterdir()) if d.is_dir())
        return out

    # ------------------------------------------------------------- writes

    def start_run(
        self, run_id: str, host: str, model: str, tag: str, config: dict,
        start_epoch: float | None = None,
    ) -> None:
        with self._lock:
            _write_json(self.run_dir(run_id) / "run.json", {
                "run_id": run_id,
                "scenario": self.scenario,
                "started_at": _now(),
                "host": host,
                "model": model,
                "tag": tag,
                "config": config,
                "status": "running",
                "start_epoch": start_epoch,
            })

    def insert_sample(self, run_id: str, s: dict[str, Any]) -> None:
        row = dict(s)
        row.setdefault("run_id", run_id)
        row.setdefault("ts", _now())
        with self._lock:
            p = self.run_dir(run_id) / "samples.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(json.dumps(row) + "\n")

    def insert_metric(
        self, run_id: str, host: str, node: str, ts_epoch: float, metrics: dict[str, Any]
    ) -> None:
        with self._lock:
            p = self.run_dir(run_id) / "metrics.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(json.dumps({
                    "run_id": run_id, "host": host, "node": node,
                    "ts_epoch": ts_epoch, "metrics": metrics,
                }) + "\n")

    def finish_run(
        self, run_id: str, status: str, machine_spec: dict, model_spec: dict
    ) -> None:
        with self._lock:
            d = self._find_dir(run_id) or self.run_dir(run_id)
            prior = _read_json(d / "run.json") or {}
            prior.update({
                "finished_at": _now(),
                "status": status,
                "machine_spec": machine_spec or {},
                "model_spec": model_spec or {},
            })
            _write_json(d / "run.json", prior)

    # ------------------------------------------------------------- reads

    def run(self, run_id: str) -> dict[str, Any] | None:
        d = self._find_dir(run_id)
        return _read_json(d / "run.json") if d else None

    def runs(self) -> list[dict[str, Any]]:
        out = []
        for d in self._run_dirs():
            data = _read_json(d / "run.json")
            if data:
                out.append(data)
        out.sort(key=lambda r: r.get("started_at") or "")
        return out

    def samples(self, run_id: str | None = None, host: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for d in self._run_dirs(run_id):
            for s in _read_jsonl(d / "samples.jsonl"):
                if run_id and s.get("run_id") != run_id:
                    continue
                if host and s.get("host") != host:
                    continue
                rows.append(s)
        rows.sort(key=lambda s: s.get("ts") or "")
        return rows

    def metric_samples(self, run_id: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for d in self._run_dirs(run_id):
            for m in _read_jsonl(d / "metrics.jsonl"):
                if run_id and m.get("run_id") != run_id:
                    continue
                rows.append(m)
        rows.sort(key=lambda m: m.get("ts_epoch") or 0.0)
        return rows

    # ------------------------------------------------------------- aggregate

    def aggregate(self, run_id: str) -> dict[str, Any]:
        """Per (phase, step): count, median/p90/min/max of core metrics."""
        rows = [s for s in self.samples(run_id) if s["ok"]]
        groups: dict[tuple, list] = {}
        for s in rows:
            groups.setdefault((s["phase"], s["step"]), []).append(s)
        agg: dict[str, Any] = {}
        for (phase, step), ss in sorted(groups.items()):
            metrics = {
                "ttft_ms": [x["ttft_ms"] for x in ss],
                "tpot_ms": [x["tpot_ms"] for x in ss],
                "out_tps": [x["out_tps"] for x in ss],
                "tg_s": [x["tg_s"] for x in ss],
                "power_w": [x["power_w"] for x in ss],
                "wh": [x["wh"] for x in ss],
                "tps_per_w": [x["tps_per_w"] for x in ss],
                "prompt_tps": [x["prompt_tps"] for x in ss],
                "tg_tokens": [x["tg_tokens"] for x in ss],
            }
            entry: dict[str, Any] = {"n": len(ss)}
            for name, vals in metrics.items():
                a = np.asarray(vals, dtype=float)
                entry[name] = {
                    "median": float(np.median(a)),
                    "mean": float(np.mean(a)),
                    "min": float(np.min(a)),
                    "max": float(np.max(a)),
                    "p90": float(np.percentile(a, 90)),
                }
            agg[f"{phase}:{step}"] = entry
        return agg
