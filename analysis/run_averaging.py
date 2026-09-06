"""Run averaging for the report's main scenarios.

The qwen3.6/qwen3.8 scenarios have multiple *complete* runs per machine, and
decode (esp. @50k) is noisy run-to-run. Instead of picking one canonical run,
we average the per-(phase, step, iter, batch) measurements across all complete
runs of a (scenario, host). Cold/PW values that are stable barely move; the
volatile decode numbers get a more stable central estimate.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from collate import read_jsonl, read_json  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# scenarios that get run-averaged (all hosts)
AVG_SCENARIOS = {"qwen3.6-dflash", "qwen3.8-dflash"}

# Whole-box power adjustment: dcgm measures GPU power only. The DGX Spark
# cluster (gx10-top) is 2 nodes, so add ~70W of system draw per node.
POWER_ADJUST_W = {"gx10-top": 140.0}  # 2 nodes x 70W


def adjust_power(sample: dict) -> dict:
    """Add the per-node system-draw estimate to a sample's power and recompute
    tps_per_w. No-op for hosts without an adjustment."""
    add = POWER_ADJUST_W.get(sample.get("host"), 0.0)
    if add and sample.get("power_w") is not None:
        out = dict(sample)
        out["power_w"] = sample["power_w"] + add
        if sample.get("out_tps"):
            out["tps_per_w"] = sample["out_tps"] / out["power_w"]
        return out
    return sample

# per-sample numeric fields to average across runs
NUMERIC = [
    "ttft_ms", "prompt_tps", "tpot_ms", "tg_ms", "tg_tokens",
    "reasoning_tokens", "content_tokens", "tg_s", "out_tps",
    "power_w", "wh", "tps_per_w", "total_ms",
    "prompt_tokens", "new_prompt_tokens", "context_tokens", "max_tokens",
    "ts_start_epoch", "ts_end_epoch",
]


def complete_run_ids(scenario: str, host: str, frac: float = 0.9) -> list[str]:
    """Runs for (scenario, host) that are essentially full runs.

    Filters out resumable/partial runs (few ok samples or missing phases) by
    keeping only runs whose ok-sample count is near the max for that host.
    """
    sc_dir = RESULTS / scenario
    if not sc_dir.is_dir():
        return []
    runs: list[tuple[str, int]] = []
    for d in sc_dir.iterdir():
        if not d.is_dir():
            continue
        run = read_json(d / "run.json")
        if not run or run.get("host") != host:
            continue
        n = sum(1 for s in read_jsonl(d / "samples.jsonl") if s.get("ok"))
        if n:
            runs.append((run["run_id"], n))
    if not runs:
        return []
    maxn = max(n for _, n in runs)
    return [rid for rid, n in runs if n >= maxn * frac]


def averaged_samples(scenario: str, host: str) -> list[dict]:
    """Synthetic sample set for (scenario, host):

    * cold / warm (single-user) — averaged across all complete runs, since each
      (phase, step, iter, batch) key is one request, so averaging is clean and
      smooths the noisy decode numbers.
    * conc / mucold / muwarm (concurrent) — taken from the LATEST complete run.
      Concurrent streams share one (phase, step, iter, batch) key, and samples
      are appended in *completion* order (not firing order), so streams can't be
      aligned across runs; averaging would pair mismatched streams and fabricate
      overlap. These are single-batch measurements anyway, so the latest run is
      the honest choice.
    """
    rids = complete_run_ids(scenario, host)
    if not rids:
        return []
    latest = sorted(rids)[-1]
    single_phase = {"cold", "warm"}

    single_groups: dict[tuple, list[dict]] = defaultdict(list)
    for rid in rids:
        for s in read_jsonl(RESULTS / scenario / rid / "samples.jsonl"):
            if not s.get("ok"):
                continue
            if s["phase"] in single_phase:
                single_groups[(s["phase"], s["step"], s["iter"], s.get("batch", 0))].append(adjust_power(s))

    out: list[dict] = []
    runids = "+".join(rid.rsplit("-", 1)[0] for rid in rids)
    for key, rows in single_groups.items():
        base = dict(rows[0])
        base["n_runs"] = len(rows)
        base["run_id"] = runids
        for k in NUMERIC:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                base[k] = float(np.mean(vals))
        out.append(base)
    # concurrent phases: latest complete run
    for s in read_jsonl(RESULTS / scenario / latest / "samples.jsonl"):
        if s.get("ok") and s["phase"] not in single_phase:
            out.append(adjust_power(s))
    return out
