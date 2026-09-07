"""Aggregate the collated samples into per-(cohort, phase, key) median tables.

Cohort = (scenario, host). Canonical run per cohort = the run with the most
ok samples (handles resumed/partial runs). Produces a small set of nested dicts
used by the chart builder and printed here for the analysis narrative.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # repo root -> import bench

from collate import read_jsonl, read_json  # noqa: E402
from bench.results import concurrent_batch_aggregates  # noqa: E402
from cache_revision import APPLE_HOSTS, revised_warm_turns, revised_batch  # noqa: E402
from run_averaging import AVG_SCENARIOS, averaged_samples, adjust_power  # noqa: E402

RESULTS = ROOT.parent / "results"

# ---------------------------------------------------------------- metadata
# host -> (hardware class, chip label, peak mem bandwidth GB/s, host display)
HOST_META: dict[str, dict] = {
    "m5-max":      {"class": "Apple Silicon", "chip": "Apple M5 Max",  "bw": 614,
                    "display": "M5 Max (Apple)", "family": "apple"},
    "macstudio":   {"class": "Apple Silicon", "chip": "Apple M3 Ultra", "bw": 819,
                    "display": "M3 Ultra (Apple)", "family": "apple"},
    "gx10-top":    {"class": "DGX Spark", "chip": "NVIDIA GB10 x2 (DGX Spark)", "bw": 273,
                    "display": "GB10 x2 (DGX Spark)", "family": "nvidia"},
    "tr-pro-6000": {"class": "Workstation GPU", "chip": "RTX PRO 6000", "bw": 1792,
                    "display": "RTX PRO 6000", "family": "nvidia"},
    "tr-pro-5090": {"class": "Workstation GPU", "chip": "RTX 5090", "bw": 1792,
                    "display": "RTX 5090", "family": "nvidia"},
    "tr-pro":      {"class": "Workstation GPU", "chip": "RTX PRO 6000", "bw": 1792,
                    "display": "RTX PRO 6000", "family": "nvidia"},
}

# scenario -> architecture + human model label
#
# `arch` is the model's structural class. Qwen3.8 Next is a MoE; what the run
# artifacts do NOT record are its parameter counts — they keep only what the
# servers advertise (model id `Qwen3.8-Flash-Next`, GGUF Q8_0 on the Mac, NVFP4
# on vLLM, max_model_len 262144), so its `params_*` stay "n/r" = not recorded
# and any claim about it is sized from measured behaviour, not a datasheet.
SCENARIO_META: dict[str, dict] = {
    "qwen3.6-dflash": {"arch": "MoE",   "label": "Qwen3.6-35B-A3B", "params_total": "35B", "params_active": "3B",  "short": "Qwen3.6 (MoE)"},
    "qwen3.8-dflash": {"arch": "Dense", "label": "Qwen3.8-27B",    "params_total": "27B", "params_active": "27B", "short": "Qwen3.8 (dense)"},
    "qwen3.8-2":      {"arch": "Dense", "label": "Qwen3.8-27B",    "params_total": "27B", "params_active": "27B", "short": "Qwen3.8 (dense)"},
    "qwen3.8":        {"arch": "Dense", "label": "Qwen3.8-27B",    "params_total": "27B", "params_active": "27B", "short": "Qwen3.8 (dense)"},
    "qwen3.8-next":   {"arch": "MoE",   "label": "Qwen3.8 Next",     "params_total": "n/r", "params_active": "n/r", "short": "Qwen3.8 Next (MoE)"},
    "deepseekv4":     {"arch": "MoE",   "label": "DeepSeek-V4-Flash", "params_total": "284B", "params_active": "13B", "short": "DeepSeek-V4 (MoE)"},
}


def canonical_run_id(scenario: str, host: str) -> str | None:
    """Pick the run with the most ok samples for (scenario, host)."""
    best: tuple[int, str | None] = (-1, None)
    sc_dir = RESULTS / scenario
    if not sc_dir.is_dir():
        return None
    for run_dir in sc_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run = read_json(run_dir / "run.json")
        if not run or run.get("host") != host:
            continue
        n = sum(1 for s in read_jsonl(run_dir / "samples.jsonl") if s.get("ok"))
        # on a tie, prefer the most recent run_id (timestamps sort chronologically)
        if n > best[0] or (n == best[0] and run["run_id"] > (best[1] or "")):
            best = (n, run["run_id"])
    return best[1]


def _load_all_samples() -> dict[str, list[dict]]:
    """run_id -> ok samples, across all scenarios."""
    out: dict[str, list[dict]] = defaultdict(list)
    for scenario_dir in sorted(RESULTS.iterdir()):
        if not scenario_dir.is_dir() or scenario_dir.name == "report":
            continue
        for run_dir in sorted(scenario_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run = read_json(run_dir / "run.json")
            if not run:
                continue
            for s in read_jsonl(run_dir / "samples.jsonl"):
                if s.get("ok"):
                    out[run["run_id"]].append(adjust_power(s))
    return out


def median(vals: list[float]) -> float:
    return float(np.median(vals)) if vals else float("nan")


def build_dataset() -> dict[str, Any]:
    """Assemble a dataset keyed by cohort -> phases -> aggregated numbers."""
    all_samples = _load_all_samples()
    ds: dict[str, dict] = {}

    cohorts = sorted({(sc, h) for sc in SCENARIO_META
                      for h in HOST_META
                      if canonical_run_id(sc, h) is not None})

    for scenario, host in cohorts:
        rid = canonical_run_id(scenario, host)
        if scenario in AVG_SCENARIOS:
            samples = averaged_samples(scenario, host)
            rid = "averaged"
        else:
            samples = all_samples.get(rid, [])
        if not samples:
            continue

        cold = [s for s in samples if s["phase"] == "cold"]
        warm = [s for s in samples if s["phase"] == "warm"]
        mucold = [s for s in samples if s["phase"] == "mucold"]
        muwarm = [s for s in samples if s["phase"] == "muwarm"]

        # Apple/omlx: revise warm turns (and multi-user warm batches) to assume
        # vLLM-equivalent prefix caching — matches site_data.py and the
        # ANALYSIS.md caveat ("Apple warm-cache / session / team numbers are
        # revised"), which previously only held for the interactive site.
        if host in APPLE_HOSTS and scenario != "deepseekv4":
            wmap = {s["step"]: s for s in warm}
            warm = list(revised_warm_turns(wmap, cold).values())
            muwarm = revised_batch(muwarm, cold)

        def grp(rows, by):
            out: dict = defaultdict(list)
            for s in rows:
                out[by(s)].append(s)
            return out

        def med_of(rows, key):
            return median([s[key] for s in rows if s.get(key) is not None])

        def stat(rows, key):
            vals = [s[key] for s in rows if s.get(key) is not None]
            if not vals:
                return {"median": float("nan"), "mean": float("nan"),
                        "p90": float("nan"), "n": 0}
            return {"median": float(np.median(vals)), "mean": float(np.mean(vals)),
                    "p90": float(np.percentile(vals, 90)), "n": len(vals)}

        def stat_of(rows, fn):
            vals = [fn(s) for s in rows if s.get("ttft_ms") and s.get("prompt_tokens")]
            if not vals:
                return {"median": float("nan"), "mean": float("nan"),
                        "p90": float("nan"), "n": 0}
            return {"median": float(np.median(vals)), "mean": float(np.mean(vals)),
                    "p90": float(np.percentile(vals, 90)), "n": len(vals)}

        cold_agg = {ctx: stat(rows, "ttft_ms") and {
            "ttft_ms": stat(rows, "ttft_ms"),
            "prompt_tps": stat(rows, "prompt_tps"),
            "out_tps": stat(rows, "out_tps"),
            "tpot_ms": stat(rows, "tpot_ms"),
            "power_w": stat(rows, "power_w"),
            "tps_per_w": stat(rows, "tps_per_w"),
        } for ctx, rows in grp(cold, lambda s: s["step"]).items()}

        warm_agg = {ctx: {
            "ttft_ms": stat(rows, "ttft_ms"),
            "prompt_tps": stat(rows, "prompt_tps"),
            # effective prompt processing = FULL context tokens / TTFT, so it is
            # directly comparable to cold prompt_tps (warm's prompt_tps field
            # only counts NEW tokens, which is a different base).
            "prompt_tps_eff": stat_of(rows, lambda s: s["prompt_tokens"] / (s["ttft_ms"] / 1000.0)),
            "out_tps": stat(rows, "out_tps"),
            "tg_ms": stat(rows, "tg_ms"),
            "total_ms": stat(rows, "total_ms"),
            "power_w": stat(rows, "power_w"),
            "tps_per_w": stat(rows, "tps_per_w"),
        } for ctx, rows in grp(warm, lambda s: s["step"]).items()}

        # mucold/muwarm: group by (context, users) where users = iter, then
        # aggregate per batch over the streams' OVERLAPPING decode windows
        # (median across batches), so agg_tps reflects real concurrency rather
        # than a union span diluted by staggered prefill / serialized time.
        def concurrent_agg(rows, users):
            per_batch = []
            for (_ctx, _u, _b), batch in sorted(
                grp(rows, lambda s: (s["step"], s["iter"], s.get("batch", 0))).items()
            ):
                agg = concurrent_batch_aggregates(batch)
                if agg["n"] == 0:
                    continue
                pw = float(np.mean([s["power_w"] for s in batch
                                   if s.get("power_w") is not None]) or float("nan"))
                per_batch.append({
                    **agg,
                    "mean_ttft_ms": float(np.mean([s["ttft_ms"] for s in batch])),
                    "mean_latency_ms": float(np.mean([s["total_ms"] for s in batch])),
                    "power_w": pw,
                    "tps_per_w": agg["agg_tps"] / pw if pw == pw and pw > 0 else float("nan"),
                })
            if not per_batch:
                return {}
            med = lambda k: float(sorted(x[k] for x in per_batch)[len(per_batch) // 2])
            return {
                "agg_tps": med("agg_tps"),
                "agg_union_tps": med("agg_union_tps"),
                "per_stream_tps": med("agg_tps") / users if users else float("nan"),
                "overlap_frac": med("overlap_frac"),
                "overlap_s": med("overlap_s"),
                "wall_s": med("wall_s"),
                "mean_ttft_ms": med("mean_ttft_ms"),
                "mean_latency_ms": med("mean_latency_ms"),
                "power_w": med("power_w"),
                "tps_per_w": med("tps_per_w"),
                "n": len(rows),
            }

        mucold_agg = {}
        for (ctx, users), rows in grp(mucold, lambda s: (s["step"], s["iter"])).items():
            a = concurrent_agg(rows, users)
            if a:
                mucold_agg[(ctx, users)] = a

        muwarm_agg = {}
        for (ctx, users), rows in grp(muwarm, lambda s: (s["step"], s["iter"])).items():
            a = concurrent_agg(rows, users)
            if a:
                muwarm_agg[(ctx, users)] = a

        ds[(scenario, host)] = {
            "scenario": scenario,
            "host": host,
            "arch": SCENARIO_META[scenario]["arch"],
            "model_label": SCENARIO_META[scenario]["label"],
            "short": SCENARIO_META[scenario]["short"],
            "host_meta": HOST_META[host],
            "run_id": rid,
            "n_samples": len(samples),
            "cold": cold_agg,
            "warm": warm_agg,
            "mucold": mucold_agg,
            "muwarm": muwarm_agg,
        }
    return ds


def _json_safe(obj):
    if isinstance(obj, dict):
        return {_jsk(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _jsk(k):
    return f"{k[0]}:{k[1]}" if isinstance(k, tuple) else str(k)


def main() -> None:
    ds = build_dataset()

    print("=" * 100)
    print("COHORTS (scenario x host) with cold-cache decode throughput (out_tps)")
    print("=" * 100)
    header = f"{'scenario':<15}{'host':<14}{'arch':<6}{'run_id':<26}{'n':>5}  " + "  ".join(f"c{ctx}" for ctx in (1000, 10000, 50000, 100000))
    print(header)
    for (sc, h), d in sorted(ds.items()):
        cells = []
        for ctx in (1000, 10000, 50000, 100000):
            st = d["cold"].get(ctx)
            cells.append(f"{st['out_tps']['median'] if st else float('nan'):8.1f}")
        print(f"{sc:<15}{h:<14}{d['arch']:<6}{d['run_id']:<26}{d['n_samples']:>5}  " + "  ".join(cells))

    print()
    print("=" * 100)
    print("TTFT (ms) cold cache — prefill latency")
    print("=" * 100)
    for ctx in (1000, 10000, 50000, 100000):
        cells = []
        for (sc, h), d in sorted(ds.items()):
            st = d["cold"].get(ctx)
            cells.append(f"{h}={st['ttft_ms']['median'] if st else float('nan'):,.0f}")
        print(f"ctx={ctx:>6}: " + "  ".join(cells))

    # dump full dataset json for the chart builder (tuple keys -> strings)
    serial = {f"{k[0]}__{k[1]}": _json_safe(v) for k, v in ds.items()}
    (ROOT / "dataset.json").write_text(json.dumps(serial, default=str, indent=2))
    print()
    print(f"dataset written: {ROOT / 'dataset.json'}")

    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("WARM CACHE — TTFT cold vs warm at ~40-50k context (prefix caching effect)")
    print("=" * 100)
    for ctx in (40000, 42000, 44000, 46000, 48000, 50000):
        cells = []
        for (sc, h), d in sorted(ds.items()):
            w = d["warm"].get(ctx)
            c = d["cold"].get(50000) or d["cold"].get(100000)
            if not w:
                continue
            wt = w["ttft_ms"]["median"]
            ct = c["ttft_ms"]["median"] if c else float("nan")
            cells.append(f"{h}[{sc[:7]}] warm={wt:,.0f} cold50k={ct:,.0f}  ({ct/wt:.0f}x)")
        print(f"ctx={ctx}:")
        for c in cells:
            print("   " + c)

    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("MULTI-USER COLD — aggregate TG (tok/s) vs single-user, at ctx=1000 and 10000")
    print("(agg_tps measured over the OVERLAPPING decode window; 'ov' = fraction of")
    print(" the union span in which all streams were decoding together)")
    print("=" * 100)
    for ctx in (1000, 10000):
        print(f"--- context {ctx} ---")
        for (sc, h), d in sorted(ds.items()):
            base = d["cold"].get(ctx, {}).get("out_tps", {}).get("median", float("nan"))
            row = [f"base={base:.0f}"]
            for users in (2, 4, 6):
                m = d["mucold"].get((ctx, users), {})
                agg = m.get("agg_tps")
                if agg:
                    ov = m.get("overlap_frac")
                    ovs = f" ov{ov:.0%}" if ov is not None else ""
                    row.append(f"{users}u={agg:.0f} ({agg/base:.1f}x{ovs})" if base else f"{users}u={agg:.0f}")
            print(f"   {h:<14}[{sc[:7]}]: " + "  ".join(row))



if __name__ == "__main__":
    main()
