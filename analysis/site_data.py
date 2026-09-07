"""Generate the embedded dataset for the interactive engineering-session site.

Reads the canonical runs (same selection as analyze.build_dataset) and emits
analysis/site/data.js as `window.SITE_DATA = {...}` with per-cohort:
  - warm engineering-session: per-turn TTFT / generation / total / throughput
  - session totals (real warm) + no-cache estimate (all turns cold)
  - where the time goes (prefill vs generation share)
  - team (multi-user warm) wall time per engineer count
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from collate import read_jsonl, read_json  # noqa: E402
from bench.results import concurrent_batch_aggregates  # noqa: E402
import cache_revision as rev  # noqa: E402
from run_averaging import AVG_SCENARIOS, averaged_samples, adjust_power, complete_run_ids, POWER_ADJUST_W  # noqa: E402

RESULTS = ROOT.parent / "results"
OUT = ROOT / "site" / "data.js"

STEPS = [40000, 42000, 44000, 46000, 48000, 50000]

SCENARIO_META = {
    "qwen3.6-dflash": {"arch": "MoE",   "label": "Qwen3.6-35B-A3B", "short": "Qwen3.6 · MoE"},
    "qwen3.8-dflash": {"arch": "Dense", "label": "Qwen3.8-27B",    "short": "Qwen3.8 · Dense"},
    "qwen3.8-next":   {"arch": "MoE",   "label": "Qwen3.8 Next",       "short": "Qwen3.8 Next · MoE"},
    "deepseekv4":     {"arch": "MoE",   "label": "DeepSeek-V4-Flash", "short": "DeepSeek V4"},
}
HOST_META = {
    "m5-max":      {"display": "M5 Max (Apple)",  "class": "Apple",   "color": "#0e7490"},
    "macstudio":   {"display": "M3 Ultra (Apple)","class": "Apple",   "color": "#2563eb"},
    "gx10-top":    {"display": "GB10 x2 (DGX Spark)", "class": "NVIDIA", "color": "#059669"},
    "tr-pro-5090": {"display": "RTX 5090",        "class": "NVIDIA",  "color": "#d97706"},
    "tr-pro-6000": {"display": "RTX PRO 6000",    "class": "NVIDIA",  "color": "#7c3aed"},
    # projection, not a measurement
    "m5ultra":     {"display": "M5 Ultra (projected)", "class": "Apple", "color": "#0ea5e9", "projected": True},
}
HOST_ORDER = ["m5-max", "macstudio", "gx10-top", "tr-pro-5090", "tr-pro-6000", "m5ultra"]
ARCH_COLOR = {"MoE": "#1d4ed8", "Dense": "#dc2626", "DeepSeek": "#111827", "Next": "#c026d3"}

# ---------------------------------------------------------------------------
# M5 Ultra projection — dual-anchor scaling, not one marketing multiplier.
#
# Two independent, physically-motivated paths to the same target, computed
# from OUR measured cold@50k numbers (never cache-revised):
#
#   Path A — M3 Ultra x official Ultra-vs-Ultra deltas (two generations apart):
#     bandwidth       1.2TB/s / 800GB/s        = 1.50x   -> decode  (bw-bound)
#     peak GPU AI compute   Apple: "up to 4.5x" vs M3 Ultra -> prefill (compute-bound)
#
#   Path B — M5 Max x physical Max->Ultra fusion ratio (same generation,
#   same architecture, exactly two dies):
#     bandwidth       1.2TB/s / 614GB/s (M5 Max, 40-core)  = 1.95x
#     GPU cores       80 / 40                              = 2.00x  -> prefill
#
# The two paths disagree by ~35-40%: our measured M5 Max already decodes
# slightly FASTER than M3 Ultra despite 25% less bandwidth (a real generational
# efficiency gain Path A can't see, since it never looks at M5-era silicon).
# Path B is grounded in that same-generation measurement, so it runs faster on
# decode; Path A's compute delta is Apple's stated ceiling, so it runs faster
# on prefill. Central estimate = geometric mean of both; low/high = the two
# paths themselves, not padding. Where no M5 Max run exists for a model
# (DeepSeek wasn't benchmarked on m5-max), only Path A is available and the
# projection is flagged single-anchor.
BW_ULTRA_VS_M3ULTRA = 1200.0 / 800.0    # Apple: "50% higher than M3 Ultra"
COMPUTE_ULTRA_VS_M3ULTRA = 4.5          # Apple: "up to 4.5x peak GPU AI compute vs M3 Ultra"
BW_ULTRA_VS_M5MAX = 1200.0 / 614.0      # 1.2TB/s vs M5 Max's 40-core-GPU config
CORES_ULTRA_VS_M5MAX = 80.0 / 40.0      # GPU core count: the exact fusion ratio


def _m5ultra_scale(mac: dict, m5max: dict | None) -> dict:
    """Dual-anchor decode/prefill speedup factors (M5 Ultra vs M3 Ultra),
    each a geometric mean of Path A and Path B with the paths kept as bounds.

    Path A and Path B are exposed individually (not just sorted lo/hi) because
    which one is which matters for the narrative: Path A (M3 Ultra x Apple's
    official multiplier) assumes decode scales with raw bandwidth, which is
    only true when the *competitor* chip you're comparing against is actually
    bandwidth-saturated on that model. We verify that per cohort by comparing
    the measured M5 Max-vs-M5Max ratio against a reference host to the
    bandwidth ratio that would predict — see bw_saturated_ref below.
    """
    dscale_a, pscale_a = BW_ULTRA_VS_M3ULTRA, COMPUTE_ULTRA_VS_M3ULTRA
    have_b = bool(m5max and m5max.get("cold50_out_tps") and mac.get("cold50_out_tps")
                  and m5max.get("cold50_ttft_ms") and mac.get("cold50_ttft_ms"))
    if have_b:
        m5max_vs_m3u_decode = m5max["cold50_out_tps"] / mac["cold50_out_tps"]
        dscale_b = m5max_vs_m3u_decode * BW_ULTRA_VS_M5MAX
        m5max_vs_m3u_ttft = mac["cold50_ttft_ms"] / m5max["cold50_ttft_ms"]  # >1: M5 Max already faster
        pscale_b = m5max_vs_m3u_ttft * CORES_ULTRA_VS_M5MAX
        dscale_lo, dscale_hi = sorted((dscale_a, dscale_b))
        pscale_lo, pscale_hi = sorted((pscale_a, pscale_b))
        dscale = (dscale_a * dscale_b) ** 0.5
        pscale = (pscale_a * pscale_b) ** 0.5
        basis = "dual-anchor (M3 Ultra measured + M5 Max measured)"
        # M5 Ultra is exactly two M5 Max dies fused, so NO metric may beat the
        # measured M5 Max by more than 2x (bandwidth 1.95x, cores 2x). Cap all
        # decode/prefill speedups at 2x the measured M5 Max to keep the
        # projection physically honest.
        cap_d = (2.0 * m5max["cold50_out_tps"] / mac["cold50_out_tps"]
                 if m5max.get("cold50_out_tps") and mac.get("cold50_out_tps") else None)
        cap_p = (2.0 * m5max["cold50_prompt_tps"] / mac["cold50_prompt_tps"]
                 if m5max.get("cold50_prompt_tps") and mac.get("cold50_prompt_tps") else None)
        if cap_d:
            dscale = min(dscale, cap_d); dscale_lo = min(dscale_lo, cap_d); dscale_hi = min(dscale_hi, cap_d)
            dscale_a = min(dscale_a, cap_d)
            if dscale_b: dscale_b = min(dscale_b, cap_d)
        if cap_p:
            pscale = min(pscale, cap_p); pscale_lo = min(pscale_lo, cap_p); pscale_hi = min(pscale_hi, cap_p)
            pscale_a = min(pscale_a, cap_p)
            if pscale_b: pscale_b = min(pscale_b, cap_p)
    else:
        dscale_b = pscale_b = None
        dscale = dscale_lo = dscale_hi = dscale_a
        pscale = pscale_lo = pscale_hi = pscale_a
        basis = "single-anchor (M3 Ultra only — no M5 Max run for this model)"
    return {"dscale": dscale, "dscale_lo": dscale_lo, "dscale_hi": dscale_hi,
            "pscale": pscale, "pscale_lo": pscale_lo, "pscale_hi": pscale_hi,
            "dscale_a": dscale_a, "dscale_b": dscale_b,
            "pscale_a": pscale_a, "pscale_b": pscale_b,
            "basis": basis}


def _bw_saturation_check(m5max: dict | None, ref: dict | None, ref_bw: float, m5max_bw: float = 614.0) -> dict | None:
    """Is `ref` (e.g. RTX PRO 6000) actually bandwidth-saturated relative to
    M5 Max on this model? Compares the MEASURED decode ratio to the ratio raw
    bandwidth would predict — a ratio near 1.0 means decode tracks bandwidth
    (the competitor is saturated, so scaling M5 Max by a bandwidth multiplier
    is trustworthy); well above 1.0 means the competitor is leaving bandwidth
    unused on this workload (measured M5 Max is closer to it than bandwidth
    alone would suggest), which is exactly when a low-bandwidth chip can close
    the gap by other means (more cores, better utilization) once doubled."""
    if not (m5max and ref and m5max.get("cold50_out_tps") and ref.get("cold50_out_tps")):
        return None
    measured_ratio = m5max["cold50_out_tps"] / ref["cold50_out_tps"]
    bw_predicted_ratio = m5max_bw / ref_bw
    return {
        "measured_ratio": measured_ratio,
        "bw_predicted_ratio": bw_predicted_ratio,
        "saturation_index": measured_ratio / bw_predicted_ratio,  # >>1 = ref underuses its bandwidth here
    }


PROJ_NOTE = ("projected from measured M3 Ultra (macstudio) and, where available, M5 Max "
             "runs, scaled by two independent paths derived from Apple's published M5 "
             "Ultra specs (1.2TB/s bandwidth, 80-core GPU, up to 4.5x peak AI compute vs "
             "M3 Ultra) — central estimate is the geometric mean of both paths; low/high "
             "are the paths themselves, not padding. Not measured.")


def canonical_run_id(scenario: str, host: str) -> str | None:
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
        # on a tie (e.g. two full runs of the same scenario), prefer the most
        # recent run_id (timestamps sort chronologically in the run_id prefix)
        if n > best[0] or (n == best[0] and run["run_id"] > (best[1] or "")):
            best = (n, run["run_id"])
    return best[1]


def _power_series(run_dir):
    """(ts -> summed watts) across every node in a run's metrics trace, plus the
    power key used. None if the run has no power exporter."""
    mrows = read_jsonl(run_dir / "metrics.jsonl")
    keys: set[str] = set()
    for r in mrows:
        m = r.get("metrics")
        if isinstance(m, dict):
            keys.update(m.keys())
    pkey = "total_power_w" if "total_power_w" in keys else ("gpu_power_w" if "gpu_power_w" in keys else None)
    if not pkey:
        return None, None
    series: dict[float, float] = {}
    for r in mrows:
        m = r.get("metrics")
        if not isinstance(m, dict) or m.get(pkey) is None:
            continue
        series[r["ts_epoch"]] = series.get(r["ts_epoch"], 0.0) + m[pkey]
    return (series or None), pkey


def _idle_floor_w(run_dir) -> float | None:
    """Lowest summed whole-box draw seen anywhere in a run's trace = its idle
    floor. Used to catch stale exporter reads (see _guard_single_power)."""
    series, _ = _power_series(run_dir)
    return min(series.values()) if series else None


def _guard_single_power(sample: dict, floor: float | None, adjust_w: float = 0.0) -> float | None:
    """Single-user power over one request window, or None when unmeasurable.

    A request window whose reading sits at the run's idle floor while the box was
    demonstrably busy is a stale exporter read, not an idle GPU: the dcgm
    exporter is known to report the idle value (and 0% util) for the first ~10 s
    of a run, so short requests that open a run — and short requests generally —
    can carry a floor value (e.g. 18 W measured for an RTX PRO 6000 streaming at
    148 tok/s). Dividing through by that yields impossible efficiencies
    (hundreds of tok/s per watt), so those samples are dropped instead of
    plotted, exactly as ANALYSIS.md already excludes short-request power.

    `adjust_w` is the whole-box estimate already ADDED to the sample by
    run_averaging.adjust_power; it is subtracted back off before the comparison
    so a cluster's +70W/node constant can't mask a stale idle reading.
    """
    pw = sample.get("power_w")
    if pw is None:
        return None
    if floor is not None and pw - adjust_w <= floor * 1.05:
        return None
    return pw


def _serving(run_dir) -> dict | None:
    """What actually served this run (server vendor, model id, quant, context
    cap) — from run.json's model_spec. Surfaces e.g. GGUF/llama.cpp vs MLX vs
    vLLM vs sglang so cross-machine model rows are read with their backends."""
    run = read_json(run_dir / "run.json") or {}
    ms = run.get("model_spec") or {}
    raw = ms.get("raw") or {}
    if not ms:
        return None
    return {
        "server": ms.get("owned_by") or "unknown",
        "model": ms.get("model") or "",
        "quant": raw.get("quant"),
        "max_model_len": ms.get("max_model_len") or raw.get("max_model_len") or raw.get("context_length"),
    }


def _batch_power_from_metrics(run_dir, streams, adjust_w: float = 0.0) -> float | None:
    """Average whole-box power over a concurrent batch's full span, from the
    run's metrics time-series (summed across cluster nodes). Per-stream power_w
    is unreliable for short concurrent requests (median over idle queue gaps),
    so this is the honest concurrent power draw."""
    series, pkey = _power_series(run_dir)
    if not series:
        return None
    start = min(s["ts_start_epoch"] for s in streams)
    end = max(s["ts_end_epoch"] for s in streams)
    vals = [v for t, v in series.items() if start <= t <= end]
    if not vals:
        return None
    return float(np.mean(vals)) + adjust_w


def _log_interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Log-log linear interpolation of y at x (x within [min,max])."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo = max(v for v in xs if v <= x)
    hi = min(v for v in xs if v >= x)
    if lo == hi:
        return ys[xs.index(lo)]
    f = (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return float(np.exp(np.log(ys[xs.index(lo)]) + f * (np.log(ys[xs.index(hi)]) - np.log(ys[xs.index(lo)]))))


def build_site_data() -> dict[str, Any]:
    cohorts: dict[str, Any] = {}
    for scenario, smeta in SCENARIO_META.items():
        for host, hmeta in HOST_META.items():
            rid = canonical_run_id(scenario, host)
            if not rid:
                continue
            if scenario in AVG_SCENARIOS:
                samples = averaged_samples(scenario, host)
                conc_run = sorted(complete_run_ids(scenario, host))[-1]  # concurrent data is from the latest run
                rid = "averaged"
            else:
                samples = [adjust_power(s) for s in read_jsonl(RESULTS / scenario / rid / "samples.jsonl") if s.get("ok")]
                conc_run = rid
            run_dir = RESULTS / scenario / conc_run
            # idle floor of THIS run's power trace, for the stale-reading guard
            pw_floor = _idle_floor_w(run_dir)
            pw_adjust = POWER_ADJUST_W.get(host, 0.0)
            serving = _serving(RESULTS / scenario / (rid if rid != "averaged" else conc_run))
            warm = {s["step"]: s for s in samples if s["phase"] == "warm"}
            cold = {s["step"]: s for s in samples if s["phase"] == "cold"}
            mucold = [s for s in samples if s["phase"] == "mucold"]
            muwarm = [s for s in samples if s["phase"] == "muwarm"]

            # Apple/omlx: the observed runs didn't reuse the prefix (server-config
            # shortfall), so REVISE warm stats to assume vLLM-equivalent caching.
            cache_revised = host in rev.APPLE_HOSTS and scenario != "deepseekv4"
            if cache_revised:
                cold_lst = list(cold.values())
                warm = rev.revised_warm_turns(warm, cold_lst)
                muwarm = rev.revised_batch(muwarm, cold_lst)

            # per-turn warm session
            turns = []
            valid = all(s in warm for s in STEPS)
            session_s = gen_s = tokens = float("nan")
            eff_tok_s = gen_pct = float("nan")
            nocache_s = eff_nocache = float("nan")
            speedup = float("nan")
            if valid:
                for st in STEPS:
                    s = warm[st]
                    turns.append({
                        "step": st,
                        "ttft_ms": s["ttft_ms"],
                        "gen_ms": s["tg_ms"],
                        "total_ms": s["total_ms"],
                        "out_tps": s["out_tps"],
                        "prompt_tps": s["prompt_tps"],
                        "power_w": s.get("power_w"),
                    })
                session_s = sum(t["total_ms"] for t in turns) / 1000.0
                gen_s = sum(t["gen_ms"] for t in turns) / 1000.0
                tokens = sum(warm[st]["tg_tokens"] for st in STEPS)
                eff_tok_s = tokens / session_s if session_s else float("nan")
                gen_pct = 100.0 * gen_s / session_s if session_s else float("nan")
                # no-cache baseline: every turn re-prefills its full context
                if len(cold) >= 2:
                    xs = sorted(cold)
                    cys = [cold[x]["ttft_ms"] for x in xs]
                    nc_s = sum((_log_interp(st, xs, cys) / 1000.0) + (warm[st]["tg_ms"] / 1000.0) for st in STEPS)
                    nocache_s = nc_s
                    eff_nocache = tokens / nc_s if nc_s else float("nan")
                    speedup = nocache_s / session_s if session_s else float("nan")

            # team (multi-user warm): wall time per engineer count = sum of batch
            # union spans; plus max CONCURRENT throughput per batch (aligned to
            # each batch's own decode/prefill spans, NOT a whole-run max):
            #   maxTG = max over steps of the overlap-based aggregate decode rate
            #   maxPP = max over steps of (sum prompt_tokens ÷ union prefill span)
            team = {}
            team_max: dict[int, dict[str, float]] = {}
            for (st, users), batch in _group(muwarm, lambda s: (s["step"], s["iter"])).items():
                agg = concurrent_batch_aggregates(batch)
                if agg["wall_s"] and agg["wall_s"] == agg["wall_s"]:
                    team[users] = team.get(users, 0.0) + agg["wall_s"]
                if agg["agg_tps"] and agg["agg_tps"] == agg["agg_tps"]:
                    m = team_max.setdefault(users, {"tg": 0.0, "pp": 0.0})
                    m["tg"] = max(m["tg"], agg["agg_tps"])
                if batch:
                    ps = [s["ts_start_epoch"] for s in batch]
                    pe = [s["ts_start_epoch"] + (s["ttft_ms"] or 0) / 1000.0 for s in batch]
                    union = max(pe) - min(ps)
                    if union > 0:
                        pp = sum(s.get("prompt_tokens") or 0 for s in batch) / union
                        m = team_max.setdefault(users, {"tg": 0.0, "pp": 0.0})
                        m["pp"] = max(m["pp"], pp)
            # per-user team latency (mean per-stream latency across the session)
            team_lat = {}
            for (st, users), batch in _group(muwarm, lambda s: (s["step"], s["iter"])).items():
                lat = float(np.mean([s["total_ms"] for s in batch]))
                team_lat.setdefault(users, []).append(lat)
            team_lat = {u: float(np.mean(v)) for u, v in team_lat.items()}

            # concurrency scaling: multi-user COLD @10k (distinct contexts, so
            # cache-independent — isolates pure concurrency scaling). users 1 = the
            # single-user cold request; 2/4/6 = per-batch overlap-aligned rates.
            conc: dict[int, dict[str, float]] = {}
            for (st, users), batch in _group(mucold, lambda s: (s["step"], s["iter"])).items():
                if st != 10000:
                    continue
                agg = concurrent_batch_aggregates(batch)
                if not (agg["agg_tps"] and agg["agg_tps"] == agg["agg_tps"]):
                    continue
                ps = [s["ts_start_epoch"] for s in batch]
                pe = [s["ts_start_epoch"] + (s["ttft_ms"] or 0) / 1000.0 for s in batch]
                pspan = max(pe) - min(ps)
                pp = sum(s.get("prompt_tokens") or 0 for s in batch) / pspan if pspan > 0 else float("nan")
                lat = float(np.mean([s["total_ms"] for s in batch])) / 1000.0
                pw = _batch_power_from_metrics(RESULTS / scenario / conc_run, batch,
                                               POWER_ADJUST_W.get(host, 0.0))
                conc[users] = {"tg": agg["agg_tps"], "pp": pp, "wall": agg["wall_s"], "lat": lat, "power_w": pw}
            c10 = cold.get(10000, {})
            conc[1] = {"tg": c10.get("out_tps"), "pp": c10.get("prompt_tps"),
                       "wall": (c10.get("total_ms") or 0) / 1000.0,
                       "lat": (c10.get("total_ms") or 0) / 1000.0,
                       "power_w": _guard_single_power(c10, pw_floor, pw_adjust)}

            # single-user reference: cold decode @50k + cold TTFT @50k
            cold50 = cold.get(50000, {})
            cold1 = cold.get(1000, {})
            # cold per-context profile (1k->200k, +400k where measured): total/prefill/gen + rates
            cold_by = {}
            for ctx in (1000, 10000, 50000, 100000, 200000, 400000):
                s = cold.get(ctx)
                if s:
                    pw = _guard_single_power(s, pw_floor, pw_adjust)
                    cold_by[str(ctx)] = {
                        "ttft_ms": s["ttft_ms"], "gen_ms": s["tg_ms"],
                        "total_ms": s["total_ms"], "pp": s["prompt_tps"], "tg": s["out_tps"],
                        "power_w": pw,
                        # ratio of the (possibly averaged) values, not a mean of ratios
                        "tps_per_w": (s["out_tps"] / pw) if (pw and s.get("out_tps")) else None,
                    }

            cohorts[f"{scenario}__{host}"] = {
                "scenario": scenario,
                "arch": smeta["arch"],
                "model_label": smeta["label"],
                "model_short": smeta["short"],
                "arch_color": ARCH_COLOR.get(smeta["arch"], "#555"),
                "host": host,
                "host_display": hmeta["display"],
                "host_class": hmeta["class"],
                "host_color": hmeta["color"],
                "run_id": rid,
                "cache_revised": cache_revised,
                "serving": serving,
                "turns": turns,
                "session_s": session_s,
                "gen_s": gen_s,
                "tokens": tokens,
                "eff_tok_s": eff_tok_s,
                "gen_pct": gen_pct,
                "nocache_s": nocache_s,
                "eff_nocache": eff_nocache,
                "speedup": speedup,
                "team": team,
                "team_max": team_max,
                "team_lat": team_lat,
                "conc": conc,
                "cold50_ttft_ms": cold50.get("ttft_ms"),
                "cold50_out_tps": cold50.get("out_tps"),
                "cold50_prompt_tps": cold50.get("prompt_tps"),
                "cold1_ttft_ms": cold1.get("ttft_ms"),
                "cold_by": cold_by,
            }
    return {
        "steps": STEPS,
        "hosts": HOST_ORDER,
        "host_meta": HOST_META,
        "arch_color": ARCH_COLOR,
        "cohorts": cohorts,
    }


def _group(rows, keyfn):
    out: dict = {}
    for r in rows:
        out.setdefault(keyfn(r), []).append(r)
    return out


def _make_projected_cohorts(cohorts: dict[str, Any]) -> dict[str, Any]:
    """Add clearly-labeled M5 Ultra projection cohorts (dual-anchor central
    estimates — see the BW_ULTRA_VS_* / COMPUTE_ULTRA_VS_* constants above)."""
    for sc in SCENARIO_META:
        mac = cohorts.get(f"{sc}__macstudio")
        if not mac or not mac["turns"]:
            continue
        m5max = cohorts.get(f"{sc}__m5-max")
        tr6000 = cohorts.get(f"{sc}__tr-pro-6000")
        sf = _m5ultra_scale(mac, m5max)
        dscale, pscale = sf["dscale"], sf["pscale"]
        sat = _bw_saturation_check(m5max, tr6000, ref_bw=1792.0)
        speedup = mac.get("speedup", 1.0)
        if speedup != speedup:  # nan guard — no revised/measured caching ratio available
            speedup = 1.0
        turns = []
        for t in mac["turns"]:
            ttft = t["ttft_ms"] / pscale
            gen = t["gen_ms"] / dscale
            turns.append({
                "step": t["step"], "ttft_ms": ttft, "gen_ms": gen,
                "total_ms": ttft + gen,
                "out_tps": t["out_tps"] * dscale if t["out_tps"] else None,
                "prompt_tps": t["prompt_tps"] * pscale if t["prompt_tps"] else None,
                "power_w": None,
            })
        session_s = sum(t["total_ms"] for t in turns) / 1000.0
        gen_s = sum(t["gen_ms"] for t in turns) / 1000.0
        tokens = mac["tokens"]
        eff = tokens / session_s
        gen_pct = 100.0 * gen_s / session_s
        team_lat = {u: (v / 1.3) for u, v in mac["team_lat"].items()} if mac.get("team_lat") else {}
        nocache_s = session_s * speedup
        team_max = {u: {"tg": m["tg"] * dscale, "pp": m["pp"] * pscale}
                    for u, m in (mac.get("team_max") or {}).items()}
        team = {u: v * (session_s / mac["session_s"]) for u, v in (mac.get("team") or {}).items()}
        m5u_conc = {u: {"tg": v["tg"] * dscale, "pp": v["pp"] * pscale,
                        "wall": v["wall"] * (session_s / mac["session_s"]),
                        "lat": v["lat"] * (session_s / mac["session_s"]),
                        "power_w": None}
                    for u, v in (mac.get("conc") or {}).items()}
        cold_by = {ctx: {"ttft_ms": v["ttft_ms"] / pscale,
                         "gen_ms": v["gen_ms"] / dscale,
                         "total_ms": v["ttft_ms"] / pscale + v["gen_ms"] / dscale,
                         "pp": (v["pp"] * pscale if v.get("pp") else None),
                         "tg": (v["tg"] * dscale if v.get("tg") else None),
                         "power_w": None, "tps_per_w": None}
                   for ctx, v in (mac.get("cold_by") or {}).items()}
        cohorts[f"{sc}__m5ultra"] = {
            "scenario": sc, "arch": mac["arch"], "arch_color": mac["arch_color"],
            "model_label": mac["model_label"], "model_short": mac["model_short"],
            "host": "m5ultra", "host_display": HOST_META["m5ultra"]["display"],
            "host_class": "Apple", "host_color": HOST_META["m5ultra"]["color"],
            "run_id": "projected",
            "projected": True, "proj_note": PROJ_NOTE,
            "proj_basis": sf["basis"],
            "proj_range": {
                "decode_lo": mac["cold50_out_tps"] * sf["dscale_lo"] if mac.get("cold50_out_tps") else None,
                "decode_hi": mac["cold50_out_tps"] * sf["dscale_hi"] if mac.get("cold50_out_tps") else None,
                "ttft50_lo_ms": mac["cold50_ttft_ms"] / sf["pscale_hi"] if mac.get("cold50_ttft_ms") else None,
                "ttft50_hi_ms": mac["cold50_ttft_ms"] / sf["pscale_lo"] if mac.get("cold50_ttft_ms") else None,
            },
            # Path A (M3 Ultra x official multiplier) and Path B (M5 Max x
            # physical fusion ratio) exposed individually, by name, so the UI
            # and fine-print copy can say which path gives which number rather
            # than just a generic "range" — the two paths can disagree about
            # whether M5 Ultra beats a given competitor, not just by how much.
            "proj_paths": {
                "decode_path_a": (mac["cold50_out_tps"] * sf["dscale_a"]) if mac.get("cold50_out_tps") else None,
                "decode_path_b": (mac["cold50_out_tps"] * sf["dscale_b"]) if (mac.get("cold50_out_tps") and sf["dscale_b"]) else None,
                "ttft50_path_a_ms": (mac["cold50_ttft_ms"] / sf["pscale_a"]) if mac.get("cold50_ttft_ms") else None,
                "ttft50_path_b_ms": (mac["cold50_ttft_ms"] / sf["pscale_b"]) if (mac.get("cold50_ttft_ms") and sf["pscale_b"]) else None,
            },
            # Is the reference GPU (RTX PRO 6000) actually bandwidth-saturated
            # on this model? saturation_index >> 1 means no — M5 Max already
            # gets closer to it than raw bandwidth would predict, which is
            # exactly when Path B (same-generation, cores-based) is the more
            # trustworthy path and Path A (bandwidth-only) understates M5 Ultra.
            "bw_saturation_vs_tr6000": sat,
            "turns": turns, "session_s": session_s, "gen_s": gen_s, "tokens": tokens,
            "eff_tok_s": eff, "gen_pct": gen_pct,
            "nocache_s": nocache_s, "eff_nocache": tokens / nocache_s, "speedup": speedup,
            "team": team, "team_max": team_max, "team_lat": team_lat,
            "conc": m5u_conc,
            "cold50_ttft_ms": mac["cold50_ttft_ms"] / pscale if mac.get("cold50_ttft_ms") else None,
            "cold50_out_tps": mac["cold50_out_tps"] * dscale if mac.get("cold50_out_tps") else None,
            "cold50_prompt_tps": (mac["cold50_prompt_tps"] * pscale) if mac.get("cold50_prompt_tps") else None,
            "cold_by": cold_by,
            "cold1_ttft_ms": (mac["cold1_ttft_ms"] / pscale) if mac.get("cold1_ttft_ms") else None,
        }
    return cohorts


def main() -> None:
    data = build_site_data()
    _make_projected_cohorts(data["cohorts"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.SITE_DATA = " + json.dumps(data, default=str) + ";")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB) with {len(data['cohorts'])} cohorts")
    for k, v in sorted(data["cohorts"].items()):
        sc, h = k.split("__")
        tag = " (PROJ)" if v.get("projected") else ""
        s = v["session_s"]
        print(f"  {v['model_short']:<20} {v['host_display']:<22} "
              f"session={s:7.1f}s eff={v['eff_tok_s']:6.1f} tok/s "
              f"decode50={v['cold50_out_tps']:6.1f} ttft50={v['cold50_ttft_ms']:8,.0f}ms{tag}")


if __name__ == "__main__":
    main()
