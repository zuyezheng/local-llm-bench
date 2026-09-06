"""Durable artifacts (JSON + CSV + charts) and cross-host report rendering.

Chart design notes (see README "What it measures" for the metric defs):

* No twin/dual y-axes anywhere. Matplotlib autoscales each y-axis
  independently to fill the plot, so two curves on separate axes will
  visually "mirror" or "align" almost regardless of whether they're
  actually related — that's an artifact of the chart type, not a signal.
  TTFT/PP/TG each get their own honestly, independently scaled panel
  instead. TPOT is dropped from charts entirely (TG = 1000/TPOT exactly, on
  the same x-axis — plotting both is plotting the same number twice; TPOT
  stays in the CSV/markdown table for anyone who wants ms/token directly).
* Prefill (TTFT/PP) is compute-bound and spans orders of magnitude across
  context lengths, so it gets log-log axes. Decode (TG) is bandwidth-bound
  and roughly flat, so it gets a zero-based linear axis — putting it on the
  same tight log scale as prefill would make small, meaningful differences
  look enormous or invisible depending on which way the auto-scale falls.
* Concurrency/multi-user impact on TG and PP is shown on ONE shared axis —
  see ``_concurrency_impact_chart``. This doesn't need normalizing: TG
  (aggregate tok/s) and PP (mean per-stream tok/s) are already the same
  unit, so both get plotted as real, unmodified numbers.
* Small multiples (one panel per context or per user-count) use ONE legend
  dimension (host/color) instead of encoding host+context in a single
  legend, which produces an unreadable wall of near-duplicate colors.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import hw_specs
from . import metrics as met
from .benchmark import HostOutcome, _batch_aggregates
from .config import Config
from .results import ResultsStore

CYCLE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0e7490", "#be185d", "#65a30d"]

# preference order for picking the ONE representative series out of several
# same-suffix metrics (e.g. dcgm exposes gpu/mem/enc/dec utilization —
# only the first one is worth a full panel). Most-relevant first.
UTIL_KEY_PRIORITY = ("gpu_util_pct", "cpu_util_pct", "gpu_active_pct", "cpu_active_pct")
TEMP_KEY_PRIORITY = ("gpu_temp_c", "cpu_temp_c", "gpu_mem_temp_c")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def _pick_key(available: set[str], priority: tuple[str, ...]) -> str | None:
    for k in priority:
        if k in available:
            return k
    return next(iter(sorted(available)), None)


def _median_series(samples: list[dict], key: str, steps: list[int]) -> list[float]:
    return [_percentile([s[key] for s in samples if s["step"] == st and s.get(key) is not None], 50)
            for st in steps]


# ------------------------------------------------------------------ write

def write_run_artifact(
    outcome: HostOutcome,
    cfg: Config,
    store: ResultsStore,
    root: str | Path,
) -> Path:
    """Write derived artifacts (charts + csv) INTO the run's results directory.

    ``root`` is the run directory (``results/<scenario>/<run_id>/``) that the
    store already created; sample/metric rows come from its own files, so this
    touches nothing but this run. Returns the run directory.
    """
    run_dir = Path(root)
    charts_dir = run_dir / "charts"
    run_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    # charts are fully derived/regenerable — clear stale ones first so a
    # chart the current code decides NOT to draw (e.g. no baseline data
    # available) can never linger under its old filename from a prior
    # version of the chart-selection logic.
    for p in charts_dir.glob("*.png"):
        p.unlink()

    samples = store.samples(outcome.run_id)
    _write_csv(samples, run_dir / "samples.csv")
    _plot_run(samples, charts_dir, outcome.host.name, store, outcome.run_id)

    metric_rows = store.metric_samples(outcome.run_id)
    if metric_rows:
        _write_metrics_csv(metric_rows, run_dir / "metrics.csv")
        _plot_metrics_timeseries(metric_rows, samples, charts_dir / "metrics_timeseries.png",
                                 outcome.host.name)

    prior = store.run(outcome.run_id) or {}
    merged: dict[str, Any] = dict(prior)
    merged.update({
        "meta": {
            "tool": "llm-bench",
            "version": "0.1.0",
            "generated_at": _utc(),
            "elapsed_s": round(outcome.elapsed_s, 2),
        },
        "machine_spec": outcome.machine_spec or prior.get("machine_spec", {}),
        "model_spec": outcome.model_spec or prior.get("model_spec", {}),
        "config": prior.get("config") or {
            "scenario": cfg.name,
            "host": outcome.host.name,
            "url": outcome.host.url,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "context_lengths": cfg.context_lengths,
            "warm_sizes": cfg.warm_sizes,
            "cold_iters": cfg.cold_iters,
            "warm_sessions": cfg.warm_sessions,
            "multi_users": cfg.multi_users,
            "encoding": cfg.encoding,
            "corpus_kind": cfg.corpus_kind,
            "code_ratio": cfg.code_ratio,
            "seed": cfg.seed,
        },
        "results": outcome.summary or {},
        "charts": sorted(p.name for p in charts_dir.iterdir()),
    })
    (run_dir / "run.json").write_text(json.dumps(merged, indent=2))
    return run_dir


def _write_csv(samples: list[dict], path: Path) -> None:
    if not samples:
        return
    cols = [
        "run_id", "host", "model", "phase", "step", "iter", "session_pos",
        "context_tokens", "prompt_tokens", "new_prompt_tokens", "max_tokens",
        "ttft_ms", "prompt_tps", "tpot_ms", "tg_ms", "tg_tokens",
        "reasoning_tokens", "content_tokens", "tg_s",
        "out_tps", "power_w", "wh", "tps_per_w",
        "total_ms", "ok", "error", "finish_reason",
        "batch", "ts_start_epoch", "ts_end_epoch",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(samples)


def _write_metrics_csv(metric_rows: list[dict], path: Path) -> None:
    import json as _json

    rows = []
    for r in metric_rows:
        raw = r["metrics"]
        if isinstance(raw, str):
            try:
                raw = _json.loads(raw)
            except (TypeError, _json.JSONDecodeError):
                continue
        if not isinstance(raw, dict):
            continue
        rows.append({"ts_epoch": r["ts_epoch"], "node": r.get("node", ""), **raw})
    if not rows:
        return
    cols = ["ts_epoch", "node"] + sorted({k for r in rows for k in r if k not in ("ts_epoch", "node")})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------- shared plots

def _metric_panel(
    ax, series: dict[str, list[dict]], key: str, ylabel: str,
    logy: bool = True, legend: bool = False,
) -> None:
    """One panel, one metric, one line per host — a real, independently-scaled
    axis with a real per-host legend. No twin/dual axis: matplotlib
    autoscales each y-axis independently to fill the plot, so two curves on
    separate axes will visually "mirror" or "align" almost regardless of
    whether they're actually related — that's an artifact of the chart type,
    not a signal, so it's not used here."""
    for i, name in enumerate(sorted(series)):
        samples = series[name]
        steps = sorted({s["step"] for s in samples})
        if not steps:
            continue
        color = CYCLE[i % len(CYCLE)]
        ax.plot(steps, _median_series(samples, key, steps), "o-", color=color, label=name)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("context tokens")
    ax.grid(True, which="both", alpha=0.3)
    if legend:
        ax.legend(fontsize=8)


def _prefill_decode_chart(series: dict[str, list[dict]], path: Path, title: str) -> None:
    """Three independently-scaled panels: TTFT, PP, TG.

    TPOT is deliberately NOT plotted here: TG = 1000 / TPOT exactly, on the
    same x-axis, with no other variable involved — showing both would be
    showing the same number twice (TPOT is still in the CSV/markdown table
    for anyone who wants ms/token directly). TTFT and PP, despite being
    related (PP = prompt_tokens / TTFT), are NOT redundant: prompt_tokens
    varies with context length in a way that reshapes the curve (fixed
    per-request overhead at small contexts, possible falloff at large ones)
    — a trend TTFT's own monotonic growth mostly hides. So both stay, each
    on its own honestly-scaled axis rather than overlaid.
    """
    if not any(series.values()):
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle(title, fontsize=13)
    _metric_panel(axes[0], series, "ttft_ms", "TTFT (ms) — wait time", logy=True, legend=True)
    axes[0].set_title("prefill latency", fontsize=10)
    _metric_panel(axes[1], series, "prompt_tps", "PP (tok/s) — compute rate", logy=True)
    axes[1].set_title("prefill throughput (normalizes for context size)", fontsize=10)
    _metric_panel(axes[2], series, "out_tps", "TG (tok/s)", logy=False)
    axes[2].set_title("decode throughput — bandwidth-bound", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _cache_effect_chart(cold: dict[str, list[dict]], warm: dict[str, list[dict]], path: Path,
                        title: str) -> None:
    """Cold vs warm prompt-processing throughput at matching context sizes —
    the direct answer to "does prefix caching help on this hardware".
    Warm's prompt_tps only counts NEW (uncached) tokens per step."""
    hosts = sorted(set(cold) | set(warm))
    if not hosts:
        return
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.suptitle(title, fontsize=13)
    for i, h in enumerate(hosts):
        color = CYCLE[i % len(CYCLE)]
        for samples, ls, alpha, tag in ((cold.get(h, []), "o-", 0.95, "cold"),
                                         (warm.get(h, []), "s--", 0.55, "warm")):
            if not samples:
                continue
            steps = sorted({s["step"] for s in samples})
            ax.plot(steps, _median_series(samples, "prompt_tps", steps), ls, color=color,
                    alpha=alpha, label=f"{h} ({tag})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("context tokens")
    ax.set_ylabel("prompt processing (tok/s) — warm counts only new tokens")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _concurrency_impact_chart(
    summaries: dict[str, dict[str, dict]], path: Path, title: str, panel_label: str,
) -> bool:
    """Multi-user impact on token generation AND prompt processing, together,
    on ONE real axis — tokens/second. No normalization needed here: TG and
    PP are already the same unit (both tok/s), so plotting the actual
    measured numbers on a shared axis is honest, not a twin-axis trick.

    * TG = aggregate generation throughput across the whole concurrent batch
      (solid circles) — the real, total tok/s this host delivers at N
      concurrent streams.
    * PP = mean prompt-processing rate PER STREAM, not summed (dotted
      triangles) — concurrent prefill isn't naturally additive across
      backends the way generation is, so this is each individual stream's
      own prefill speed, averaged across the batch.

    Small multiples: one panel per context/step. Same host = same color for
    both lines; marker+linestyle distinguishes the metric (see legend).
    """
    hosts = sorted(summaries)
    pairs = sorted({tuple(int(x) for x in k.split(":")) for s in summaries.values() for k in s})
    if not pairs:
        return False
    steps = sorted({p[0] for p in pairs})
    ncols = min(3, len(steps))
    nrows = -(-len(steps) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.6 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=13)
    any_panel = False
    for idx, step in enumerate(steps):
        ax = axes[idx // ncols][idx % ncols]
        plotted = False
        for i, h in enumerate(hosts):
            rows = sorted(
                (it, v) for k, v in summaries[h].items()
                for st, it in [tuple(int(x) for x in k.split(":"))]
                if st == step
            )
            if not rows:
                continue
            color = CYCLE[i % len(CYCLE)]
            ax.plot([it for it, _ in rows], [v["agg_tps"] for _, v in rows], "o-", color=color)
            pp_pts = [(it, v["mean_prompt_tps"]) for it, v in rows if v.get("mean_prompt_tps")]
            if pp_pts:
                ax.plot([p[0] for p in pp_pts], [p[1] for p in pp_pts], "^:", color=color)
            plotted = True
        if not plotted:
            ax.axis("off")
            continue
        any_panel = True
        ax.set_title(f"{panel_label} = {step:,}", fontsize=10)
        ax.set_xlabel("concurrent streams")
        ax.set_ylabel("tokens / second")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            host_handles = [Line2D([0], [0], color=CYCLE[i % len(CYCLE)], marker="o", ls="-")
                            for i in range(len(hosts))]
            leg1 = ax.legend(host_handles, hosts, fontsize=7, loc="upper left", title="host")
            ax.add_artist(leg1)
            metric_handles = [
                Line2D([0], [0], color="#374151", marker="o", ls="-"),
                Line2D([0], [0], color="#374151", marker="^", ls=":"),
            ]
            metric_labels = ["TG — aggregate tok/s", "PP — mean per-stream tok/s"]
            ax.legend(metric_handles, metric_labels, fontsize=6.5, loc="upper right")
    if not any_panel:
        plt.close(fig)
        return False
    for idx in range(len(steps), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _bandwidth_chart(
    cold: dict[str, list[dict]], chip_by_host: dict[str, str | None], path: Path, title: str,
) -> None:
    """Raw compute vs. memory-bandwidth comparison: x=prompt processing
    (compute-bound prefill), y=generation throughput (bandwidth-bound decode),
    one path per host across context lengths. A host up-and-to-the-right wins
    on both axes; a host far right but low is compute-strong/bandwidth-weak
    and vice versa. Peak bandwidth (vendor-published, approximate) is
    annotated per host for reference — see bench/hw_specs.py."""
    hosts = sorted(cold)
    if not hosts:
        return
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    fig.suptitle(title, fontsize=13)
    for i, h in enumerate(hosts):
        samples = cold[h]
        steps = sorted({s["step"] for s in samples})
        if not steps:
            continue
        xs = _median_series(samples, "prompt_tps", steps)
        ys = _median_series(samples, "out_tps", steps)
        color = CYCLE[i % len(CYCLE)]
        ax.plot(xs, ys, "o-", color=color, ms=6, label=None)
        chip = chip_by_host.get(h)
        bw = hw_specs.peak_bandwidth_gbps(chip)
        tag = f"{h}\n{chip}" if chip else h
        if bw:
            tag += f"\n(peak {bw:,.0f} GB/s)"
        ax.annotate(tag, (xs[-1], ys[-1]), fontsize=7, color=color,
                    xytext=(6, 0), textcoords="offset points", va="center")
        ax.plot([], [], "o-", color=color, label=h)  # legend proxy
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prompt processing (tok/s) — compute-bound")
    ax.set_ylabel("generation throughput (tok/s) — bandwidth-bound")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _efficiency_bar_chart(
    eff_by_host: dict[str, dict[str, dict]], path: Path, title: str, phase: str = "cold",
) -> None:
    """Grouped bars: tok/s/W and J/token per host, per context — the
    screenshot-friendly power/efficiency comparison (vs. the raw power trace,
    which answers a different question)."""
    hosts = sorted(h for h in eff_by_host if any(k.startswith(f"{phase}:") for k in eff_by_host[h]))
    if not hosts:
        return
    contexts = sorted({int(k.split(":")[1]) for h in hosts for k in eff_by_host[h] if k.startswith(f"{phase}:")})
    if not contexts:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.8 / max(len(hosts), 1)
    x = np.arange(len(contexts))
    for i, h in enumerate(hosts):
        tpw = [eff_by_host[h].get(f"{phase}:{c}", {}).get("tps_per_w", 0.0) for c in contexts]
        jpt = [eff_by_host[h].get(f"{phase}:{c}", {}).get("j_per_token", 0.0) for c in contexts]
        offs = x + (i - (len(hosts) - 1) / 2) * width
        axes[0].bar(offs, tpw, width=width * 0.9, color=CYCLE[i % len(CYCLE)], label=h)
        axes[1].bar(offs, jpt, width=width * 0.9, color=CYCLE[i % len(CYCLE)], label=h)
    axes[0].set_ylabel("tokens/s per watt")
    axes[0].set_title("Generation efficiency", fontsize=10)
    axes[1].set_ylabel("joules / token")
    axes[1].set_title("Energy per token", fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:,}" for c in contexts])
        ax.set_xlabel(f"context tokens ({phase} cache)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------- per-run

def _plot_run(samples: list[dict], out_dir: Path, host: str, store: ResultsStore, run_id: str) -> None:
    cold = [s for s in samples if s["phase"] == "cold" and s["ok"]]
    warm = [s for s in samples if s["phase"] == "warm" and s["ok"]]

    if cold:
        _prefill_decode_chart({host: cold}, out_dir / "cold_perf.png",
                              f"{host} — cold cache (fresh prompt per request)")
    if warm:
        _prefill_decode_chart({host: warm}, out_dir / "warm_session.png",
                              f"{host} — warm cache, progressive coding session (shared prefix)")
    if cold and warm:
        _cache_effect_chart({host: cold}, {host: warm}, out_dir / "cache_effect.png",
                            f"{host} — cold vs warm cache")

    eff = met.compute_run_efficiency(store, run_id)
    if eff:
        _efficiency_bar_chart({host: eff}, out_dir / "efficiency.png",
                              f"{host} — power efficiency (cold cache)", phase="cold")

    conc_summary = {host: _batch_aggregates(store, run_id, ("conc",))}
    if conc_summary[host]:
        _concurrency_impact_chart(conc_summary, out_dir / "concurrency.png",
                                  f"{host} — concurrency sweep: TG & PP", panel_label="context")
    mucold_summary = {host: _batch_aggregates(store, run_id, ("mucold",))}
    if mucold_summary[host]:
        _concurrency_impact_chart(mucold_summary, out_dir / "multiuser_cold.png",
                                  f"{host} — multi-user cold: TG & PP", panel_label="context")
    muwarm_summary = {host: _batch_aggregates(store, run_id, ("muwarm",))}
    if muwarm_summary[host]:
        _concurrency_impact_chart(muwarm_summary, out_dir / "multiuser_warm.png",
                                  f"{host} — multi-user warm: TG & PP", panel_label="context")


def _plot_metrics_timeseries(
    metric_rows: list[dict], samples: list[dict], path: Path, host: str
) -> None:
    """Per-node power/utilization/temperature traces + summed cluster power."""
    from .metrics import build_power_series, cluster_value_for_window

    series = build_power_series(metric_rows)
    if not series:
        return
    ts = [t for t, _, _ in series]
    tmax = max(ts) if ts else 1.0
    nodes = sorted({n for _, n, _ in series})
    keys = {k for _, _, r in series for k in r}

    def per_node(key: str) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for t, node, red in series:
            if key in red:
                out.setdefault(node, []).append((t, float(red[key])))
        return out

    def short(n: str) -> str:
        return n.split("//")[-1]

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(f"{host} — hardware metrics during run (per node + summed)", fontsize=13)

    ok = [s for s in samples if s["ok"]]
    for s in ok:
        color = "#1d4ed8" if s["phase"] == "cold" else ("#b45309" if s["phase"] == "warm" else "#047857")
        for ax in axes:
            ax.axvspan(s["ts_start_epoch"], s["ts_end_epoch"], alpha=0.08, color=color)

    # power: per-node lines + summed cluster total
    power_keys = [k for k in keys if k.endswith("_w") or k.endswith("_ws")]
    pkey = next((k for k in ("total_power_w", "all_power_w", "gpu_power_w") if k in power_keys),
                power_keys[0] if power_keys else None)
    if pkey:
        summed = [
            (t, cluster_value_for_window(series, t - 1, t + 1, pkey))
            for t in ts
        ]
        axes[0].plot([x[0] for x in summed], [x[1] for x in summed], "-",
                     color="#111827", lw=2.0, label=f"{short(pkey)} (cluster sum)")
    for i, node in enumerate(nodes):
        data = per_node(pkey) if pkey else {}
        pts = data.get(node, [])
        if pts:
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=2.5,
                         color=CYCLE[i % len(CYCLE)], label=f"{short(node)} {short(pkey)}")
    axes[0].set_ylabel("power (W)")
    axes[0].legend(fontsize=7, ncol=2)

    # utilization/temperature: pick the most relevant series present, not just
    # the first one alphabetically (a mostly-unused engine like video
    # decode/encode would otherwise crowd out the actual compute utilization).
    util_key = _pick_key({k for k in keys if k.endswith("_util_pct")}, UTIL_KEY_PRIORITY)
    for i, node in enumerate(nodes):
        data = per_node(util_key) if util_key else {}
        pts = data.get(node, [])
        if pts:
            axes[1].plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=2.5,
                         color=CYCLE[i % len(CYCLE)], label=short(node))
    axes[1].set_ylabel(f"utilization (%) — {short(util_key) if util_key else 'n/a'}")
    axes[1].legend(fontsize=7, ncol=2)

    temp_key = _pick_key({k for k in keys if k.endswith("_temp_c")}, TEMP_KEY_PRIORITY)
    for i, node in enumerate(nodes):
        data = per_node(temp_key) if temp_key else {}
        pts = data.get(node, [])
        if pts:
            axes[2].plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=2.5,
                         color=CYCLE[i % len(CYCLE)], label=short(node))
    axes[2].set_ylabel(f"temperature (°C) — {short(temp_key) if temp_key else 'n/a'}")
    axes[2].set_xlabel("seconds since run start")
    axes[2].legend(fontsize=7, ncol=2)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, tmax)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------- cross-host report

def build_report(store: ResultsStore, cfg: Config, root: str | Path) -> Path:
    """Aggregate across runs WITHIN one scenario (post-processing phase) and
    render comparison charts + a markdown summary under ``root/report``.

    ``store`` must already be scoped to a single scenario (``ResultsStore(...,
    scenario=...)``) — a scenario bundles one model + its hardware sweep
    (README: "self-contained run file"), which is the only unit where
    comparing hosts side by side is apples-to-apples. Comparing across
    scenarios would silently mix different models onto one chart.
    """
    out_dir = Path(root) / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    # see write_run_artifact: clear stale derived charts before regenerating.
    for p in out_dir.glob("*.png"):
        p.unlink()
    runs = store.runs()

    samples = store.samples()
    if not samples:
        return out_dir
    cold = [s for s in samples if s["phase"] == "cold" and s["ok"]]
    warm = [s for s in samples if s["phase"] == "warm" and s["ok"]]
    conc = [s for s in samples if s["phase"] == "conc" and s["ok"]]
    mucold = [s for s in samples if s["phase"] == "mucold" and s["ok"]]
    muwarm = [s for s in samples if s["phase"] == "muwarm" and s["ok"]]

    hosts = sorted({s["host"] for s in samples})
    cold_by_host = {h: [s for s in cold if s["host"] == h] for h in hosts}
    warm_by_host = {h: [s for s in warm if s["host"] == h] for h in hosts}
    chip_by_host = {r["host"]: hw_specs.chip_label(r.get("machine_spec")) for r in runs}
    bw_by_host = {h: hw_specs.peak_bandwidth_gbps(chip_by_host.get(h)) for h in hosts}

    efficiency = met.compute_efficiency_all(store)
    eff_by_host = {h: (efficiency.get(_latest_run_for_host(runs, h)) or {}).get("efficiency", {})
                   for h in hosts}

    def summary_by_host(phases: tuple[str, ...]) -> dict[str, dict[str, dict]]:
        out: dict[str, dict[str, dict]] = {}
        for h in hosts:
            rid = _latest_run_for_host(runs, h)
            if not rid:
                continue
            s = _batch_aggregates(store, rid, phases)
            if s:
                out[h] = s
        return out

    conc_by_host = summary_by_host(("conc",))
    mucold_by_host = summary_by_host(("mucold",))
    muwarm_by_host = summary_by_host(("muwarm",))

    charts_made: list[str] = []
    if cold:
        _prefill_decode_chart(cold_by_host, out_dir / "report_cold.png",
                              "Cold cache — cross-host comparison")
        charts_made.append("report_cold.png")
        _bandwidth_chart(cold_by_host, chip_by_host, out_dir / "report_bandwidth.png",
                         "Raw compute vs. memory bandwidth (cold cache)")
        charts_made.append("report_bandwidth.png")
    if warm:
        _prefill_decode_chart(warm_by_host, out_dir / "report_warm.png",
                              "Warm cache — progressive coding session, cross-host")
        charts_made.append("report_warm.png")
    if cold and warm:
        _cache_effect_chart(cold_by_host, warm_by_host, out_dir / "report_cache_effect.png",
                            "Cold vs warm cache — cross-host")
        charts_made.append("report_cache_effect.png")
    if eff_by_host:
        _efficiency_bar_chart(eff_by_host, out_dir / "report_efficiency.png",
                              "Power efficiency — cold cache, cross-host", phase="cold")
        charts_made.append("report_efficiency.png")
    if conc_by_host:
        if _concurrency_impact_chart(conc_by_host, out_dir / "report_conc.png",
                                     "Concurrency sweep — TG & PP, cross-host", panel_label="context"):
            charts_made.append("report_conc.png")
    if mucold_by_host:
        if _concurrency_impact_chart(mucold_by_host, out_dir / "report_multiuser_cold.png",
                                     "Multi-user cold — TG & PP, cross-host", panel_label="context"):
            charts_made.append("report_multiuser_cold.png")
    if muwarm_by_host:
        if _concurrency_impact_chart(muwarm_by_host, out_dir / "report_multiuser_warm.png",
                                     "Multi-user warm — TG & PP, cross-host", panel_label="context"):
            charts_made.append("report_multiuser_warm.png")

    conc_summary = {r["run_id"]: _batch_aggregates(store, r["run_id"], ("conc",)) for r in runs}
    mu_cold_summary = {r["run_id"]: _batch_aggregates(store, r["run_id"], ("mucold",)) for r in runs}
    mu_warm_summary = {r["run_id"]: _batch_aggregates(store, r["run_id"], ("muwarm",)) for r in runs}

    md = _markdown_report(runs, cold, warm, hosts, efficiency, conc_summary,
                          mu_cold_summary, mu_warm_summary, chip_by_host, bw_by_host, charts_made)
    (out_dir / "report.md").write_text(md)
    return out_dir


def _latest_run_for_host(runs: list[dict], host: str) -> str | None:
    for r in reversed(runs):
        if r["host"] == host:
            return r["run_id"]
    return None


def _markdown_report(runs, cold, warm, hosts, efficiency, conc_summary,
                     mu_cold_summary, mu_warm_summary, chip_by_host=None,
                     bw_by_host=None, charts_made=None) -> str:
    chip_by_host = chip_by_host or {}
    bw_by_host = bw_by_host or {}
    lines = ["# llm-bench report", "", f"_generated {_utc()}_", ""]

    if charts_made:
        lines.append("## Charts")
        lines.append("")
        for c in charts_made:
            lines.append(f"- `{c}`")
        lines.append("")

    lines.append("## Hardware reference")
    lines.append("")
    lines.append("_Peak memory bandwidth is vendor-published (approximate, not independently "
                 "verified) — see `bench/hw_specs.py`. Used only as a reference annotation on "
                 "`report_bandwidth.png`, never to compute a \"% of peak\" figure, since that "
                 "would also require the model's active-parameter count and bit-width._")
    lines.append("")
    lines.append("| host | chip / GPU | peak mem bandwidth (GB/s) |")
    lines.append("|---|---|---|")
    for h in hosts:
        bw = bw_by_host.get(h)
        lines.append(f"| {h} | {chip_by_host.get(h) or '—'} | {f'{bw:,.0f}' if bw else '—'} |")
    lines.append("")

    lines.append("## Runs")
    lines.append("")
    lines.append("| run | host | model | status | samples | power key |")
    lines.append("|---|---|---|---|---|---|")
    for r in runs:
        n = len([s for s in cold + warm if s["run_id"] == r["run_id"]])
        pk = ""
        eff = efficiency.get(r["run_id"])
        if eff and eff["efficiency"]:
            pk = next(iter(eff["efficiency"].values())).get("power_key", "")
        lines.append(f"| {r['run_id']} | {r['host']} | {r['model']} | {r['status']} | {n} | {pk} |")
    lines.append("")
    if cold:
        lines.append("## Cold cache (median)")
        lines.append("")
        lines.append("| host | context | TTFT (ms) | PP (tok/s) | TPOT (ms) | TG (tok/s) | power (W) | tok/s/W |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for ctx in sorted({s["step"] for s in cold}):
            for h in hosts:
                vals = [s for s in cold if s["host"] == h and s["step"] == ctx]
                if not vals:
                    continue
                f = lambda k: f"{_percentile([s[k] for s in vals], 50):,.1f}"
                rid = _latest_run_for_host(runs, h)
                eff = efficiency.get(rid, {}).get("efficiency", {}).get(f"cold:{ctx}")
                pw = f"{eff['avg_power_w']:,.1f}" if eff else "—"
                tpw = f"{eff['tps_per_w']:,.2f}" if eff else "—"
                lines.append(
                    f"| {h} | {ctx:,} | {f('ttft_ms')} | {f('prompt_tps')} | {f('tpot_ms')} | {f('out_tps')} | {pw} | {tpw} |"
                )
        lines.append("")
    if warm:
        lines.append("## Warm cache — progressive session (median across sessions)")
        lines.append("")
        lines.append("| host | context | TTFT (ms) | PP new-tok (tok/s) | TG (tok/s) | power (W) | tok/s/W |")
        lines.append("|---|---|---|---|---|---|---|")
        for ctx in sorted({s["step"] for s in warm}):
            for h in hosts:
                vals = [s for s in warm if s["host"] == h and s["step"] == ctx]
                if not vals:
                    continue
                rid = _latest_run_for_host(runs, h)
                eff = efficiency.get(rid, {}).get("efficiency", {}).get(f"warm:{ctx}")
                pw = f"{eff['avg_power_w']:,.1f}" if eff else "—"
                tpw = f"{eff['tps_per_w']:,.2f}" if eff else "—"
                lines.append(
                    f"| {h} | {ctx:,} | {_percentile([s['ttft_ms'] for s in vals], 50):,.1f} "
                    f"| {_percentile([s['prompt_tps'] for s in vals], 50):,.1f} "
                    f"| {_percentile([s['out_tps'] for s in vals], 50):,.1f} | {pw} | {tpw} |"
                )
        lines.append("")

    # concurrency sweep: aggregate throughput + speedup + power at each concurrency
    conc_rows = []
    for r in runs:
        cs = conc_summary.get(r["run_id"], {})
        for k, v in cs.items():
            ctx, conc = k.split(":")
            conc_rows.append((r["host"], int(ctx), int(conc), v))
    if conc_rows:
        lines.append("## Concurrency sweep (aggregate generation, median across batches)")
        lines.append("")
        lines.append("| host | context | conc | agg TG (tok/s) | per-stream TG (tok/s) | speedup | mean TTFT (ms) | mean lat (ms) | power (W) | tok/s/W |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for h in hosts:
            rows = sorted([x for x in conc_rows if x[0] == h], key=lambda x: (x[1], x[2]))
            for _h, ctx, conc, v in rows:
                rid = _latest_run_for_host(runs, h)
                eff = efficiency.get(rid, {}).get("efficiency", {}).get(f"conc:{ctx}:{conc}")
                pw = f"{eff['avg_power_w']:,.1f}" if eff else "—"
                tpw = f"{eff['tps_per_w']:,.2f}" if eff else "—"
                base = next((x[3]["agg_tps"] for x in rows if x[1] == ctx and x[2] == 1), None)
                speed = f"{v['agg_tps'] / base:,.2f}x" if base else "—"
                lines.append(
                    f"| {h} | {ctx:,} | {conc} | {v['agg_tps']:,.1f} | {v['agg_tps'] / conc:,.1f} | {speed} | "
                    f"{v['mean_ttft_ms']:,.0f} | {v['mean_latency_ms']:,.0f} | {pw} | {tpw} |"
                )
        lines.append("")

    # ------------------------------------------------------ multi-user report
    if any(mu_cold_summary.values()) or any(mu_warm_summary.values()):
        lines.append("## Multi-user (distinct context per user)")
        lines.append("")
        lines.append("_Each user gets a different corpus slice; contexts are never shared._")
        lines.append("")
        if any(mu_cold_summary.values()):
            lines.append("### Multi-user cold")
            lines.append("")
            lines.append("| host | users | context | agg TG (tok/s) | vs single-user | mean TTFT (ms) | mean lat (ms) | power (W) | tok/s/W |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for h in hosts:
                rid = _latest_run_for_host(runs, h)
                cs = mu_cold_summary.get(rid, {})
                rows = sorted((int(k.split(":")[1]), int(k.split(":")[0]), v) for k, v in cs.items())
                for users, ctx, v in rows:
                    eff = efficiency.get(rid, {}).get("efficiency", {}).get(f"mucold:{ctx}:{users}")
                    pw = f"{eff['avg_power_w']:,.1f}" if eff else "—"
                    tpw = f"{eff['tps_per_w']:,.2f}" if eff else "—"
                    single = _percentile([s["out_tps"] for s in cold if s["host"] == h and s["step"] == ctx], 50)
                    vs = f"{v['agg_tps'] / single:,.2f}x" if single else "—"
                    lines.append(
                        f"| {h} | {users} | {ctx:,} | {v['agg_tps']:,.1f} | {vs} | "
                        f"{v['mean_ttft_ms']:,.0f} | {v['mean_latency_ms']:,.0f} | {pw} | {tpw} |"
                    )
            lines.append("")
        if any(mu_warm_summary.values()):
            lines.append("### Multi-user warm (per-user progressive sessions)")
            lines.append("")
            lines.append("| host | users | step | agg TG (tok/s) | mean TTFT (ms) | mean lat (ms) | power (W) |")
            lines.append("|---|---|---|---|---|---|---|")
            for h in hosts:
                rid = _latest_run_for_host(runs, h)
                ws = mu_warm_summary.get(rid, {})
                rows = sorted((int(k.split(":")[1]), int(k.split(":")[0]), v) for k, v in ws.items())
                for users, step, v in rows:
                    eff = efficiency.get(rid, {}).get("efficiency", {}).get(f"muwarm:{step}:{users}")
                    pw = f"{eff['avg_power_w']:,.1f}" if eff else "—"
                    lines.append(
                        f"| {h} | {users} | {step:,} | {v['agg_tps']:,.1f} | "
                        f"{v['mean_ttft_ms']:,.0f} | {v['mean_latency_ms']:,.0f} | {pw} |"
                    )
            lines.append("")
    return "\n".join(lines)
