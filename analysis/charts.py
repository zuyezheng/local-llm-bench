"""Cross-scenario analytics charts: MoE vs dense on 5 hosts, concurrency,
context/cache effects, efficiency, and a "which hardware for which job" map.

Reads the canonical dataset from analyze.build_dataset() and writes PNGs to
analysis/charts/ plus prints the headline numbers used in the narrative.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze import build_dataset  # noqa: E402
from cache_revision import APPLE_HOSTS  # noqa: E402

OUT = ROOT / "charts"
OUT.mkdir(parents=True, exist_ok=True)

MOE = "qwen3.6-dflash"
DENSE = "qwen3.8-dflash"

# ---- visual system -------------------------------------------------------
# host -> color (line charts). Architecture -> bar color (bar charts).
HOST_COLOR = {
    "m5-max": "#0e7490",
    "macstudio": "#2563eb",
    "gx10-top": "#059669",
    "tr-pro-5090": "#d97706",
    "tr-pro-6000": "#7c3aed",
}
HOST_DISPLAY = {
    "m5-max": "M5 Max (Apple)",
    "macstudio": "M3 Ultra (Apple)",
    "gx10-top": "GB10 x2 (DGX Spark)",
    "tr-pro-5090": "RTX 5090",
    "tr-pro-6000": "RTX PRO 6000",
}
HOST_ORDER = ["m5-max", "macstudio", "gx10-top", "tr-pro-5090", "tr-pro-6000"]
ARCH_COLOR = {"MoE": "#1d4ed8", "Dense": "#dc2626"}
ARCH_LABEL = {"MoE": "MoE \u00b7 Qwen3.6-35B-A3B (~3B active)", "Dense": "Dense \u00b7 Qwen3.8-27B (27B active)"}
ARCH_LS = {"MoE": "-", "Dense": "--"}
ARCH_MK = {"MoE": "o", "Dense": "s"}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
})

CTX_50K = 50000


def short_arch(arch: str) -> str:
    return "MoE" if arch == "MoE" else "Dense"


def ds_cold(ds, sc, host, ctx, key):
    row = ds[(sc, host)]["cold"].get(ctx)
    return row[key]["median"] if row else float("nan")


def fmt_w(v):
    if v >= 1000:
        return f"{v/1000:.1f}s"
    return f"{v:.0f}ms"


def save(fig, name, tight_rect=None):
    fig.tight_layout(rect=tight_rect or [0, 0, 1, 0.94])
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / name}")


def set_yzero(ax):
    ax.set_ylim(bottom=0)


# ---------------------------------------------------------------- chart 1
def chart_decode_headline(ds) -> None:
    """Grouped bar: single-user generation throughput at 50k context."""
    hosts = HOST_ORDER
    moe = [ds_cold(ds, MOE, h, CTX_50K, "out_tps") for h in hosts]
    dense = [ds_cold(ds, DENSE, h, CTX_50K, "out_tps") for h in hosts]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(hosts))
    w = 0.38
    b1 = ax.bar(x - w / 2, moe, w, color=ARCH_COLOR["MoE"], label=ARCH_LABEL["MoE"])
    b2 = ax.bar(x + w / 2, dense, w, color=ARCH_COLOR["Dense"], label=ARCH_LABEL["Dense"])
    for xi, (m, d) in enumerate(zip(moe, dense)):
        speedup = m / d if d and m == m and d == d else float("nan")
        if not np.isnan(speedup) and speedup >= 1.15:
            ax.annotate(f"{speedup:.1f}x", (xi, max(m, d) + 4), ha="center",
                        fontsize=10, fontweight="bold", color=ARCH_COLOR["MoE"])
        elif not np.isnan(speedup):
            ax.annotate("tie", (xi, max(m, d) + 4), ha="center",
                        fontsize=9, color="#6b7280")
    ax.set_xticks(x)
    ax.set_xticklabels([HOST_DISPLAY[h] for h in hosts])
    ax.set_ylabel("generation throughput (tokens/s) \u2014 higher is better")
    ax.set_title("MoE vs dense decode, single user, 50k context",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.margins(y=0.18)
    set_yzero(ax)
    fig.suptitle("MoE decodes up to ~2x faster where memory bandwidth is the bottleneck \u2014 "
                 "but on the biggest card they tie",
                 fontsize=11, y=0.99)
    save(fig, "01_decode_moe_vs_dense.png", [0, 0, 1, 0.9])


# ---------------------------------------------------------------- chart 2
def chart_prefill(ds) -> None:
    """Prompt-processing throughput (PP = prompt tokens / TTFT) vs context length,
    cold cache, log-log, one panel per arch. PP normalizes for context size, so
    it is the fair cross-length prefill comparison (absolute TTFT isn't)."""
    contexts = sorted({c for sc in (MOE, DENSE) for h in HOST_ORDER for c in ds[(sc, h)]["cold"]})
    contexts = [c for c in contexts if c <= 100000]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, sc, title in ((axes[0], MOE, "MoE \u00b7 Qwen3.6-35B-A3B"),
                          (axes[1], DENSE, "Dense \u00b7 Qwen3.8-27B")):
        for h in HOST_ORDER:
            if (sc, h) not in ds:
                continue
            ys = [ds_cold(ds, sc, h, c, "prompt_tps") for c in contexts]
            ax.plot(contexts, ys, ARCH_LS["MoE" if sc == MOE else "Dense"],
                    color=HOST_COLOR[h], marker=ARCH_MK["MoE" if sc == MOE else "Dense"],
                    ms=4, lw=1.8, label=HOST_DISPLAY[h])
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("prompt / context length (tokens)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("prompt processing (tokens/s) \u2014 higher is better")
    fig.suptitle("Prefill: prompt processing throughput vs context length "
                 "(cold cache, log-log) — TTFT is context-length dependent; PP isn't",
                 fontsize=13, fontweight="bold")
    save(fig, "02_prefill_pp_context.png", [0, 0, 1, 0.9])


# ---------------------------------------------------------------- chart 3
def chart_cache_effect(ds) -> None:
    """Cold vs warm prompt-processing throughput at ~50k context per host.

    Cold PP = full fresh-prompt ingest rate; warm PP = effective ingest rate of
    a turn whose prefix is cached (full context tokens / warm TTFT), so the two
    are directly comparable tokens/s. The ratio is the prefix-cache benefit.

    Apple/omlx warm PP here is the cache_revision.py REVISED value (assumes
    vLLM-equivalent caching) \u2014 the omlx servers that produced these runs did
    not actually reuse the prefix (warm PP measured == cold PP). Bars for
    Apple hosts are annotated "(assumed)" so this chart never reads as a
    measurement it isn't; see ANALYSIS.md \u00a73 for the raw finding.
    """
    ctx = CTX_50K
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for ax, sc, title in ((axes[0], MOE, "MoE \u00b7 Qwen3.6-35B-A3B"),
                          (axes[1], DENSE, "Dense \u00b7 Qwen3.8-27B")):
        hosts = [h for h in HOST_ORDER if (sc, h) in ds]
        cold = []
        warm = []
        for h in hosts:
            c = ds[(sc, h)]["cold"].get(ctx, {}).get("prompt_tps", {}).get("median", float("nan"))
            w = ds[(sc, h)]["warm"].get(ctx, {}).get("prompt_tps_eff", {}).get("median", float("nan"))
            cold.append(c)
            warm.append(w)
        x = np.arange(len(hosts))
        wdt = 0.38
        bc = ax.bar(x - wdt / 2, cold, wdt, color="#6b7280", label="cold (fresh prompt)")
        bw = ax.bar(x + wdt / 2, warm, wdt, color="#16a34a",
                    alpha=0.9, label="warm (cached prefix)")
        for xi, (h, c, w) in enumerate(zip(hosts, cold, warm)):
            if c and w and c == c and w == w:
                tag = " (assumed)" if h in APPLE_HOSTS else ""
                ax.annotate(f"{w/c:.0f}x{tag}" if w > c else "no cache gain",
                            (xi, max(c, w) * 1.06), ha="center", fontsize=7.5)
        ax.set_ylim(bottom=0)
        ax.set_xticks(x)
        ax.set_xticklabels([HOST_DISPLAY[h] for h in hosts], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.margins(y=0.22)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("prompt processing (tokens/s) \u2014 higher is better")
    fig.suptitle("Prefix caching at ~50k context: vLLM hosts measured 3-17x the "
                 "fresh-prompt rate; Apple bars ASSUME the same caching (the omlx "
                 "runs tested did not actually reuse the prefix)",
                 fontsize=12.5, fontweight="bold")
    save(fig, "03_cache_effect.png", [0, 0, 1, 0.88])


# ---------------------------------------------------------------- chart 4
# scenario panels for the concurrency/QoS charts: (scenario, panel title)
CONC_SCENARIOS = [
    (MOE, "MoE \u00b7 Qwen3.6-35B-A3B"),
    (DENSE, "Dense \u00b7 Qwen3.8-27B"),
    ("deepseekv4", "MoE \u00b7 DeepSeek-V4-Flash (larger)"),
]


def chart_concurrency(ds) -> None:
    """Aggregate multi-user throughput vs users (1=single-user cold).

    agg_tps is measured over the streams' OVERLAPPING decode window, so it
    reflects true concurrency on servers that batch (vLLM) and the achieved
    serialized rate where streams never overlap (Apple/MLX, some consumer GPUs).
    """
    ctx = 10000
    users = [1, 2, 4, 6]
    fig, axes = plt.subplots(1, len(CONC_SCENARIOS), figsize=(5.0 * len(CONC_SCENARIOS), 5.2),
                             sharey=True)
    if len(CONC_SCENARIOS) == 1:
        axes = [axes]
    for ax, (sc, title) in zip(axes, CONC_SCENARIOS):
        for h in HOST_ORDER:
            if (sc, h) not in ds:
                continue
            base = ds_cold(ds, sc, h, ctx, "out_tps")
            agg = {u: ds[(sc, h)]["mucold"].get((ctx, u), {}).get("agg_tps", float("nan"))
                   for u in (2, 4, 6)}
            ys = [base] + [agg[u] for u in (2, 4, 6)]
            ls = ARCH_LS["MoE"] if sc in (MOE, "deepseekv4") else ARCH_LS["Dense"]
            mk = "D" if sc == "deepseekv4" else ARCH_MK["MoE" if sc == MOE else "Dense"]
            ax.plot(users, ys, ls, color=HOST_COLOR[h], marker=mk, ms=5, lw=1.8,
                    label=HOST_DISPLAY[h])
            ov6 = ds[(sc, h)]["mucold"].get((ctx, 6), {}).get("overlap_frac", 0.0)
            if ov6 and ov6 == ov6:
                ax.annotate(f"ov {ov6:.0%}", (6, ys[-1]), xytext=(4, 5),
                            textcoords="offset points", fontsize=7.5, color="#6b7280")
        ax.set_xticks(users)
        ax.set_xlabel("concurrent users (distinct contexts)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("aggregate generation throughput (tokens/s across all users)")
    fig.suptitle("Concurrency at 10k context: vLLM servers batch streams and scale; "
                 "Apple/MLX and the RTX 5090 serialize (ov 0%)",
                 fontsize=13, fontweight="bold")
    save(fig, "04_concurrency_aggregate.png", [0, 0, 1, 0.88])


# ---------------------------------------------------------------- chart 5
def chart_per_user_qos(ds) -> None:
    """Per-stream latency under concurrency (10k context)."""
    ctx = 10000
    users = [2, 4, 6]
    fig, axes = plt.subplots(1, len(CONC_SCENARIOS), figsize=(5.0 * len(CONC_SCENARIOS), 5.2),
                             sharey=True)
    if len(CONC_SCENARIOS) == 1:
        axes = [axes]
    for ax, (sc, title) in zip(axes, CONC_SCENARIOS):
        for h in HOST_ORDER:
            if (sc, h) not in ds:
                continue
            lat = [ds[(sc, h)]["mucold"].get((ctx, u), {}).get("mean_latency_ms", float("nan"))
                   for u in users]
            ls = ARCH_LS["MoE"] if sc in (MOE, "deepseekv4") else ARCH_LS["Dense"]
            mk = "D" if sc == "deepseekv4" else ARCH_MK["MoE" if sc == MOE else "Dense"]
            ax.plot(users, lat, ls, color=HOST_COLOR[h], marker=mk, ms=5, lw=1.8,
                    label=HOST_DISPLAY[h])
        ax.set_ylim(bottom=0)
        ax.set_xticks(users)
        ax.set_xlabel("concurrent users")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("per-stream latency (ms) \u2014 lower is better")
    fig.suptitle("Per-user experience at 10k context: latency inflates fastest on "
                 "Apple, and worst for the dense model",
                 fontsize=13, fontweight="bold")
    save(fig, "05_per_user_qos.png", [0, 0, 1, 0.88])


# ---------------------------------------------------------------- chart 6
def chart_efficiency(ds) -> None:
    """tokens/s per watt at 50k context, grouped bars MoE vs dense."""
    hosts = HOST_ORDER
    ctx = CTX_50K
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    moe_tpw = [ds[(MOE, h)]["cold"].get(ctx, {}).get("tps_per_w", {}).get("median", float("nan"))
               for h in hosts]
    dense_tpw = [ds[(DENSE, h)]["cold"].get(ctx, {}).get("tps_per_w", {}).get("median", float("nan"))
                 for h in hosts]
    moe_w = [ds[(MOE, h)]["cold"].get(ctx, {}).get("power_w", {}).get("median", float("nan"))
             for h in hosts]
    dense_w = [ds[(DENSE, h)]["cold"].get(ctx, {}).get("power_w", {}).get("median", float("nan"))
               for h in hosts]
    x = np.arange(len(hosts))
    wdt = 0.38
    b1 = ax.bar(x - wdt / 2, moe_tpw, wdt, color=ARCH_COLOR["MoE"], label=ARCH_LABEL["MoE"])
    b2 = ax.bar(x + wdt / 2, dense_tpw, wdt, color=ARCH_COLOR["Dense"], label=ARCH_LABEL["Dense"])
    for xi, (m, d, mw, dw) in enumerate(zip(moe_tpw, dense_tpw, moe_w, dense_w)):
        if m == m and mw == mw:
            ax.annotate(f"{mw:.0f}W", (xi - wdt / 2, m + 0.02), ha="center", fontsize=8, color="#374151")
        if d == d and dw == dw:
            ax.annotate(f"{dw:.0f}W", (xi + wdt / 2, d + 0.02), ha="center", fontsize=8, color="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels([HOST_DISPLAY[h] for h in hosts])
    ax.set_ylabel("generation efficiency (tokens/s per watt) \u2014 higher is better")
    ax.set_title("Tokens per watt at 50k context (labels = avg power draw)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.margins(y=0.22)
    set_yzero(ax)
    fig.suptitle("Efficiency: the DGX Spark leads per watt, and MoE roughly "
                 "doubles efficiency on every host",
                 fontsize=11, y=0.99)
    save(fig, "06_efficiency.png", [0, 0, 1, 0.9])


# ---------------------------------------------------------------- chart 7
def chart_verdict(ds) -> None:
    """Positioning map: single-user decode speed vs concurrency scaling.

    Also plots DeepSeek-V4-Flash (a much larger MoE) on its two hosts as
    reference points so the big-MoE story is visible too.
    """
    ctx = CTX_50K
    cctx = 10000
    fig, ax = plt.subplots(figsize=(10, 6.5))
    plotted_any = False
    for sc in (MOE, DENSE):
        arch = "MoE" if sc == MOE else "Dense"
        for h in HOST_ORDER:
            if (sc, h) not in ds:
                continue
            speed = ds_cold(ds, sc, h, ctx, "out_tps")
            agg6 = ds[(sc, h)]["mucold"].get((cctx, 6), {}).get("agg_tps", float("nan"))
            if not (speed == speed and agg6 == agg6):
                continue
            plotted_any = True
            ax.scatter(speed, agg6, s=240, color=HOST_COLOR[h],
                       marker="o" if arch == "MoE" else "s",
                       edgecolor="white", linewidth=1.2, zorder=3)
            lbl = f"{short_arch(arch)} \u00b7 {HOST_DISPLAY[h]}"
            ax.annotate(lbl, (speed, agg6), xytext=(7, 5),
                        textcoords="offset points", fontsize=8.5, color="#111827")
    # reference: no-scaling line agg6 == speed (flat aggregate)
    ax.plot([0, 250], [0, 250], ":", color="#9ca3af", lw=1.5, label="no concurrency gain (agg = 1 user)")
    # DeepSeek-V4-Flash (much larger MoE) as reference points on its 2 hosts
    for h in HOST_ORDER:
        if ("deepseekv4", h) not in ds:
            continue
        speed = ds_cold(ds, "deepseekv4", h, ctx, "out_tps")
        agg6 = ds[("deepseekv4", h)]["mucold"].get((cctx, 6), {}).get("agg_tps", float("nan"))
        if speed == speed and agg6 == agg6:
            plotted_any = True
            ax.scatter(speed, agg6, s=210, color="#111827",
                       marker="D", edgecolor="white", linewidth=1.2, zorder=3)
            ax.annotate(f"DeepSeek-V4 \u00b7 {HOST_DISPLAY[h]}",
                        (speed, agg6), xytext=(7, -9), textcoords="offset points",
                        fontsize=8, color="#374151")
    ax.set_xlabel(f"single-user decode at {ctx/1000:.0f}k context (tokens/s)")
    ax.set_ylabel(f"aggregate throughput at 6 users, 10k context (tokens/s)")
    ax.set_title("Hardware positioning: how fast, and does it scale with users?",
                 fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    set_yzero(ax)
    if plotted_any:
        fig.suptitle("Top-right = fast single user AND scales to many users. "
                     "Bottom-right = fast alone, poor for teams.",
                     fontsize=11, y=0.99)
        save(fig, "07_verdict_map.png", [0, 0, 1, 0.9])
    else:
        plt.close(fig)


# ---------------------------------------------------------------- run
def main() -> None:
    ds = build_dataset()
    # purge stale filenames from renamed charts
    for stale in ("02_prefill_ttft_context.png",):
        p = OUT / stale
        if p.exists():
            p.unlink()
            print(f"  removed stale {stale}")
    print("building charts ...")
    chart_decode_headline(ds)
    chart_prefill(ds)
    chart_cache_effect(ds)
    chart_concurrency(ds)
    chart_per_user_qos(ds)
    chart_efficiency(ds)
    chart_verdict(ds)
    print("done")


if __name__ == "__main__":
    main()
