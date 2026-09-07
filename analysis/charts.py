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
NEXT = "qwen3.8-next"

# hosts shared by ALL THREE models (qwen3.8-next deliberately skipped m5-max
# and the RTX 5090), so every three-way comparison is drawn on this subset.
NEXT_HOSTS = ["macstudio", "gx10-top", "tr-pro-6000"]

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
ARCH_COLOR = {"MoE": "#1d4ed8", "Dense": "#dc2626", "Next": "#c026d3"}
ARCH_LABEL = {"MoE": "MoE \u00b7 Qwen3.6-35B-A3B (~3B active)", "Dense": "Dense \u00b7 Qwen3.8-27B (27B active)"}
ARCH_LS = {"MoE": "-", "Dense": "--"}
ARCH_MK = {"MoE": "o", "Dense": "s"}

# per-scenario line/marker style for the multi-model panels (concurrency, QoS,
# verdict map) — one style per model, so a panel can hold 3-4 models at once.
SCEN_STYLE = {
    MOE:      {"ls": "-",  "mk": "o", "label": "MoE \u00b7 Qwen3.6-35B-A3B"},
    DENSE:    {"ls": "--", "mk": "s", "label": "Dense \u00b7 Qwen3.8-27B"},
    NEXT:     {"ls": "-.", "mk": "^", "label": "MoE \u00b7 Qwen3.8 Next"},
    "deepseekv4": {"ls": ":", "mk": "D", "label": "MoE \u00b7 DeepSeek-V4-Flash (larger)"},
}
NEXT_LABEL = "Qwen3.8 Next (MoE)"

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
    cold cache, log-log, one panel per model. PP normalizes for context size, so
    it is the fair cross-length prefill comparison (absolute TTFT isn't)."""
    scs = [(MOE, "MoE \u00b7 Qwen3.6-35B-A3B"), (DENSE, "Dense \u00b7 Qwen3.8-27B"),
           (NEXT, "MoE \u00b7 Qwen3.8 Next")]
    contexts = sorted({c for sc, _ in scs for h in HOST_ORDER for c in ds.get((sc, h), {"cold": {}})["cold"]})
    contexts = [c for c in contexts if c <= 100000]
    fig, axes = plt.subplots(1, len(scs), figsize=(6.25 * len(scs), 5.2), sharey=True)
    for ax, (sc, title) in zip(axes, scs):
        for h in HOST_ORDER:
            if (sc, h) not in ds:
                continue
            ys = [ds_cold(ds, sc, h, c, "prompt_tps") for c in contexts]
            ax.plot(contexts, ys, SCEN_STYLE[sc]["ls"],
                    color=HOST_COLOR[h], marker=SCEN_STYLE[sc]["mk"],
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
    scs = [(MOE, "MoE \u00b7 Qwen3.6-35B-A3B"), (DENSE, "Dense \u00b7 Qwen3.8-27B"),
           (NEXT, "MoE \u00b7 Qwen3.8 Next")]
    fig, axes = plt.subplots(1, len(scs), figsize=(6.25 * len(scs), 5.2), sharey=True)
    for ax, (sc, title) in zip(axes, scs):
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
    fig.suptitle("Prefix caching at ~50k context: vLLM/sglang hosts measured 3-17x the "
                 "fresh-prompt rate; Apple bars ASSUME the same caching (the Apple runs tested "
                 "did not actually reuse the prefix)",
                 fontsize=12.5, fontweight="bold")
    save(fig, "03_cache_effect.png", [0, 0, 1, 0.88])


# ---------------------------------------------------------------- chart 4
# scenario panels for the concurrency/QoS charts: (scenario, panel title)
CONC_SCENARIOS = [
    (MOE, "MoE \u00b7 Qwen3.6-35B-A3B"),
    (DENSE, "Dense \u00b7 Qwen3.8-27B"),
    (NEXT, "MoE \u00b7 Qwen3.8 Next (3 hosts)"),
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
            st = SCEN_STYLE[sc]
            ax.plot(users, ys, st["ls"], color=HOST_COLOR[h], marker=st["mk"], ms=5, lw=1.8,
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
    fig.suptitle("Concurrency at 10k context: vLLM/sglang servers batch streams and scale; "
                 "Apple/llama.cpp and the RTX 5090 serialize (ov 0%). Qwen3.8 Next ran "
                 "on 3 of the 5 machines only.",
                 fontsize=12, fontweight="bold")
    save(fig, "04_concurrency_aggregate.png", [0, 0, 1, 0.86])


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
            st = SCEN_STYLE[sc]
            ax.plot(users, lat, st["ls"], color=HOST_COLOR[h], marker=st["mk"], ms=5, lw=1.8,
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


# ---------------------------------------------------------------- chart 8
def chart_three_way(ds) -> None:
    """Qwen3.8 Next (a MoE) vs the two models already in the report, on the three
    machines all three ran (macstudio / gx10-top / tr-pro-6000).

    Four panels: single-user decode and prefill at 50k, the 6-user aggregate at
    10k, and the 6-turn warm session (Apple's session bar is the cache-REVISED
    one, labelled "assumed", exactly as in chart 3).
    """
    scs = [MOE, DENSE, NEXT]
    labels = {MOE: "Qwen3.6 (MoE)", DENSE: "Qwen3.8 (dense)", NEXT: NEXT_LABEL}
    mcolor = {MOE: ARCH_COLOR["MoE"], DENSE: ARCH_COLOR["Dense"], NEXT: ARCH_COLOR["Next"]}
    hosts = [h for h in NEXT_HOSTS if all((sc, h) in ds for sc in scs)]
    if not hosts:
        return

    def session_s(sc, h):
        """Total 6-turn warm session (s); nan if this run lacks the 40k-50k steps.
        For Apple hosts this is the cache-REVISED session (see cache_revision.py)."""
        warm = ds[(sc, h)]["warm"]
        if not all(c in warm for c in (40000, 50000)):
            return float("nan")
        return sum(warm[c]["total_ms"]["median"] for c in sorted(warm)) / 1000.0

    panels = [
        ("decode @50k (tok/s) — higher is better",
         lambda sc, h: ds_cold(ds, sc, h, CTX_50K, "out_tps"), True),
        ("prompt processing @50k (tok/s) — higher is better",
         lambda sc, h: ds_cold(ds, sc, h, CTX_50K, "prompt_tps"), True),
        ("aggregate decode, 6 users @10k (tok/s) — higher is better",
         lambda sc, h: ds[(sc, h)]["mucold"].get((10000, 6), {}).get("agg_tps", float("nan")), True),
        ("6-turn warm session, 40k→50k (s) — lower is better",
         session_s, False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for ax, (title, getter, zero) in zip(axes.ravel(), panels):
        x = np.arange(len(hosts))
        wdt = 0.26
        for i, sc in enumerate(scs):
            vals = [getter(sc, h) for h in hosts]
            off = (i - 1) * wdt
            ax.bar(x + off, vals, wdt, color=mcolor[sc], label=labels[sc])
            for xi, h, v in zip(x, hosts, vals):
                if v is None or v != v:
                    continue
                tag = "\n(assumed)" if (title.startswith("6-turn") and h in APPLE_HOSTS) else ""
                ax.annotate((f"{v:,.0f}{tag}" if v >= 100 else f"{v:.1f}{tag}"),
                            (xi + off, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7, color="#374151")
        ax.set_xticks(x)
        ax.set_xticklabels([HOST_DISPLAY[h] for h in hosts], fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.margins(y=0.24)
        if zero:
            set_yzero(ax)
    axes[0][0].legend(fontsize=9, loc="upper left", ncol=3)
    fig.suptitle("Qwen3.8 Next (MoE) joins the comparison — on the three machines all three "
                 "models ran (M3 Ultra / GB10 x2 / RTX PRO 6000). Apple warm-session bar ASSUMES "
                 "prefix caching; the run itself did not reuse the prefix.",
                 fontsize=11.5, fontweight="bold")
    save(fig, "08_qwen38_next_three_way.png", [0, 0, 1, 0.92])


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
    for sc in (MOE, DENSE, NEXT):
        for h in HOST_ORDER:
            if (sc, h) not in ds:
                continue
            speed = ds_cold(ds, sc, h, ctx, "out_tps")
            agg6 = ds[(sc, h)]["mucold"].get((cctx, 6), {}).get("agg_tps", float("nan"))
            if not (speed == speed and agg6 == agg6):
                continue
            plotted_any = True
            ax.scatter(speed, agg6, s=240, color=HOST_COLOR[h],
                       marker=SCEN_STYLE[sc]["mk"],
                       edgecolor="white", linewidth=1.2, zorder=3)
            lbl = ds[(sc, h)]["short"] + " · " + HOST_DISPLAY[h]
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
    chart_three_way(ds)
    print("done")


if __name__ == "__main__":
    main()
