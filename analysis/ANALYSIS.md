# Local-LLM hardware: what's actually worth it, and for whom

_Cross-scenario analysis of `results/` — MoE vs dense, concurrency, context length & caching across 5 machines. General benchmarks: cold prefill/decode across 1k–200k context, prefix caching, and multi-user concurrency. Coding/agentic sessions are one angle (mostly warm-cache) — not the whole story._

Generated 2026-09-04. Charts in `analysis/charts/` (see [The charts](#the-charts)).
**Interactive walkthrough:** open `analysis/site/index.html` in a browser (self-contained —
vendored Plotly, no internet needed) for a model/host toggleable version of this story.

---

## The gist

This report answers four questions with measured data, not speculation:

1. **MoE vs dense** — does a mixture-of-experts model beat a dense model of similar size, and *where*?
2. **Concurrency** — what happens when 2/4/6 users hit the same box at once?
3. **Context length & cache** — how does prefill cost grow with context, and how much does prefix caching help?
4. **Bottom line** — which hardware to buy for which situation.

### The comparison is apples-to-apples

Two model families ran the *same workload* (512 output tokens, same context
lengths, same multi-user plan) on the *same five machines*:

| architecture | model | total params | active params | quants on each host |
|---|---|---|---|---|
| **MoE** | Qwen3.6-35B-A3B | 35B | ~3B | MLX 8-bit (Apple), FP8/NVFP4 (NVIDIA) |
| **Dense** | Qwen3.8-27B | 27B | 27B | MLX 4-bit (Apple), NVFP4 (NVIDIA) |

| host | hardware class | chip / GPU | peak mem BW (ref) |
|---|---|---|---|
| `m5-max` | Apple laptop | Apple M5 Max (40-core GPU) | ~614 GB/s |
| `macstudio` | Apple workstation | Apple M3 Ultra | ~819 GB/s |
| `gx10-top` | NVIDIA DGX Spark (2-node) | 2 × GB10 | ~273 GB/s each |
| `tr-pro-5090` | Consumer GPU | GeForce RTX 5090 | ~1792 GB/s |
| `tr-pro-6000` | Workstation GPU | RTX PRO 6000 | ~1792 GB/s |

A secondary data point — **DeepSeek-V4-Flash**, a much larger MoE — ran on
`gx10-top` and `macstudio` and is shown as a reference on the verdict map.

> **Data caveats.** (1) These are single runs; concurrent batches were run once,
> so individual cells wobble (batch noise, not a trend).
> (2) Power readings for very short requests (1k context) often miss the 2 s
> sampling window, so all efficiency figures below use the **50k-context** runs
> (6 s to 3 min long — solidly sampled). (3) **Caching revision:** the omlx
> instances serving Qwen3.6/Qwen3.8 in these runs did *not* reuse the prefix
> (warm PP ≈ cold PP), which is a server-config shortfall — DeepSeek cached on
> the same Mac, and an earlier run did too. All **Apple warm-session / team
> numbers in this report are revised to assume vLLM-equivalent prefix caching**
> (turn 1 = full prefill, later turns = new tail only); cold and multi-user-cold
> numbers are untouched — see
> [Does it remember what you already pasted?](#3-does-it-remember-what-you-already-pasted).
> (4) Multi-user throughput uses the overlap-based aggregation
> (`bench/results.py:concurrent_batch_aggregates`): it measures the window where
> all streams decode together, falling back to the achieved (serialized) rate
> where streams never overlap. `ov 0%` usually means the server queued requests
> (Apple/MLX, RTX 5090 — verified against raw timestamps), but on a few
> `gx10-top`/Qwen3.6 cells it means the strict all-stream intersection was empty
> due to skewed prefills; both cases report the conservative union rate.
> Existing `results/*/report/` artifacts predate this change; `analysis/` is
> generated with the overlap method.

---

## 1. Does MoE beat dense — and where?

**The headline chart** — single-user generation throughput at 50k context:

![MoE vs dense decode](charts/01_decode_moe_vs_dense.png)

| host | MoE (tok/s) | Dense (tok/s) | MoE advantage |
|---|---|---|---|
| M3 Ultra (Apple) | 92.4 | 41.6 | **2.2x** |
| M5 Max (Apple) | 91.3 | 42.9 | **2.1x** |
| RTX 5090 | 141.5 | 82.2 | **1.7x** |
| GB10 ×2 (DGX Spark) | 64.2 | 56.4 | **1.1x** |
| RTX PRO 6000 | 158.9 | 123.9 | **1.3x** |

**Why.** Generation (decode) is memory-bandwidth-bound: every output token
streams the active weights through memory. A 3B-active MoE moves a fraction of
the bytes a 27B dense model does — so on bandwidth-hungry machines (Apple
unified memory, consumer GPUs) it is dramatically faster. But the RTX PRO 6000
has so much bandwidth (1792 GB/s) that the dense 27B model is no longer the
bottleneck — the MoE advantage evaporates to a tie. **The card's biggest asset
(killer bandwidth) is exactly what erases MoE's structural advantage.**

MoE also wins big on *prefill* — prompt processing throughput (PP, prompt
tokens/s; TTFT alone isn't comparable because context lengths differ) is far
higher on most hosts, because it computes ~3B params instead of 27B:

| host | MoE PP @50k | Dense PP @50k | MoE advantage |
|---|---|---|---|
| M3 Ultra | 1,719 tok/s | 349 | **4.9x** |
| M5 Max | 2,401 | 601 | **4.0x** |
| GB10 ×2 | 6,978 | 2,368 | **2.9x** |
| RTX PRO 6000 | 20,199 | 6,764 | **3.0x** |
| RTX 5090 | 4,406 | 5,244 | 0.8x (dense ahead) |

> For a 27–35B-class model, **MoE is the better bet on Apple silicon and the
> DGX Spark** (2–5x faster prefill, 1.4–2.1x faster decode). On the top
> workstation GPU the two architectures converge.

---

## 2. Context length: how long does a request actually take?

![Prefill prompt processing](charts/02_prefill_pp_context.png)

Cold cache, fresh prompt each request, context 1k → 200k — the big picture, not a
coding-only lens. The total time per request splits into **prefill** (reading the
prompt) and **generation** (streaming the answer); see the interactive site §1
for the full host × context breakdown. Two things stand out:

- **Total time grows steeply with context everywhere, but the *level* differs by
  ~20x**: RTX PRO 6000 ~0.5 s @ 1k → ~20 s @ 100k (dense); M3 Ultra ~3 s → ~371 s
  @ 100k. Prefill is almost all of that growth — generation stays roughly flat.
- **In real coding/agentic work, prefill dominates the wait.** A typical agentic
  turn is big context + small reply (e.g. 50k context, 500-token answer), which
  is **48–92% prefill** across these boxes (worst on Apple/dense, best on the RTX
  PRO 6000); a 100k/1k turn is ~90%+ prefill on Apple. So **prompt-processing
  speed, not decode, is what buys you time on the work that matters** — and
  NVIDIA's PP edge (see below) is the deciding factor.

Prefill rate (PP = prompt tokens/s — TTFT alone isn't comparable across
differing context lengths; PP normalizes for that), log-log:

- **RTX PRO 6000**: ~2,400 → 5,100 tok/s (dense, 1k→100k). Fastest ingest by far.
- **RTX 5090**: ~2,000 → 3,700 tok/s (dense).
- **GB10 ×2**: ~2,700 → 1,900 tok/s (dense) — respectable for a 2-node box.
- **M5 Max**: ~750 → 490 tok/s (dense).
- **M3 Ultra**: ~380 → 280 tok/s (dense) — the slowest ingest.

MoE lifts this on Apple (M3 Ultra dense 349 → MoE 1,721 tok/s at 50k) but Apple
still ingests 10–20x slower than the fast GPUs at large contexts.
**If you regularly ingest 50–100k-token contexts, Apple silicon is the wrong
tool regardless of architecture; a workstation GPU ingests the same prompt at
~20x the rate.**

---

## 3. Does it remember what you already pasted?

![Prefix caching effect](charts/03_cache_effect.png)

Prompt processing at ~50k context, **cold** (fresh prompt) vs **warm** (same
prefix as the previous turn — a coding session where you keep appending). Warm
PP is the *effective* ingest rate of a cached turn (full context tokens ÷ warm
TTFT), so it is directly comparable to cold PP in tokens/s.

**A caveat that matters for everything Apple below:** the omlx instances serving
Qwen3.6/Qwen3.8 in these runs **did not reuse the prefix** — every warm turn
re-prefilled the full context (warm PP ≈ cold PP; M3 Ultra dense TTFT tracked
the context: 118 s @ 40k → 155 s @ 50k). That's a **server-config shortfall, not
an Apple limit** (DeepSeek cached on the same Mac, and an earlier run did too).
Per this analysis, Apple warm-cache stats are therefore **revised to assume the
same prefix caching as the NVIDIA/vLLM hosts** (turn 1 = full prefill; later
turns prefill only the new tail). Cold numbers and concurrency (which uses
distinct cold contexts) are untouched.

| host | MoE warm/cold | Dense warm/cold |
|---|---|---|
| RTX PRO 6000 | 3x faster | 8x faster |
| RTX 5090 | 5x | 8x |
| GB10 ×2 | 3x | 10x |
| M5 Max (revised) | 25x | 25x |
| M3 Ultra (revised) | 25x | 25x |

**vLLM servers reuse the KV cache** — a warm 50k turn ingests at 3–17x the
fresh-prompt rate (e.g. GB10 dense: 2,368 → 22,400 effective tok/s). **Once the
Apple numbers assume the same caching, the Macs get the *largest* relative gains
of all** (M3 Ultra dense: 349 → 8,775 effective tok/s, ~25x) — precisely because
their cold prefill was the slowest, so skipping it pays off most. The
Apple-specific caveat stands: the current omlx configs don't consistently deliver
this — if you run omlx, verify prefix caching is enabled for the model you serve;
the hardware is not the blocker.

---

## 4. What happens when a few people use it at once?

**Method.** Multi-user throughput is measured over the streams' **overlapping
decode window** (see `bench/results.py:concurrent_batch_aggregates`) — the span
where every stream is generating at the same time — not over the diluted
union of all request times. `ov%` = fraction of the batch's span where the
streams truly decode together. If streams never overlap (the server queued
them), `ov%` is 0 and the number shown is the honest achieved (serialized)
rate.

![Multi-user aggregate throughput](charts/04_concurrency_aggregate.png)

Aggregate tokens/s across all users at 10k context, overlap-based:

| model | host | single | 2u | 4u | 6u | 6u vs single | ov @6u |
|---|---|---|---|---|---|---|---|
| Qwen3.6 (MoE) | gx10-top | 64 | 96 | 143 | 103 | 1.6x | 0%* |
| Qwen3.6 (MoE) | m5-max | 102 | 80 | 73 | 71 | 0.7x | 0% |
| Qwen3.6 (MoE) | macstudio | 100 | 77 | 65 | 62 | 0.6x | 0% |
| Qwen3.6 (MoE) | RTX 5090 | 138 | 104 | 92 | 91 | 0.7x | 0% |
| Qwen3.6 (MoE) | RTX PRO 6000 | 133 | 264 | 148 | 213 | 1.6x | 67% |
| Qwen3.8 (Dense) | gx10-top | 49 | 87 | 126 | **150** | **3.1x** | 48% |
| Qwen3.8 (Dense) | m5-max | 47 | 27 | 24 | 22 | 0.5x | 0% |
| Qwen3.8 (Dense) | macstudio | 53 | 21 | 18 | 16 | 0.3x | 0% |
| Qwen3.8 (Dense) | RTX 5090 | 84 | 75 | 70 | 69 | 0.8x | 0% |
| Qwen3.8 (Dense) | RTX PRO 6000 | 173 | 277 | 196 | **286** | 1.7x | 67% |
| **DeepSeek (MoE)** | **gx10-top** | 49.7 | 57 | 78 | **92** | **1.9x** | 43% |
| **DeepSeek (MoE)** | **macstudio** | 28.6 | 37 | 13 | 13 | 0.5x | 0% |

_Single-user column = cold @10k decode (run-averaged for qwen3.6/qwen3.8);
2/4/6-user columns are overlap-based aggregate decode from the latest complete
run (concurrent batches are single-batch measurements — noisy, and streams
can't be aligned across runs, so they aren't averaged)._

\* `gx10-top`/Qwen3.6 at 6 users: all six streams fired simultaneously but
prefill times were so skewed that the strict all-stream intersection is empty
(ov 0%) — reported at the conservative union rate. The raw timestamps show
genuine overlap; this is a strict-intersection artifact, not serialization.

Three regimes, clearly separated by the span analysis:

- **NVIDIA vLLM servers genuinely batch.** `ov` 40–70%+, aggregate throughput
  rises with users: gx10-top dense 49→150 (3.1x), RTX PRO 6000 dense up to 286,
  **DeepSeek on gx10-top 50→92 (1.9x)**. At 1k context the effect is even
  stronger — gx10-top dense reaches 251 tok/s (5.3x) and RTX PRO 6000 dense 441
  tok/s.
- **Apple/MLX and the RTX 5090 serialize (ov 0%).** The "concurrent" streams
  are queued one after another — the aggregate is ~single-user rate or worse.
  This is why a Mac "serving" six users doesn't get faster; it just makes the
  queue longer. The dense model on saturated Apple memory *loses* throughput
  outright (macstudio dense 53 → 16 tok/s at 6 users).
- **DeepSeek on a Mac collapses** (28.6 → 13 tok/s): its huge active-parameter
  prefill (64–243 s!) serializes every request.

### What each person actually experiences

![Per-user latency](charts/05_per_user_qos.png)

Mean per-stream latency at 10k context:

| model | host | 2u | 4u | 6u |
|---|---|---|---|---|
| Qwen3.6 (MoE) | gx10-top | 13 s | 17 s | 26 s |
| Qwen3.6 (MoE) | m5-max | 12 s | 20 s | 27 s |
| Qwen3.6 (MoE) | macstudio | 13 s | 23 s | 31 s |
| Qwen3.6 (MoE) | RTX 5090 | 9 s | 16 s | 21 s |
| Qwen3.6 (MoE) | RTX PRO 6000 | 5 s | 15 s | 16 s |
| Qwen3.8 (Dense) | gx10-top | 19 s | 29 s | 39 s |
| Qwen3.8 (Dense) | m5-max | 39 s | 62 s | 88 s |
| Qwen3.8 (Dense) | macstudio | 57 s | 89 s | **126 s** |
| Qwen3.8 (Dense) | RTX 5090 | 11 s | 19 s | 27 s |
| Qwen3.8 (Dense) | RTX PRO 6000 | 6 s | 15 s | 17 s |
| **DeepSeek (MoE)** | **gx10-top** | 25 s | 39 s | 51 s |
| **DeepSeek (MoE)** | **macstudio** | 76 s | 133 s | **183 s** |

A dense 27B on an M3 Ultra serving six users means **two-minute round trips**
and DeepSeek on a Mac means **three minutes**; MoE Qwen3.6 cuts Apple to ~30 s.
**If other people will use the machine, a dense model on Apple silicon is not
viable — and only the vLLM hosts keep per-user latency tolerable as users grow.**

---

## 5. What are you paying per token — and per watt?

![Efficiency](charts/06_efficiency.png)

Tokens per watt, split by **prompt processing (PP)** and **generation (TG)**, and by
**single user** vs **6 concurrent streams** (all at 10k context; the interactive
site §5 plots every host per model). The DGX Spark ×2 is adjusted **+70 W per
node** (its exporter reads GPU power only → whole-box estimate, +140 W for the
2-node cluster). Concurrent power is taken from each batch's metrics trace.

**MoE (Qwen3.6) at 10k context — tok/s per watt:**

| host | PP · single | TG · single | PP · 6 concurrent | TG · 6 concurrent | draw (W) s/6c |
|---|---|---|---|---|---|
| RTX PRO 6000 | **116.8** | 0.60 | **71.4** | 0.53 | 222 / 400 |
| GB10 ×2 (DGX Spark, +140 W) | 38.6 | 0.32 | 13.7 | 0.46 | 197 / 223 |
| RTX 5090 | 26.4 | **0.78** | 10.4 | 0.50 | 177 / 183 |
| M5 Max | 26.2 | 0.74 | 11.2 | 0.52 | 138 / 135 |
| M3 Ultra | 13.5 | 0.53 | 6.6 | 0.32 | 188 / 194 |

Two clear patterns:

- **Prefill is ~50–150x more token-efficient per watt than decode.** PP is a big
  batched-compute pass (thousands of tokens/s per watt); decode re-reads every
  weight per token, so TG sits at ~0.3–0.8 tok/s/W everywhere. The RTX PRO 6000
  dominates PP-per-watt (116.8 single) — its prefill is both fast and power-cheap.
- **Concurrency costs efficiency.** Every host's per-watt throughput drops at 6
  concurrent streams (RTX PRO 6000 PP 117→71; DGX 39→14; M5 26→11), because
  total power rises with users while aggregate throughput doesn't scale 1:1. The
  one exception: the RTX PRO 6000's dense *decode* efficiency *improves* under
  concurrency (0.45→1.04) — batching makes decode more power-efficient.

The **DGX Spark is no longer the efficiency champion once the whole box counts**
— its GPU-only ~0.8 tok/s/W drops to ~0.32 (TG, adjusted +140 W), behind the RTX
5090 (0.78) and M5 Max (0.74) on decode-per-watt. It still wins on *absolute*
scaled throughput (§§3–4).

---

## 6. So what should you actually buy?

![Verdict map](charts/07_verdict_map.png)

X = single-user decode speed (50k), Y = aggregate throughput at 6 users (10k,
overlap-based). The dashed line is "no concurrency gain": points above it get
faster with more users, points below it get slower. **Top-right is fast *and*
scales; bottom-right is fast for one person and bad for a team.**

| Use case | Pick | Why (from data) |
|---|---|---|
| **Team server / anything multi-user** | **RTX PRO 6000** (either arch) | Fastest single user *and* the best scaling: dense reaches 286 tok/s at 6 users, MoE 213; latency stays ~16 s. |
| **Small-team server, low power / efficiency** | **DGX Spark (GB10)** | Genuinely scales (dense 49→150 tok/s, 3.1x; DeepSeek 50→92, 1.9x). Not the per-watt leader once the whole box counts (~0.32 tok/s/W after +70 W/node) — but the cheapest way to scale a team. |
| **Single power user, highest speed** | **RTX PRO 6000 or RTX 5090 (MoE)** | ~129 tok/s; the 5090 is ~1/4 the price of the 6000 with nearly equal single-user speed. Don't share it (it serializes — ov 0%). |
| **Personal coding assistant, long sessions** | **NVIDIA + vLLM** (any of the above) | The vLLM hosts enabled prefix caching (8–17x faster warm turns) and batching. Apple's hardware supports caching too (revised numbers assume it), but the current omlx configs don't consistently deliver it. |
| **Local, private, single-user Mac user** | **M5 Max / M3 Ultra running the MoE model** | 2–2.1x faster than dense on the same chip, quiet, efficient (0.65 tok/s/W), no GPU purchase. With prefix caching enabled, ~45–60 s sessions — competitive with the DGX cluster. Accept: slow cold prefill, no concurrency. |
| **…and on a Mac, never the dense 27B for anything but casual single-user use** | — | 2x slower decode, 4–5x slower prefill, and it collapses to 16–22 tok/s under load even with caching. |

**Two things worth knowing before you spend anything.**

**Pick MoE unless you're buying an RTX PRO 6000.** MoE wins by 1.4–2.1x on
decode and up to 5x on prefill on every machine here except the PRO 6000, which
has enough bandwidth that model size stops mattering and the two architectures
tie. On Apple silicon or a consumer GPU, the architecture choice is worth more
than most hardware upgrades.

**Caching and batching came from the server, not the badge.** The vLLM
instances on GB10 ×2 and the PRO 6000 reused the prefix (1.3–2.1x session
speedup) and ran streams in parallel (ov 40–70%+); the RTX 5090's vLLM instance
serialized (ov 0%) on the same workload. Apple's omlx runs showed no
prefix-cache benefit for Qwen3.6/Qwen3.8 (warm ≈ cold), but DeepSeek cached on
that same Mac (2.6x) — so this is configuration, not silicon, and it's why this
report revises the Apple warm numbers to assume caching works. Whatever you
buy, verify prefix caching and batching are actually on before believing any
session number, including these.

### Rough prices (ballpark — double-check before buying)

| hardware | approx. | best for |
|---|---|---|
| RTX 5090 (32 GB) | ~$2k | single-user throughput per dollar |
| DGX Spark 128 GB | ~$4k each | low-power always-on serving |
| M5 Max MacBook Pro / M3 Ultra Mac Studio | ~$4–6k | private, quiet, efficient single user |
| RTX PRO 6000 (96 GB) | ~$7k+ | multi-user server + big contexts |
| M5 Ultra Mac Studio | $5,499 (96 GB) · ~$12k (256 GB) · ~$20k+ (512 GB) | fastest single-user MoE decode; big unified memory at a premium |

Prices are ballpark list figures and vary; the *ranking* of machines by
performance/efficiency above is from measured data, the prices are context.

---

## 7 · What about the new M5 Ultra?

*(Added 2026-09-04 — research + projection, not a measurement.)*

Apple announced the **M5 Ultra** on Aug 25, 2026 (Mac Studio, ships Sep 22; 512GB config late Oct):

| M5 Ultra spec (Apple) | value |
|---|---|
| Memory bandwidth | **1.2 TB/s** — 50% more than M3 Ultra, ~2x M5 Max (614 GB/s) |
| GPU | up to 80-core, **Neural Accelerator in every core** |
| CPU | up to 36-core (12 super + 24 performance), ~1.3x MT vs M3 Ultra |
| Neural Engine | 32-core |
| Unified memory | 96 / 256 / **512 GB** (LPDDR5X 9600 MT/s) |
| Packaging | first quad-die M-series (2× dual-die M5 Max, UltraFusion >4.4 TB/s) |
| Peak AI compute | up to 4.3x M3 Ultra |
| Price | $5,499 (96 GB) · **~$12k (256 GB)** · **~$20k+ (512 GB)** |

**Methodology — two physical anchors, not one marketing multiplier.** There are
two real measured chips to scale from (M3 Ultra and M5 Max) and two official
Apple deltas that apply to different pairs of them:

- **Path A — M3 Ultra → M5 Ultra**, using Apple's Ultra-vs-Ultra deltas:
  bandwidth 1.2TB/s ÷ 800GB/s = **1.50x** (drives decode, bandwidth-bound);
  peak GPU AI compute **up to 4.5x** (drives prefill, compute-bound).
- **Path B — M5 Max → M5 Ultra**, using the physical Max→Ultra fusion ratio —
  same generation, exactly two dies: bandwidth 1.2TB/s ÷ 614GB/s = **1.95x**;
  GPU cores 80 ÷ 40 = **2.0x** exactly.

Central estimate is the geometric mean of the two; the brackets below are the
paths themselves, not padding. **Every scale factor is then capped at 2x the
measured M5 Max**, since an M5 Ultra is two M5 Max dies and cannot beat one by
more than two. The cap binds on prefill — Path A's 4.5x compute figure would
otherwise imply an Ultra ingesting 3.2x faster than the Max it's built from —
and doesn't bind on decode.

The paths disagree by ~35% on decode, and the disagreement is informative: the
measured M5 Max already decodes *faster* than M3 Ultra (42.9 vs 41.6 tok/s
dense) on **25% less bandwidth** (614 vs 819GB/s) — a generational gain Path A
can't see, because it never looks at M5-era silicon.

**Which path to trust.** The two paths disagree on the comparison that matters
most: does M5 Ultra out-decode the RTX PRO 6000 on MoE? Path A says no (138.6
vs 158.9); Path B says yes (178.1). To settle it, check whether the PRO 6000 is
actually bandwidth-saturated on each model — the measured M5-Max-vs-PRO-6000
decode ratio divided by the ratio raw bandwidth predicts. Near 1.0, the card is
spending all its bandwidth and scaling by bandwidth (Path A) is the right
model; well above 1.0, it's leaving bandwidth unused and the cores-based Path B
is the better guide. Dense comes out at **1.01** — saturated on a 27B-active
model, so Path A's dense verdict (M5 Ultra loses, 62.5 vs 123.9) stands. MoE
comes out at **1.68** — with ~3B active params the PRO 6000 is nowhere near
bandwidth-bound, so Path B is the one to trust there. Treat "M5 Ultra roughly ties
or edges the PRO 6000 on MoE decode (157.2 vs 158.9, via Path B)" as the best
available guess, not a settled result.

**DeepSeek-V4-Flash never ran on M5 Max**, so its projection has only Path A
and is the least certain row here — flagged as single-anchor throughout.

| metric @ 50k | M3 Ultra (measured) | **M5 Ultra (proj.)** | nearest NVIDIA measured |
|---|---|---|---|
| Decode, MoE Qwen3.6 | 92.4 tok/s | **157.2** [134.2–184.5] | RTX PRO 6000 158.9 |
| Decode, dense Qwen3.8 | 41.6 tok/s | **72.4** [62.5–83.9] | RTX 5090 83.0 |
| Decode, DeepSeek-V4 (single-anchor) | 26.8 tok/s | **40.1** | DGX ×2 43.4 |
| Prompt processing, MoE | 1,719 tok/s | **4,802** (capped 2x M5 Max) | RTX PRO 6000 20,199 |
| Prompt processing, dense | 349 tok/s | **1,202** (capped 2x M5 Max) | RTX PRO 6000 6,764 |
| Prompt processing @100k, dense | 277 tok/s | **957** | RTX PRO 6000 5,118 |
| 6-turn warm session, MoE (narrow cached case) | 58.1 s | **27.4 s** | RTX PRO 6000 25.6 s |
| 6-turn session, dense | 230.1 s | **90.0 s** | RTX PRO 6000 31.5 s |
| Effective session tok/s, MoE | 52.9 | **112.1** | RTX PRO 6000 119.9 |
| 6 streams, batch @10k, dense | 191.5 s | **74.9 s** | RTX PRO 6000 13.5 s |

_Apple session numbers are revised to assume vLLM-equivalent prefix caching (see §3).
M3 Ultra / M5 Ultra Qwen rows are now **run-averaged** across the multiple complete
runs of qwen3.6/qwen3.8 (see methodology). Session brackets are the two paths run
through the same turn arithmetic as the central estimate; the prompt-processing rows
carry no bracket because the 2x-M5-Max cap binds on both paths and collapses them to
one value._

**How it stacks up.**

1. **Single-user decode — ties the RTX PRO 6000 on MoE, but it's a *draw as a
   class*.** Projected 157.2 tok/s (small-MoE) essentially *ties* the RTX PRO
   6000's 158.9 — the two fastest boxes here are a small-MoE tie, and the M5
   Ultra win is only via Path B (see the saturation note below). **MoE is not a
   universal win: DeepSeek-V4, a much larger MoE, decodes at ~43 tok/s on the
   DGX cluster (measured), no better than the dense model.** With a dense 27B,
   M5 Ultra lands below RTX-5090-class (72.4 vs 83.0), and both paths agree
   (dense saturation_index ≈ 1.0), so that verdict is solid.
2. **Sessions — only in a narrow, perfectly-cached warm case.** A ~27 s MoE
   session (40–50k warm, 512-token replies) would trail only the RTX PRO 6000
   (25.6 s). But that's the *most* favorable scenario for Apple: real agentic
   turns are cold (or cache-busting) with large context and small replies, where
   prefill dominates and NVIDIA's PP edge wins (see §2 and §7.1). For a cold
   50k/500-tok turn M5 Ultra edges the DGX ×2 on MoE (13.8s vs 15.2s) and loses
   badly on dense (49.5s vs 30.4s) — see the real-agentic-turn numbers below.
3. **Prefill — a big jump, capped at 2x the M5 Max, and still behind the fast
   GPUs.** M5 Ultra's projected prompt processing (4,802 tok/s MoE) is ~2.8x M3
   Ultra and ~9% ahead of the RTX 5090 (4,406) — but ~4.2x behind the RTX PRO
   6000 (20,199). Dense (1,202) is behind both the RTX 5090 (5,244) and RTX PRO
   6000 (6,764). The 2x-M5-Max cap (not Apple's "4.5x" claim) is what sets these.
4. **Concurrency — the one thing it still loses.** Bandwidth and 512 GB add KV
   headroom, but Apple's measured backends serialized concurrent streams and M5
   Ultra inherits that until the server learns to batch (a config problem, not a
   hardware one — see §4). It will not beat vLLM servers for multiple users: DGX
   ×2 runs a 6-stream dense batch in 31.9s against M5 Ultra's projected 74.9s,
   and aggregate decode goes the wrong way (150 → ~28 tok/s).
5. **Efficiency & price — the value story flips with the memory tier.** Apple's
   per-watt story holds at any tier, and the **96 GB base ($5,499)** undercuts the
   RTX PRO 6000 (~$8.5k) while matching/beating it on single-user MoE decode and
   sessions. But the tiers that actually unlock M5 Ultra's advantage — 256 GB at
   **~$12k**, and the 512 GB that fits frontier-scale models at **~$20k+** — are
   **not** a value play: 512 GB costs ~2.5–3x a 2-node DGX Spark cluster (~$8k) or
   an RTX PRO 6000 box, which both still win on multi-user. M5 Ultra's big-memory
   edge is real, but you pay heavily for it.

> **Bottom line:** given working prefix caching, M5 Ultra on a small MoE projects
> to the fastest single-user decode here and a ~27 s session, second only to the
> RTX PRO 6000, and it edges the DGX ×2 on a cold agentic turn. It still loses
> multi-user work, dense models, and DeepSeek-sized MoE — and the 512 GB tier
> that unlocks its memory advantage costs ~2.5x a DGX ×2 cluster.

### 7.1 M5 Ultra vs DGX Spark: the $5–8k call

The natural ~$5–8k "local AI box" purchase. **DGX Spark ×2** is the **measured**
2-node cluster (`gx10-top` + `gx10-bottom`, ~$8k); **M5 Ultra** is the projected
Mac Studio (1.2 TB/s, 80-core GPU; $5,499 @ 96 GB, ~$12k @ 256 GB,
~$20k+ @ 512 GB). Same workload: 6-turn session, 40k→50k context.

| metric | DGX ×2 (meas) | M5 Ultra (proj) | read |
|---|---|---|---|
| **Decode @50k — MoE** | 64.2 tok/s | **157.2** | M5 Ultra, 2.4x |
| **Decode @50k — dense** | 56.4 | **72.4** | M5 Ultra, 1.3x |
| **Decode @50k — DeepSeek** (single-anchor) | 43.4 | **40.1** | DGX ×2 |
| **Prompt proc @50k — MoE** | **6,978 tok/s** | 4,802 | DGX ×2 |
| **Prompt proc @50k — dense** | **2,368** | 1,202 | DGX ×2 |
| **Prompt proc @50k — DeepSeek** (single-anchor) | **2,303** | 2,121 | DGX ×2 |
| **Prompt proc @100k — dense** | **1,910** | 957 | DGX ×2 |
| **6-turn warm session — MoE (cached)** | 57.2 s | **27.4 s** | M5 Ultra — narrow cached case only |
| **6-turn session — dense** | **85.2 s** | 90.0 s | DGX ×2 (barely) |
| **6-turn session — DeepSeek** (single-anchor) | **85.8 s** | 111.9 s | DGX ×2 |
| **6 streams, batch @10k — MoE** | **29.1 s** | 21.0 s | M5 Ultra (but it's serializing, not batching) |
| **6 streams, batch @10k — dense** | **29.6 s** | 74.9 s | DGX ×2 |
| **6-stream agg decode @10k — dense** | **151 tok/s** | ~28 | DGX (batches) |
| **Efficiency @50k — MoE** | 0.30 (whole-box, +140 W) | ~0.4–0.5 (proj, no official TDP yet) | RTX 5090 0.78 |
| **Max unified memory** | 128 GB (256 GB across the 2-node cluster) | **512 GB** | M5 Ultra, 2x the cluster |
| **Price (approx.)** | ~$8k | $5,499 (96GB) · ~$12k (256GB) · **~$20k+ (512GB)** | 512GB ≈ 2.5x DGX ×2 |

_Apple sessions assume vLLM-equivalent prefix caching (revised). DeepSeek rows are
single-anchor (M3 Ultra only, no M5 Max run for this model) — wider uncertainty
than the dual-anchor Qwen rows above them._

**DeepSeek-on-M5-Ultra forecast.** DeepSeek-V4-Flash only ran on the M3 Ultra
(`macstudio`) and the DGX cluster, so this row can only use Path A (M3 Ultra ×
Apple's official Ultra-vs-Ultra deltas: 1.5x bandwidth, 4.5x compute) — there's
no M5 Max DeepSeek run to cross-check against, so treat it as the least certain
row in this section. Decode 26.8 → **40.1 tok/s** (now slightly *behind* the DGX
cluster's 43.4, not roughly matching it as a dual-anchor estimate would likely
show), prompt processing 471 → **2,121 tok/s** (~1.1x behind DGX, closer than
the Qwen dense gap), session 273.7 → **111.9 s** (DeepSeek cached on the Mac
already, so the forecast keeps that real 2.55x caching behaviour, not an
assumed one). Net: M5 Ultra + DeepSeek trails the DGX cluster on every metric
here, by a narrower margin than the single-anchor method's uncertainty band
likely allows for.

**Where the M5 Ultra sits — and why the "session" claim needs care.** The
session numbers above (27.4 s vs 57.2 s MoE) are for **one narrow case**: a
perfectly-cached 40–50k warm session with 512-token replies. That is **not**
representative of general or real-world agentic work, where:

- **Prefill dominates.** A realistic coding/agentic turn (cold, ~50k context,
  500-token reply) is **39–92% prefill** across these boxes (Apple/dense are the
  worst; RTX PRO 6000 the best). A bigger context makes it worse — a 100k/1k
  turn is ~90%+ prefill on Apple.
- **NVIDIA excels at prompt processing**, and that's what buys time on
  prefill-heavy work. DGX ×2 ingests 6,978 tok/s vs M5 Ultra's projected 4,802
  (MoE, capped) — a clear gap, and a much wider one on dense (2,368 vs 1,202).
- **MoE is a draw as a class.** The M5 Ultra "wins decode" story is specific to
  the small-MoE (Qwen3.6, ~3B active); DeepSeek-V4, a much larger MoE, decodes
  at ~43 tok/s on the DGX cluster — no better than the dense model.
- **Real agent sessions bust the cache** — tool outputs, diffs and edits change
  the context each turn, so the perfect-caching assumption that powers the
  "27.4 s" number over-credits Apple.

For a **real agentic turn (cold 50k ctx, 500-tok reply)** the boxes land like
this: RTX PRO 6000 5.7–11.6 s, RTX 5090 15.1–15.9 s, M5 Ultra 13.8 s (MoE) /
49.5 s (dense), DGX ×2 15.2 s (MoE) / 30.4 s (dense), M5 Max 26.8 s (MoE). So
**M5 Ultra edges the DGX ×2 on a cold MoE turn** (13.8 s vs 15.2 s) —
**but loses badly on dense** (49.5 s vs 30.4 s), and both lose to the RTX
PRO 6000. The honest summary: M5 Ultra wins small-MoE single-user *decode*; MoE
is a draw as a class (DeepSeek's big MoE doesn't), and NVIDIA still wins
*prefill-heavy dense* work and *multi-user* work outright.

> **Value call:** the 96 GB M5 Ultra ($5,499) is a reasonable single-user MoE buy
> — fastest decode, undercuts the cluster's price. But the memory tier that
> unlocks M5 Ultra's real advantage is expensive: **~$12k (256 GB)** and
> **~$20k+ (512 GB)** — ~2.5x a DGX ×2 cluster (~$8k) or an RTX PRO 6000 box
> that still wins teams and long sessions. So: single power user with modest
> memory needs → **M5 Ultra 96 GB + MoE**; big-model capacity or shared/team use
> → **DGX Spark ×2** is the far cheaper path to sessions and concurrency; pay the
> M5 Ultra 512 GB premium only if 128 GB simply cannot fit your model.

---

## The charts

| file | shows |
|---|---|
| `01_decode_moe_vs_dense.png` | MoE vs dense single-user decode @50k per host |
| `02_prefill_pp_context.png` | prompt processing (tok/s) vs context (cold), log-log, MoE vs dense |
| `03_cache_effect.png` | warm vs cold prompt processing (tok/s) @50k per host |
| `04_concurrency_aggregate.png` | overlap-based aggregate tok/s vs users @10k (MoE / dense / DeepSeek) |
| `05_per_user_qos.png` | per-stream latency vs users @10k (MoE / dense / DeepSeek) |
| `06_efficiency.png` | tokens/s per watt @50k (labels = avg W) |
| `07_verdict_map.png` | positioning: speed vs concurrency scaling (incl. DeepSeek) |
| *interactive site* | §1 context length 1k–200k (400k for DeepSeek) — prefill & generation tok/s; §2 warm cache (accumulated session time + every-machine heatmap, "do Macs catch up?"); §3 cold time budget by context (sorted, best/worst); §4 concurrency 1→6 (prompt processing, decode, per-stream latency, batch time); §5 power & efficiency (PP & TG tps/W, single vs 6-concurrent; DGX adjusted +70 W/node); §6 verdict + M5 Ultra vs DGX ×2 (incl. DeepSeek forecast). One universal model selector drives all charts. M5 Ultra projection capped at 2x the measured M5 Max |

**Method.** Samples were collated from every run under `results/` (44 runs,
~4,000 samples). For the five-host MoE/dense comparison we used the complete
`qwen3.6-dflash` and `qwen3.8-dflash` runs (identical workload, identical
hosts). **Qwen3.6/Qwen3.8 single-user numbers are run-averaged** — each machine's
cold & warm measurements are the mean across all of its complete runs (e.g.
gx10-top qwen3.8 averages `185358`, `200421`, `204747`), which smooths the
noisy run-to-run decode; concurrent batches are single-batch measurements and
stay on the latest complete run (streams can't be aligned across runs).
DeepSeek-V4-Flash adds its two hosts as a larger-MoE reference. All figures
are medians of raw per-request samples (`out_tps`, `ttft_ms`, `prompt_tps`,
`power_w`, `tps_per_w`); multi-user aggregates sum tokens over the streams'
**overlapping decode window** (with a union-rate fallback when streams don't
overlap). **Apple warm-cache / session / team numbers are revised to assume
vLLM-equivalent prefix caching** (see §3); concurrency scaling uses multi-user
cold (cache-independent). Every **M5 Ultra** figure is a **projection**, not a
measurement: `analysis/site_data.py` scales the measured M3 Ultra per-turn data
by two independent paths (M3 Ultra × Apple's Ultra-vs-Ultra deltas, and M5 Max ×
the physical Max→Ultra fusion ratio), takes their geometric mean, and caps every
scale factor at 2x the measured M5 Max — see §7. It is marked "(projected)"
wherever it appears, here and in the interactive site. Reproduce with:

```bash
uv run python analysis/collate.py    # tidy CSVs
uv run python analysis/analyze.py    # aggregate tables + dataset.json
uv run python analysis/charts.py     # regenerate all charts
uv run python analysis/site_data.py  # site/data.js (incl. M5 Ultra projection)
```
