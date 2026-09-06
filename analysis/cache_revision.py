"""Prefix-caching revision for Apple/omlx cohorts.

The observed omlx runs re-processed the full context every warm turn (warm PP
~= cold PP) — a server-config shortfall, not an Apple hardware limit (DeepSeek
on the same Mac cached, and an earlier run did too). Per the analysis decision,
we REVISE the Apple warm-cache stats to assume the same level of prefix caching
as the NVIDIA/vLLM hosts:

  * turn 1 of a warm session = a full cold prefill (measured)
  * turns 2+  = prefill ONLY the new tail: ttft = new_tokens / cold_pp(step)

Generation time is untouched (caching doesn't change decode). Cold requests are
untouched. This applies to the single-user warm session AND the multi-user warm
(muwarm) batches, so Apple numbers reflect "if omlx cached like vLLM".
"""
from __future__ import annotations

import math
from typing import Callable

APPLE_HOSTS = {"m5-max", "macstudio"}


def pp_interpolator(cold_samples: list[dict]) -> Callable[[int], float]:
    """Log-log interpolator of cold prompt-processing rate (tok/s) by context.

    ``cold_samples``: raw cold samples (need ``step`` and ``prompt_tps``).
    Returns step -> prompt_tps, extrapolating flat at the ends.
    """
    pts = sorted((s["step"], s["prompt_tps"]) for s in cold_samples
                 if s.get("prompt_tps") and s.get("step"))
    # dedupe by step (cold + any warm samples share steps)
    pts = [(x, y) for x, y in pts if x]  # safety
    uniq: dict[int, float] = {}
    for x, y in pts:
        uniq[x] = y
    pts = sorted(uniq.items())
    if not pts:
        return lambda step: float("nan")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    def f(step: int) -> float:
        if step <= xs[0]:
            return ys[0]
        if step >= xs[-1]:
            return ys[-1]
        lo = max(i for i, x in enumerate(xs) if x <= step)
        hi = min(i for i, x in enumerate(xs) if x >= step)
        if lo == hi:
            return ys[lo]
        d = math.log(xs[hi]) - math.log(xs[lo])
        if d == 0:
            return ys[lo]
        frac = (math.log(step) - math.log(xs[lo])) / d
        return math.exp(math.log(ys[lo]) + frac * (math.log(ys[hi]) - math.log(ys[lo])))

    return f


def revised_ttft_ms(sample: dict, pp: Callable[[int], float]) -> float:
    """TTFT for a warm turn assuming the prefix is cached: only the new tail is
    prefilled at the host's cold prompt-processing rate."""
    new = sample.get("new_prompt_tokens") or sample.get("prompt_tokens") or 0
    rate = pp(sample["step"])
    if not new or not rate or rate != rate:
        return sample["ttft_ms"]
    return (new / rate) * 1000.0


def revised_warm_turns(warm: dict[int, dict], cold_samples: list[dict]) -> dict[int, dict]:
    """Return revised warm samples (turn 1 = measured full prefill, rest = cached)."""
    pp = pp_interpolator(cold_samples)
    out: dict[int, dict] = {}
    steps = sorted(warm)
    for i, step in enumerate(steps):
        s = dict(warm[step])
        if i > 0:  # first turn is a cache miss (full prefill, measured)
            s["ttft_ms"] = revised_ttft_ms(s, pp)
        s["total_ms"] = s["ttft_ms"] + s["tg_ms"]
        out[step] = s
    return out


def revised_batch(batch: list[dict], cold_samples: list[dict]) -> list[dict]:
    """Revise a muwarm batch's per-stream TTFTs to assume cached warm turns."""
    pp = pp_interpolator(cold_samples)
    return [{**s, "ttft_ms": revised_ttft_ms(s, pp)} for s in batch]
