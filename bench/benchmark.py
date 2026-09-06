"""Benchmark orchestration: cold-cache, warm-cache, concurrency and multi-user.

The run is driven by a deterministic PLAN of (phase, step, iter, batch) items.
Each item's prompts are derived from a per-item RNG seeded from
(host, cfg.seed, phase, step, iter, batch), so:
  * warm / multi-user warm bases use the largest corpus region -> reproducible
  * resuming a run reproduces the same prompts for the items that remain
"""
from __future__ import annotations

import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import metrics as met
from . import spec
from .client import CompletionsClient
from .config import Config, HostConfig
from .corpus import Corpus
from .results import ResultsStore, concurrent_batch_aggregates

CONCURRENT_PHASES = ("conc", "mucold", "muwarm")
PREFIX_PHASES = ("warm", "muwarm")


def _iter_label(phase: str) -> str:
    """'iter=' is a session/repeat index for cold+warm, but for the
    concurrent-batch phases the same field holds the concurrency level
    (users fired at once) — label it accordingly so logs aren't misread."""
    return "concurrency" if phase in CONCURRENT_PHASES else "iter"


SYSTEM_PROMPT = (
    "You are an AI assisting on a real software-engineering and writing task. "
    "Continue the provided content exactly where it ends. Write fluent, coherent "
    "continuation: code where the content is code, prose where it is prose. "
    "Do not summarize or restate the content; just keep producing it."
)
USER_SUFFIX = "\n\n<|continue|> Continue writing from this exact point. Do not stop early."


@dataclass
class HostOutcome:
    run_id: str
    host: HostConfig
    model: str
    ok: bool
    error: str = ""
    summary: dict[str, Any] | None = None
    machine_spec: dict[str, Any] | None = None
    model_spec: dict[str, Any] | None = None
    n_samples: int = 0
    elapsed_s: float = 0.0


@dataclass
class PlanItem:
    phase: str
    step: int
    iter: int
    batch: int
    texts: list[str] = field(default_factory=list)  # prompts for the concurrent requests

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (self.phase, self.step, self.iter, self.batch)


def _stable_seed(url: str, base_seed: int) -> int:
    return base_seed ^ zlib.crc32(url.encode())


def _key_rng(host_url: str, base_seed: int, phase: str, step: int, it: int, batch: int) -> np.random.Generator:
    s = zlib.crc32(f"{host_url}|{phase}|{step}|{it}|{batch}".encode()) ^ (int(base_seed) & 0xFFFFFFFF)
    return np.random.default_rng(s)


# ------------------------------------------------------------------- plan

def build_plan(cfg: Config, host_url: str, cap: int | None, corpus: Corpus) -> list[PlanItem]:
    """Deterministic list of every request this host must issue."""
    plan: list[PlanItem] = []

    def fits(n: int) -> bool:
        return not (cap and n + cfg.max_tokens > cap)

    if not cfg.skip_cold:
        for ctx in cfg.context_lengths:
            if not fits(ctx):
                continue
            for it in range(cfg.cold_iters):
                rng = _key_rng(host_url, cfg.seed, "cold", ctx, it, 0)
                plan.append(PlanItem("cold", ctx, it, 0, [
                    corpus.select_text(ctx, rng, cfg.corpus_kind, cfg.code_ratio)]))

    if not cfg.skip_warm and cfg.warm_sizes:
        max_warm = max(cfg.warm_sizes)
        for s in range(cfg.warm_sessions):
            prefixes = corpus.build_prefixes(max_warm, cfg.warm_sizes, np.random.default_rng(cfg.seed), cfg.warm_kind)
            for step, text in prefixes:
                if not fits(step):
                    continue
                plan.append(PlanItem("warm", step, s, 0, [text]))

    # The concurrency sweep and the multi-user phase are alternative ways to
    # measure concurrent throughput. When multi_users is configured, the
    # multi-user phase covers it (with distinct contexts per user), so the
    # fixed 1-N concurrency sweep is skipped automatically — no skip_conc needed.
    if not cfg.skip_conc and cfg.concurrency and not cfg.multi_users:
        for ctx in cfg.concurrency_contexts:
            if not fits(ctx):
                continue
            for conc in cfg.concurrency:
                for b in range(cfg.conc_batches):
                    rng = _key_rng(host_url, cfg.seed, "conc", ctx, conc, b)
                    plan.append(PlanItem("conc", ctx, conc, b, [
                        corpus.select_text(ctx, rng, cfg.corpus_kind, cfg.code_ratio)
                        for _ in range(conc)
                    ]))

    if not cfg.skip_multiuser:
        if cfg.multi_users and cfg.multi_user_contexts:
            for u in cfg.multi_users:
                for ctx in cfg.multi_user_contexts:
                    if not fits(ctx):
                        continue
                    for b in range(cfg.multi_user_batches):
                        rng = _key_rng(host_url, cfg.seed, "mucold", ctx, u, b)
                        plan.append(PlanItem("mucold", ctx, u, b, [
                            corpus.select_text(ctx, rng, cfg.corpus_kind, cfg.code_ratio)
                            for _ in range(u)
                        ]))
        if cfg.multi_users and cfg.multi_user_warm_sizes:
            mu_steps = sorted(set(cfg.multi_user_warm_sizes))
            mu_max = max(mu_steps)
            for u in cfg.multi_users:
                for b in range(cfg.multi_user_batches):
                    user_bases = corpus.build_user_prefixes(
                        u, mu_max, mu_steps, np.random.default_rng(cfg.seed), cfg.warm_kind
                    )
                    for st in mu_steps:
                        if not fits(st):
                            continue
                        texts = [dict(base).get(st) for base in user_bases if any(x == st for x, _ in base)]
                        if texts:
                            plan.append(PlanItem("muwarm", st, u, b, texts))
    return plan


def done_set(db: ResultsStore, run_id: str) -> set[tuple[str, int, int, int]]:
    """(phase, step, iter, batch) items whose requests ALL succeeded."""
    bykey: dict[tuple, list[bool]] = {}
    for s in db.samples(run_id):
        bykey.setdefault((s["phase"], s["step"], s["iter"], s.get("batch", 0)), []).append(bool(s["ok"]))
    return {k for k, oks in bykey.items() if oks and all(oks)}


# ------------------------------------------------------------- batch summary

def _batch_aggregates(db: ResultsStore, run_id: str, phases: tuple[str, ...]) -> dict[str, Any]:
    """Aggregate throughput per (step, iter) from concurrent-batch phases."""
    ss = [s for s in db.samples(run_id) if s["phase"] in phases and s["ok"]]
    groups: dict[tuple, dict[int, list[dict]]] = {}
    for s in ss:
        groups.setdefault((s["step"], s["iter"]), {}).setdefault(s.get("batch", 0), []).append(s)
    out: dict[str, Any] = {}
    for (ctx, it), batches in sorted(groups.items()):
        per_batch = []
        for _, batch in batches.items():
            if not batch:
                continue
            agg = concurrent_batch_aggregates(batch)
            if agg["n"] == 0:
                continue
            per_batch.append({
                **agg,
                "mean_ttft_ms": sum(s["ttft_ms"] for s in batch) / len(batch),
                "mean_latency_ms": sum(s["total_ms"] for s in batch) / len(batch),
                # mean (not summed) per-stream prompt-processing rate: unlike
                # generation, concurrent prefill isn't naturally additive
                # across backends, so this tracks whether each individual
                # stream's OWN prefill rate degrades under concurrency.
                "mean_prompt_tps": sum(s["prompt_tps"] for s in batch) / len(batch),
            })
        if not per_batch:
            continue
        med = lambda k: float(sorted(x[k] for x in per_batch)[len(per_batch) // 2])
        out[f"{ctx}:{it}"] = {
            "n_batches": len(per_batch),
            "agg_tps": med("agg_tps"),
            "agg_union_tps": med("agg_union_tps"),
            "wall_s": med("wall_s"),
            "overlap_s": med("overlap_s"),
            "overlap_frac": med("overlap_frac"),
            "total_tokens": med("total_tokens"),
            "mean_ttft_ms": med("mean_ttft_ms"),
            "mean_latency_ms": med("mean_latency_ms"),
            "mean_prompt_tps": med("mean_prompt_tps"),
        }
    return out


# ------------------------------------------------------------------- runner

def run_host(
    host: HostConfig,
    cfg: Config,
    corpus: Corpus,
    db: ResultsStore,
    tag: str = "",
    resume_run_id: str | None = None,
    progress: Optional[Callable[[str], None]] = None,
) -> HostOutcome:
    """Run (or resume) the full benchmark for one host."""
    model_list = spec.fetch_model_list(host.url)
    model = host.model or cfg.model or spec.discover_model(host.url, model_list)
    model_info = spec.model_spec(host.url, model, model_list)
    client = CompletionsClient(
        host.url, timeout=cfg.timeout, read_timeout=cfg.read_timeout,
        connect_timeout=cfg.connect_timeout, max_retries=cfg.max_retries,
    )

    caps = [c for c in (host.max_context, model_info.get("max_model_len")) if c]
    cap = int(min(caps)) if caps else None

    # --- identity: fresh run vs resume ---
    resuming = resume_run_id is not None
    run_id = resume_run_id or (
        f"{time.strftime('%Y%m%d-%H%M%S')}-{host.name}{('-'+tag) if tag else ''}"
    )
    prior = db.run(run_id)
    if resuming:
        if prior is None:
            print(f"[{host.name}] resume: run {run_id} not found in DB", flush=True)
            return HostOutcome(run_id=run_id, host=host, model=model or "?", ok=False,
                               error="run not found")
        if cfg.seed == 0 and prior.get("config", {}).get("seed"):
            # restore the original run's seed so prompts reproduce exactly
            cfg.seed = int(prior["config"]["seed"])
    start_wall = float(prior["start_epoch"]) if (resuming and prior and prior.get("start_epoch")) else time.time()

    plan = build_plan(cfg, host.url, cap, corpus)
    done = done_set(db, run_id) if resuming else set()
    pending = [p for p in plan if p.key not in done]
    if resuming:
        print(f"[{host.name}] resume {run_id}: {len(done)}/{len(plan)} items done, "
              f"{len(pending)} remaining", flush=True)
    if not pending:
        n_done = len([s for s in db.samples(run_id) if s["ok"]])
        summary = {
            "run_id": run_id, "host": host.name, "model": model or "?", "status": "ok",
            "n_samples": n_done, "errors": [], "elapsed_s": 0.0,
            "resumed": resuming, "items_total": len(plan), "items_done": len(done),
            "stats": met.run_aggregates(db, run_id),
            "aggregate": db.aggregate(run_id),
            "efficiency": met.compute_run_efficiency(db, run_id),
            "concurrency": _batch_aggregates(db, run_id, ("conc",)),
            "multi_user": {
                "cold": _batch_aggregates(db, run_id, ("mucold",)),
                "warm": _batch_aggregates(db, run_id, ("muwarm",)),
            },
        }
        db.finish_run(run_id, "ok", prior.get("machine_spec", {}) or {},
                      prior.get("model_spec", {}) or {})
        return HostOutcome(run_id=run_id, host=host, model=model or "?", ok=True,
                           summary=summary, machine_spec=prior.get("machine_spec"),
                           model_spec=prior.get("model_spec") or model_info,
                           n_samples=n_done, elapsed_s=0.0)

    outcome = HostOutcome(run_id=run_id, host=host, model=model or "?", ok=True)
    db.start_run(run_id, host.name, model or "?", tag, cfg.serializable_config(), start_wall)
    errors: list[str] = []
    n = 0
    emit_lock = threading.Lock()

    exporters = host.all_exporters()
    sampler = None
    if cfg.metrics_enabled and exporters:
        sampler = met.MetricsSampler(exporters, cfg.metrics_interval, run_id, host.name, db, start_wall)
        sampler.start()

    def phase_header(msg: str) -> None:
        print(f"[{host.name}] --- {msg} ---", flush=True)

    # Warm-cache sessions share a byte-identical, growing prefix: a
    # prefix-caching server only has to prefill the NEW tail each step, so
    # TTFT reflects processing `new_prompt_tokens`, not the full context.
    # Tracks the last-seen prompt_tokens per growing session (keyed so each
    # concurrent user in a muwarm batch is tracked independently), seeded
    # from prior samples so resumed runs keep computing correct deltas.
    prev_prompt_tokens: dict[tuple, int] = {}
    for s in db.samples(run_id):
        if s.get("ok") and s.get("phase") in PREFIX_PHASES:
            key = (s["phase"], s["iter"], s.get("batch", 0), s.get("session_pos", 0))
            prev_prompt_tokens[key] = s["prompt_tokens"]

    def emit(
        phase: str, step: int, it: int, res, ts_start: float, ts_end: float,
        batch: int = 0, session_pos: int = 0,
    ) -> dict[str, float]:
        nonlocal n
        with emit_lock:
            n += 1
            # ---- per-sample derived metrics ----
            new_prompt_tokens = res.prompt_tokens
            prompt_tps = res.prompt_tps
            if res.ok and phase in PREFIX_PHASES:
                key = (phase, it, batch, session_pos)
                prev = prev_prompt_tokens.get(key)
                if prev is not None:
                    new_prompt_tokens = max(res.prompt_tokens - prev, 0)
                    prompt_tps = (
                        new_prompt_tokens / (res.ttft_ms / 1000.0) if res.ttft_ms else 0.0
                    )
                prev_prompt_tokens[key] = res.prompt_tokens
            content = max(res.tg_tokens - res.reasoning_tokens, 0)
            # tgs: generated tokens INCLUDING reasoning, per generation second.
            # content+reasoning avoids double-counting when the server already
            # folds reasoning into completion_tokens, and still counts thinking
            # when it does not.
            tg_s = (content + res.reasoning_tokens) / (res.tg_ms / 1000.0) if res.tg_ms > 0 else 0.0
            dur_s = max(ts_end - ts_start, res.total_ms / 1000.0)
            power_total = 0.0
            if res.ok:
                try:
                    series = met.build_power_series(db.metric_samples(run_id))
                    pk = met.pick_power_key(series)
                    if pk:
                        power_total = met.sample_power(series, ts_start, ts_end, pk)
                except Exception:
                    power_total = 0.0
            # concurrent phases divide cluster power by the number of users
            n_users = it if phase in CONCURRENT_PHASES else 1
            power_w = power_total / n_users if n_users else power_total
            wh = power_w * dur_s / 3600.0
            tps_per_w = res.out_tps / power_w if power_w > 0 else 0.0
            db.insert_sample(
                run_id,
                {
                    "host": host.name, "model": model or "?", "phase": phase,
                    "step": step, "iter": it, "session_pos": session_pos,
                    "context_tokens": res.prompt_tokens,
                    "prompt_tokens": res.prompt_tokens,
                    "new_prompt_tokens": new_prompt_tokens,
                    "max_tokens": cfg.max_tokens,
                    "ttft_ms": res.ttft_ms, "prompt_tps": prompt_tps,
                    "tpot_ms": res.tpot_ms, "tg_ms": res.tg_ms,
                    "tg_tokens": res.tg_tokens, "out_tps": res.out_tps,
                    "total_ms": res.total_ms, "ok": res.ok,
                    "reasoning_tokens": res.reasoning_tokens,
                    "content_tokens": content, "tg_s": tg_s,
                    "power_w": power_w, "wh": wh, "tps_per_w": tps_per_w,
                    "error": res.error, "finish_reason": res.finish_reason,
                    "batch": batch,
                    "ts_start_epoch": ts_start, "ts_end_epoch": ts_end,
                },
            )
            if progress:
                progress(host.name)
            return {"elapsed_s": dur_s, "ppts": prompt_tps, "tgs": tg_s, "avg_w": power_w}

    def _read_budget(content: str) -> float:
        est = max(len(corpus.enc.encode(content)), 1)
        return min(cfg.read_timeout, max(cfg.min_read_timeout, est / cfg.min_pp_tps + cfg.ttft_margin))

    def call(phase: str, step: int, it: int, batch: int, content: str,
             barrier: threading.Barrier | None = None, session_pos: int = 0) -> None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content + USER_SUFFIX},
        ]
        budget = _read_budget(content)
        if barrier is not None:
            # synchronized start: every user blocks here after its local prep,
            # then all fire their HTTP request at the same instant
            try:
                barrier.wait(timeout=120.0)
            except threading.BrokenBarrierError:
                pass  # a sibling failed before the gate — proceed anyway
        res = client.chat(
            model or "", messages, cfg.max_tokens,
            temperature=cfg.temperature, seed=cfg.seed if cfg.seed else None,
            read_timeout=budget,
        )
        t_end = time.time()
        label = _iter_label(phase)
        if not res.ok:
            errors.append(f"{phase} step={step} {label}={it} batch={batch}: {res.error}")
        per = None
        try:
            per = emit(
                phase, step, it, res, t_end - res.total_ms / 1000.0 - start_wall,
                t_end - start_wall, batch, session_pos,
            )
        except Exception as e:  # noqa: BLE001 — never let bookkeeping kill a request
            errors.append(f"{phase} step={step} {label}={it} batch={batch}: emit error: {e}")
            print(f"[{host.name}] WARN emit failed for {phase} step={step}: {e}", flush=True)
        status = " FAILED" if not res.ok else ""
        if per:
            print(
                f"[{host.name}] {phase} step={step:,} {label}={it}{status} "
                f"| {per['elapsed_s']:.1f}s | ppts {per['ppts']:,.0f} "
                f"| tgs {per['tgs']:,.1f} | avg_w {per['avg_w']:,.1f}",
                flush=True,
            )
        else:
            print(f"[{host.name}] {phase} step={step:,} {label}={it}{status}", flush=True)

    def call_batch(item: PlanItem) -> None:
        n = len(item.texts)
        # one thread per user; a barrier (for n>1) releases all users at once
        barrier = threading.Barrier(n) if n > 1 else None
        threads = [
            threading.Thread(
                target=call,
                args=(item.phase, item.step, item.iter, item.batch, t, barrier, pos),
            )
            for pos, t in enumerate(item.texts)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # --- execute pending items, grouped by phase for readable headers ---
    last_phase: str | None = None
    for item in pending:
        if item.phase != last_phase:
            last_phase = item.phase
            if item.phase == "cold":
                phase_header("cold cache (random slices per request)")
            elif item.phase == "warm":
                phase_header("warm cache — progressive coding session")
            elif item.phase == "conc":
                phase_header("concurrency sweep")
            elif item.phase == "mucold":
                phase_header("multi-user cold (distinct context per user)")
            elif item.phase == "muwarm":
                phase_header("multi-user warm (per-user progressive sessions)")
        call_batch(item)

    elapsed = time.time() - start_wall
    status = "ok" if not errors else "partial"
    if sampler:
        sampler.stop()
        sampler.join(timeout=cfg.metrics_interval * 3)

    machine = spec.fetch_machine_spec(host.ssh)
    prom = spec.fetch_prometheus_metrics(host.url)
    if prom:
        machine = dict(machine or {})
        machine["prometheus"] = prom
    hw: dict[str, Any] = {}
    for exp in exporters:
        got = met.fetch(exp.url, exp.type, gpu_model=getattr(exp, "gpu_model", None))
        if got.get("info"):
            hw[f"{exp.name}:info"] = got["info"]
    if hw:
        machine = dict(machine or {})
        machine["exporter_info"] = hw

    eff = met.compute_run_efficiency(db, run_id)
    run_stats = met.run_aggregates(db, run_id)
    outcome.ok = status == "ok"
    outcome.error = "; ".join(errors[:5])
    outcome.machine_spec = machine
    outcome.model_spec = model_info
    outcome.n_samples = n
    outcome.elapsed_s = elapsed
    outcome.summary = {
        "run_id": run_id,
        "host": host.name,
        "model": model,
        "status": status,
        "n_samples": n,
        "errors": errors[:50],
        "elapsed_s": elapsed,
        "resumed": resuming,
        "items_total": len(plan),
        "items_done": len(done),
        "stats": run_stats,
        "metrics_fetches": sampler.fetches if sampler else 0,
        "aggregate": db.aggregate(run_id),
        "efficiency": eff,
        "concurrency": _batch_aggregates(db, run_id, ("conc",)),
        "multi_user": {
            "cold": _batch_aggregates(db, run_id, ("mucold",)),
            "warm": _batch_aggregates(db, run_id, ("muwarm",)),
        },
    }
    db.finish_run(run_id, status, machine, model_info)
    print(
        f"[{host.name}] done in {elapsed:.0f}s | ppts {run_stats['ppts']:,.0f} "
        f"| tgs {run_stats['tgs']:,.1f} | avg_w {run_stats['avg_w']:,.1f} "
        f"| {n} samples, status={status}",
        flush=True,
    )
    return outcome
