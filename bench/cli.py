"""llm-bench CLI: init / run / report."""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any

from . import __version__
from . import artifact
from .benchmark import run_host
from .config import Config, DEFAULT_SCENARIO_PATH, default_scenario_yaml, PROJECT_ROOT
from .corpus import Corpus
from .results import ResultsStore


def _p(arg: str | None) -> Path | None:
    return Path(arg).expanduser().resolve() if arg else None


def cmd_init(args: argparse.Namespace) -> int:
    out = _p(args.scenario) or (PROJECT_ROOT / "config" / "scenarios" / "example.yaml")
    if out.exists() and not args.force:
        print(f"[init] {out} already exists; not overwriting. Use --force.")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(default_scenario_yaml())
    print(f"[init] wrote {out}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    scenario_path = _p(args.scenario) or DEFAULT_SCENARIO_PATH
    if not scenario_path.exists():
        print(f"[run] scenario file not found: {scenario_path}")
        return 1
    print(f"[run] scenario: {scenario_path}")

    import yaml as _yaml
    raw = _yaml.safe_load(scenario_path.read_text()) or {}
    cfg = Config.load(_p(args.config))
    cfg.apply_dict(raw)
    tag = args.tag or raw.get("tag", "")

    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model"] = args.model
    if args.max_tokens:
        overrides["max_tokens"] = args.max_tokens
    if args.contexts:
        overrides["context_lengths"] = args.contexts
    if args.cold_iters:
        overrides["cold_iters"] = args.cold_iters
    if args.warm_sessions is not None:
        overrides["warm_sessions"] = args.warm_sessions
    if args.skip_cold:
        overrides["skip_cold"] = True
    if args.skip_warm:
        overrides["skip_warm"] = True
    if args.concurrency:
        overrides["concurrency"] = args.concurrency
    if args.concurrency_contexts:
        overrides["concurrency_contexts"] = args.concurrency_contexts
    if args.conc_batches:
        overrides["conc_batches"] = args.conc_batches
    if args.skip_conc:
        overrides["skip_conc"] = True
    if args.skip_multiuser:
        overrides["skip_multiuser"] = True
    if args.multi_users:
        overrides["multi_users"] = args.multi_users
    if args.multi_user_contexts:
        overrides["multi_user_contexts"] = args.multi_user_contexts
    if args.multi_user_warm_sizes:
        overrides["multi_user_warm_sizes"] = args.multi_user_warm_sizes
    if args.multi_user_batches:
        overrides["multi_user_batches"] = args.multi_user_batches
    if args.cold_prefix:
        overrides["cold_prefix"] = args.cold_prefix
    if args.corpus_kind:
        overrides["corpus_kind"] = args.corpus_kind
    if args.no_metrics:
        overrides["metrics_enabled"] = False
    if args.metrics_interval:
        overrides["metrics_interval"] = args.metrics_interval
    cfg.apply_dict(overrides)

    hosts = cfg.hosts
    if args.hosts:
        want = set(args.hosts.split(","))
        hosts = [h for h in hosts if h.name in want or h.url in want]
    if not hosts:
        print(f"[run] no hosts in scenario {scenario_path}. Add a `hosts:` list.")
        return 1

    results_dir = _p(args.results_dir) or cfg.results_dir
    corpus_dir = _p(args.corpus_dir) or cfg.corpus_dir
    store = ResultsStore(results_dir, scenario=cfg.name or "runs")

    # ------------------------------------------------------------- resume
    resume_run_id = args.resume
    if resume_run_id:
        run_row = store.run(resume_run_id)
        if run_row is None:
            print(f"[run] resume: run {resume_run_id} not found in {results_dir}")
            return 1
        stored = run_row.get("config") or {}
        cfg.apply_dict(stored)          # restore sizing + seed from the original run
        tag = run_row.get("tag") or tag
        # resume into the SAME scenario dir the run originally lived in
        store.scenario = run_row.get("scenario") or store.scenario
        hosts = [h for h in hosts if h.name == run_row["host"]]
        if not hosts:
            print(f"[run] resume: host '{run_row['host']}' is not in scenario {scenario_path}")
            return 1
        print(f"[run] resuming {resume_run_id} on {run_row['host']} "
              f"(status was {run_row['status']})")
    else:
        # default: fresh, cache-busting seed per run unless explicitly pinned
        import random as _random
        cfg.seed = args.seed if args.seed is not None else _random.randrange(0x7FFFFFFF)

    print(f"[run] building corpus pool ({cfg.encoding}) ...")
    corpus = Corpus(corpus_dir, encoding=cfg.encoding)
    corpus.load_pool()

    def work(host):
        return run_host(host, cfg, corpus, store, tag=tag, resume_run_id=resume_run_id)

    # daemon threads: a Ctrl-C in the main thread can exit cleanly without
    # waiting for in-flight (possibly 20-min) requests; completed samples are
    # already appended to the run's files and resume re-attempts the rest.
    outcomes: dict[str, Any] = {}
    threads = [
        threading.Thread(
            target=lambda h: outcomes.__setitem__(h.name, work(h)),
            args=(h,), daemon=True, name=h.name,
        )
        for h in hosts
    ]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n[run] Ctrl-C — stopping. Completed samples are already saved.", flush=True)
        print("[run] find the run id with:  uv run llm-bench runs", flush=True)
        print("[run] then resume with:      uv run llm-bench run --scenario "
              f"{scenario_path.name} --resume <RUN_ID>", flush=True)
        return 130

    outcomes = [outcomes[h.name] for h in hosts if h.name in outcomes]

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    for o in outcomes:
        agg = o.summary or {}
        st = agg.get("stats") or {}
        status = o.summary.get("status") if o.summary else "?"
        print(f"\n[host] {o.host.name}  model={o.model}  status={status}  samples={o.n_samples}")
        print(
            f"  total_time_s={o.elapsed_s:,.0f} | ppts={st.get('ppts') or 0:,.0f} "
            f"| tgs={st.get('tgs') or 0:,.1f} | avg_w={st.get('avg_w') or 0:,.1f}"
        )
        for k, v in (agg.get("aggregate") or {}).items():
            phase, _, ctx = k.partition(":")
            ppts = v["prompt_tps"]["median"]
            tgs = v["tg_s"]["median"]
            aw = v["power_w"]["median"]
            print(f"  {phase:<6} {int(ctx):>7,} ctx | ppts {ppts:>8,.0f} | tgs {tgs:>7,.1f} | avg_w {aw:>6,.1f}")
        run_dir = artifact.write_run_artifact(o, cfg, store, store.run_dir(o.run_id))
        print(f"[run] artifact: {run_dir}")

    print("\n[run] per-run data written to "
          f"{results_dir / (cfg.name or 'runs')}")
    print("[run] aggregate across runs with:  uv run llm-bench report")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Build one cross-host report per scenario.

    A scenario is the unit of apples-to-apples hardware comparison (README:
    "self-contained run file" — one model + its hardware sweep), so reports
    are never mixed across scenarios: a `deepseekv4` run and a `qwen3.8` run
    on the same host name are different models and must not land on one
    chart/legend entry together.
    """
    results_dir = _p(args.results_dir) or Config.load().results_dir
    if not results_dir.is_dir():
        print(f"[report] no results under {results_dir}")
        return 1
    scenarios = sorted(
        d.name for d in results_dir.iterdir() if d.is_dir() and d.name != "report"
    )
    if not scenarios:
        print(f"[report] no scenario runs found under {results_dir}")
        return 1
    for sc in scenarios:
        store = ResultsStore(results_dir, scenario=sc)
        if not store.runs():
            continue
        rep_dir = artifact.build_report(store, Config.load(), results_dir / sc)
        print(f"[report] {sc}: {rep_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-bench", description=__doc__)
    p.add_argument("--version", action="version", version=f"llm-bench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="write a scenario template file")
    p_init.add_argument("--scenario", default=None,
                        help="output path (default config/scenarios/example.yaml)")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="benchmark all hosts in a scenario")
    p_run.add_argument("--config", default=None,
                       help="optional base config yaml layered under the scenario")
    p_run.add_argument("--scenario", default=None,
                       help=f"scenario file (default {DEFAULT_SCENARIO_PATH.relative_to(PROJECT_ROOT)})")
    p_run.add_argument("--hosts", default=None, help="comma-separated host names/urls to run")
    p_run.add_argument("--model", default=None, help="override model id")
    p_run.add_argument("--max-tokens", type=int, default=None)
    p_run.add_argument("--contexts", type=int, nargs="+", default=None,
                       help="cold-cache context lengths (tokens)")
    p_run.add_argument("--cold-iters", type=int, default=None)
    p_run.add_argument("--warm-sessions", type=int, default=None)
    p_run.add_argument("--skip-cold", action="store_true")
    p_run.add_argument("--skip-warm", action="store_true")
    p_run.add_argument("--skip-conc", action="store_true", help="skip the concurrency sweep")
    p_run.add_argument("--skip-multiuser", action="store_true", help="skip the multi-user phase")
    p_run.add_argument("--multi-users", type=int, nargs="+", default=None,
                       help="user counts for the multi-user phase (e.g. 2 4 8)")
    p_run.add_argument("--multi-user-contexts", type=int, nargs="+", default=None,
                       help="per-user cold contexts for the multi-user phase")
    p_run.add_argument("--multi-user-warm-sizes", type=int, nargs="+", default=None,
                       help="per-user warm steps for the multi-user phase")
    p_run.add_argument("--multi-user-batches", type=int, default=None)
    p_run.add_argument("--concurrency", type=int, nargs="+", default=None,
                       help="concurrent-stream levels to sweep")
    p_run.add_argument("--concurrency-contexts", type=int, nargs="+", default=None,
                       help="context lengths for the concurrency sweep (<=100k)")
    p_run.add_argument("--conc-batches", type=int, default=None)
    p_run.add_argument("--cold-prefix", choices=["random", "deterministic"], default=None)
    p_run.add_argument("--corpus-kind", choices=["code", "writing", "mixed"], default=None)
    p_run.add_argument("--seed", type=int, default=None,
                       help="reproducible corpus seed; defaults to a random per-run seed")
    p_run.add_argument("--resume", default=None,
                       help="run_id to resume: re-runs only the failed/remaining items of that run")
    p_run.add_argument("--no-metrics", action="store_true",
                       help="disable hardware prometheus monitoring (macmon/dcgm)")
    p_run.add_argument("--metrics-interval", type=float, default=None,
                       help="seconds between hardware-exporter polls")
    p_run.add_argument("--tag", default="", help="extra label appended to run id")
    p_run.add_argument("--results-dir", default=None,
                       help="root for per-run results (default results/)")
    p_run.add_argument("--corpus-dir", default=None)
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="post-process: aggregate all runs into a cross-host report")
    p_rep.add_argument("--results-dir", default=None)
    p_rep.set_defaults(func=cmd_report)

    p_runs = sub.add_parser("runs", help="list runs (find a run_id to resume)")
    p_runs.add_argument("--results-dir", default=None)
    p_runs.set_defaults(func=cmd_runs)

    return p


def cmd_runs(args: argparse.Namespace) -> int:
    store = ResultsStore(_p(args.results_dir) or Config.load().results_dir)
    print(f"{"run_id":<46} {"host":<10} {"model":<42} {"status":<8} {"samples":>7} {"ok":>4}")
    print("-" * 128)
    for r in reversed(store.runs()):
        ss = store.samples(r["run_id"])
        ok = sum(1 for s in ss if s["ok"])
        print(f"{r['run_id']:<46} {r['host']:<10} {str(r['model'] or '?'):<42} "
              f"{r['status']:<8} {len(ss):>7} {ok:>4}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
