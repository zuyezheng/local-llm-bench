"""Collate every sample across all scenarios/runs into a tidy dataset.

Loads run.json + samples.jsonl from results/<scenario>/<run_id>/ and writes a
consolidated CSV + a summary of run coverage (which runs exist, status, counts).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main() -> None:
    runs: list[dict] = []
    samples: list[dict] = []

    for scenario_dir in sorted(ROOT.iterdir()):
        if not scenario_dir.is_dir() or scenario_dir.name == "report":
            continue
        for run_dir in sorted(scenario_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run = read_json(run_dir / "run.json")
            if not run:
                continue
            run["scenario"] = scenario_dir.name
            runs.append(run)
            for s in read_jsonl(run_dir / "samples.jsonl"):
                s["scenario"] = scenario_dir.name
                s["run_id"] = run["run_id"]
                s["host"] = run["host"]
                s["model"] = run["model"]
                s["run_status"] = run.get("status", "")
                samples.append(s)

    runs.sort(key=lambda r: (r["scenario"], r["started_at"] or ""))
    samples.sort(key=lambda s: (s["scenario"], s["run_id"], s["ts"] or ""))

    # ---- run coverage ----
    cov = []
    for r in runs:
        n = sum(1 for s in samples if s["run_id"] == r["run_id"])
        ok_n = sum(1 for s in samples if s["run_id"] == r["run_id"] and s.get("ok"))
        phases = {}
        for s in samples:
            if s["run_id"] == r["run_id"] and s.get("ok"):
                phases[s["phase"]] = phases.get(s["phase"], 0) + 1
        cov.append({
            "scenario": r["scenario"],
            "run_id": r["run_id"],
            "host": r["host"],
            "model": r["model"],
            "status": r.get("status"),
            "samples_total": n,
            "samples_ok": ok_n,
            "phases": " ".join(f"{k}={v}" for k, v in sorted(phases.items())),
            "started": (r.get("started_at") or "")[:19],
        })

    # ---- tidy samples ----
    keep_cols = [
        "scenario", "run_id", "host", "model", "run_status",
        "phase", "step", "iter", "session_pos", "batch",
        "context_tokens", "prompt_tokens", "new_prompt_tokens", "max_tokens",
        "ttft_ms", "prompt_tps", "tpot_ms", "tg_ms", "tg_tokens",
        "reasoning_tokens", "content_tokens", "tg_s", "out_tps",
        "power_w", "wh", "tps_per_w", "total_ms", "ok", "error", "finish_reason",
    ]
    with (OUT / "samples_tidy.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep_cols, extrasaction="ignore")
        w.writeheader()
        for s in samples:
            w.writerow({k: s.get(k, "") for k in keep_cols})

    with (OUT / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov[0].keys()) if cov else ["scenario"])
        w.writeheader()
        w.writerows(cov)

    print(f"scenarios: {len(runs and [r['scenario'] for r in runs])} runs across "
          f"{len({r['scenario'] for r in runs})} scenarios")
    print(f"total samples: {len(samples)}")
    print()
    for row in cov:
        print(f"{row['scenario']:<14} {row['host']:<14} {row['status']:<8} "
              f"ok={row['samples_ok']:<4} {row['phases']}")


if __name__ == "__main__":
    sys.exit(main())
