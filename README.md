# llm-bench

Benchmark the **same open-weight model across heterogeneous hardware** — DGX Spark / Grace-Blackwell (vLLM) to Apple M3 Ultra (MLX/omlx) — with realistic corpora and a durable, chartable artifact per run.

Current target model: **DeepSeek V4 Flash** (DeepSeek-V4-Flash-0731 on gx10-top, DeepSeek-V4-Flash-4bit MLX on macstudio). Latest scenario added: **Qwen3.8-Flash-Next** (`config/scenarios/qwen3.8-next.yaml` — macstudio/GGUF, gx10-top/sglang, tr-pro-6000/vLLM), analysed in `analysis/ANALYSIS.md` §1b and in `analysis/site/index.html`.

## What it measures

Every request is streamed and timed per-token:

| metric | meaning |
|---|---|
| **TTFT** | time to first output token (ms) — prompt processing latency |
| **PP** | prompt processing throughput = prompt tokens / TTFT (tok/s) |
| **TPOT** | mean time per output token (ms) |
| **TG** | generation throughput = output tokens / generation time (tok/s) |
| **tg_s** | (output + reasoning tokens) / generation time (tok/s) |
| **power_w** | average watts over the sample — cluster-summed, divided by users for concurrent phases |
| **wh** | total watt-hours for the sample (`power_w × duration / 3600`) |
| **tps_per_w** | generation throughput per watt (`out_tps / power_w`) |

The model (`DeepSeek-V4-Flash`) is a **reasoning model**, so both `content` and `reasoning` deltas count as generated tokens (vLLM streams `delta.reasoning`).

## Scenarios (run/tag config files)

A scenario is a **self-contained run file**: it bundles the machine config
(hosts + metric exporters), the model, the workload sizing, and the run tag —
so each (model, machines, workload) combo is reproducible from one file.

```bash
uv run llm-bench run --scenario config/scenarios/deepseekv4.yaml
```

```yaml
# config/scenarios/deepseekv4.yaml
name: deepseekv4
tag: deepseekv4
max_tokens: 1024
cold_iters: 1

# --- machines (hosts + exporters are part of the scenario) ---
hosts:
  - name: mac-host
    url: http://<mac-host>:8000
    model: DeepSeek-V4-Flash-4bit
    metrics:
      - {type: macmon, urls: [http://<mac-host>:9101/metrics]}
  - name: gpu-host
    url: http://<linux-host>:8000
    model: deepseek-ai/DeepSeek-V4-Flash-0731
    metrics:
      - type: dcgm
        urls:                      # multiple URLs of one type = cluster nodes,
          - http://<linux-host>:9400/metrics    #   summed power/utilization
          - http://<linux-host-2>:9400/metrics

# --- workload ---
context_lengths: [1000, 10000, 50000, 100000, 200000, 400000]   # single-user cold
warm_sizes: [1000, 10000, 20000, ..., 200000]                    # single-user warm (+10k/step)
multi_users: [2, 4, 8]                                           # each user = distinct context
multi_user_contexts: [1000, 10000, 50000, 100000]
multi_user_warm_sizes: [1000, 10000, 20000, 30000, 40000, 50000]
```

When a scenario defines `hosts`, they are the machine config for
that run. To test another model on different machines, copy the scenario file
and edit `hosts` + `model` + sizing.

The checked-in `deepseekv4.yaml`/`qwen3.*.yaml` scenarios ship **without** a
`hosts:` block on purpose — real machine URLs are environment-specific and
don't belong in a public repo. Put them in a local, gitignored file (anything
under `config/local/`) and layer it under the scenario with `--config`:

```bash
uv run llm-bench run --scenario config/scenarios/deepseekv4.yaml \
  --config config/local/deepseekv4.hosts.yaml
```

See [`config/scenarios/example-mac.yaml`](config/scenarios/example-mac.yaml)
and [`config/scenarios/example-linux.yaml`](config/scenarios/example-linux.yaml)
for complete, runnable single-host templates (MLX/macmon and vLLM/dcgm
respectively) to copy into your local override or a new scenario.

## Scenarios (workloads)

- **Cold cache** — a fresh, random real-world slice at each context length (default `1000, 10k, 50k, 100k, 200k, 500k`), repeated `cold_iters` times. `cold_prefix: random` busts the prefix cache; `deterministic` reuses the same slice for reproducible A/B.
- **Warm cache / coding session** — one contiguous corpus slice whose prefix grows in **10k-token steps up to 200k** (`10k → 20k → … → 200k`). Each turn shares a byte-identical prefix with the previous one, exactly like a coding session where you keep pasting more of the project. Prefix-caching servers (vLLM) can exploit this.
- **Concurrency** — multi-user phases fire N concurrent streams (N = `multi_users`), each with a **distinct** context slice; the fixed 1–N concurrency sweep (`concurrency`) runs only when `multi_users` is absent, since multi-user already measures concurrent throughput. Reports aggregate generation throughput across the batch, per-request TTFT/latency, and cluster power under load.

## Corpora

Real code (pandas, django, numpy, pytorch, rustc, linux, llvm, TypeScript lib.dom.d.ts, …) and public-domain books (Gutenberg), tokenized once with `tiktoken` (`o200k_base`) into a cached pool; prompts are **contiguous slices of real text** sized to hit the target context, not random token strings. Mixed prompts are 70% code / 30% writing by default.

## Usage

```bash
uv sync                      # install deps
uv run llm-bench run         # benchmark using config/scenarios/deepseekv4.yaml
uv run llm-bench report      # post-process: aggregate all runs into results/report/
uv run llm-bench runs        # list runs (run_id, host, status, sample counts)
uv run llm-bench init        # write config/scenarios/example.yaml template
```

## Resuming partial runs

If a run is interrupted — crash, manual stop, or a server aborting a long
prefill (the omlx server occasionally drops >200k prefill requests) — the
completed samples are already persisted on disk under
`results/<scenario>/<run_id>/`. Resume it to **skip completed items and only
run the failed/remaining ones**, appending to the same run:

```bash
uv run llm-bench run --scenario config/scenarios/deepseekv4.yaml --resume <RUN_ID>
```

- `llm-bench runs` shows run IDs to resume.
- Prompts are derived deterministically from `(host, seed, phase, step, iter, batch)`, so a resumed warm session keeps the same growing prefix and repeated `resume` runs never duplicate completed work.
- Requests that stall or abort are cut short by a per-request TTFT read-timeout budget (`prompt_tokens / min_pp_tps + ttft_margin`), retried once, then marked failed — `resume` re-attempts them.
- Seeds default to a random value per run (cache-busting across runs); pass `--seed N` for a reproducible corpus slice.

Common overrides:

```bash
uv run llm-bench run \
  --hosts gx10-top,macstudio \
  --contexts 1000 10000 100000 \
  --cold-iters 3 \
  --warm-sessions 1 \
  --max-tokens 1024 \
  --tag ds-v4-0829
```

For a quick sanity check before a long run:

```bash
uv run llm-bench run --contexts 1000 --cold-iters 1 --max-tokens 64 --warm-sessions 0
```

### Storage

No database. Each host run writes a self-contained directory, incrementally,
under `results/<scenario>/<run_id>/`:

- `samples.jsonl` — one JSON object per request, appended as each sample completes
- `metrics.jsonl` — one JSON object per hardware-exporter poll, appended live
- `run.json` — run metadata (host, model, status, config, start/end), machine
  spec (ssh + Prometheus/vLLM gauges when available), model spec, and aggregated
  results (median/p90/min/max of TTFT, PP, TPOT, TG per scenario), errors
- `samples.csv`, `metrics.csv` — derived flat copies
- `charts/cold_perf.png`, `charts/warm_session.png`, `charts/metrics_timeseries.png`

Because every run only ever appends to its own files, concurrent benchmarks on
different hosts never touch shared state — no locking and no DB. `uv run llm-bench report`
is the post-processing phase: it scans all run directories and renders the
cross-host comparison into `results/report/`.

## Hardware monitoring (GPU / CPU / RAM / power / efficiency)

While each host is benchmarked, a background sampler polls Prometheus exporters
every `metrics_interval` (default 2s) and appends a time-series to the run's
`metrics.jsonl` (and, in the report phase, into `metrics.csv` + `charts/metrics_timeseries.png`). Each request's
time window is correlated with the power series to derive:

| metric | meaning |
|---|---|
| `power_w` | median power draw during the request window |
| `energy_ws` | power × request duration (Joules) |
| `j_per_token` | energy per generated token |
| `tps_per_w` | generation throughput per watt |

Configured per host in the scenario file:

```yaml
hosts:
  - name: mac-host
    url: http://<mac-host>:8000
    metrics:
      - {type: macmon, urls: [http://<mac-host>:9101/metrics]}
  - name: gpu-host
    url: http://<linux-host>:8000
    metrics:
      - type: dcgm            # multiple urls of one type = one cluster
        urls:                 #   whose power/utilization is SUMMED
          - http://<linux-host>:9400/metrics
          - http://<linux-host-2>:9400/metrics
        # gpu_model: "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
        #             # restrict to one device model on mixed-GPU hosts
```

- **macmon** — CPU/GPU/ANE/RAM power + utilization + temperature, and
  `macmon_sys_power_watts` (used for efficiency). Also auto-surfaces the chip
  spec (`Apple_M3_Ultra`, RAM size) into the artifact's machine spec.
- **dcgm** — per-GPU utilization, memory copy util, power, temperature, SM clock,
  and cumulative energy (`DCGM_FI_DEV_*`). GPU model (`NVIDIA GB10`) is pulled
  into the machine spec automatically. On hosts with mixed devices, add
  `gpu_model: "<modelName>"` to restrict collection to one device model (the
  exporter's `modelName` label); multiple devices of that model still aggregate.
- `type: auto` sniffs which exporter family the endpoint serves.
- **Clusters** — several URLs under one `metrics` entry are cluster nodes whose
  power/utilization is summed for that host (per-node rows in `metrics.csv`,
  summed power line in `metrics_timeseries.png`, `nodes:` count in efficiency).

Efficiency is computed per (phase, context): median `power_w`, `j_per_token`,
and `tps_per_w`, reported in `run.json` and the cross-host `report.md`. Note the
power key differs per platform (`gpu_power_w` for dcgm vs `total_power_w` for
macmon); the report prints which key each run used so comparisons stay honest.

Disable or tune with `--no-metrics` or `--metrics-interval 1`.

## Config

Scenario files under `config/scenarios/` bundle hosts, per-host model id,
context lengths, warm sizes, and sampling behavior. See `bench/config.py` for
every knob. Set `ssh:` on a host (e.g. `user@<mac-host>`) to also capture
server-side hardware specs (`nvidia-smi`, lspci, mem).

The checked-in scenarios (`deepseekv4.yaml`, `qwen3.*.yaml`) intentionally omit
`hosts:` — real machine URLs are environment-specific and shouldn't live in a
public repo. Keep yours in a gitignored file under `config/local/` and layer
it in with `uv run llm-bench run --scenario <scenario>.yaml --config config/local/<name>.hosts.yaml`.
`config/scenarios/example-mac.yaml` and `config/scenarios/example-linux.yaml`
are complete, runnable single-host templates for each platform.
