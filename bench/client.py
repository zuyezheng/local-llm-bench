"""Streaming OpenAI-compatible chat client with precise TTFT / per-token timing.

We use SSE streaming and timestamp every token so we can report:

  * ttft_ms        — time to first token (prompt processing latency)
  * prompt_tps     — input tokens / TTFT (prompt processing throughput)
  * tpot_ms        — mean time per output token
  * tg_ms / out_tps — end-to-end generation time / output tokens per second

The client is written for heterogeneous backends (vLLM, MLX/omlx, LM Studio,
etc.). Some servers reject `ignore_eos` / `stream_options`; we detect that and
fall back to a plain request automatically.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx


class ClientError(Exception):
    def __init__(self, message: str, status: int = 0, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class Result:
    ok: bool
    ttft_ms: float = 0.0
    prompt_tps: float = 0.0
    tpot_ms: float = 0.0
    tg_ms: float = 0.0
    tg_tokens: int = 0
    out_tps: float = 0.0
    prompt_tokens: int = 0
    total_ms: float = 0.0
    inter_token_ms: list[float] = field(default_factory=list)
    reasoning_tokens: int = 0
    error: str = ""
    status: int = 0
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    extra_params_used: bool = False


class CompletionsClient:
    """One instance per host; internally serial so timing is not polluted."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 3600.0,
        read_timeout: float = 1800.0,
        connect_timeout: float = 15.0,
        max_retries: int = 1,
        backoff: float = 2.0,
        use_extra_params: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._connect_timeout = connect_timeout
        # Separate timeouts so a stalled stream is retried quickly while still
        # allowing long prefill silence (up to ~30 min on 500k contexts).
        self._http = httpx.Client(
            timeout=httpx.Timeout(
                timeout=timeout,
                connect=connect_timeout,
                read=read_timeout,
                write=60.0,
                pool=connect_timeout,
            ),
            follow_redirects=True,
        )
        self.extra_params = use_extra_params  # ignore_eos / stream_options

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float = 0.0,
        seed: int | None = None,
        read_timeout: float | None = None,
    ) -> Result:
        # A global deadline bounds total wall time across retries so a stalled
        # request cannot consume hours.
        deadline = time.monotonic() + self.timeout
        attempts = 0
        while True:
            try:
                return self._chat_once(model, messages, max_tokens, temperature, seed, read_timeout)
            except ClientError as e:
                if e.retryable and attempts < self.max_retries and time.monotonic() < deadline:
                    attempts += 1
                    time.sleep(min(self.backoff * attempts, max(1.0, deadline - time.monotonic())))
                    continue
                return Result(
                    ok=False,
                    error=str(e),
                    status=e.status,
                )

    def _chat_once(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        seed: int | None,
        read_timeout: float | None,
    ) -> Result:
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if self.extra_params:
            # vLLM supports both; include_usage gives us exact prompt tokens.
            payload["stream_options"] = {"include_usage": True}
            payload["ignore_eos"] = True
        if seed is not None:
            payload["seed"] = seed

        t0 = time.monotonic()
        # Per-request read timeout: lets the caller budget TTFT by prompt size so
        # a stalled stream is cut short instead of waiting the full 30 min.
        if read_timeout is not None:
            tout = httpx.Timeout(
                timeout=self.timeout,
                connect=self._connect_timeout,
                read=read_timeout,
                write=60.0,
                pool=self._connect_timeout,
            )
        else:
            tout = self._http.timeout
        try:
            with self._http.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload, timeout=tout
            ) as r:
                if r.status_code != 200:
                    body = r.read().decode("utf-8", "replace")[:2000]
                    if (
                        r.status_code in (400, 422)
                        and self.extra_params
                        and _looks_like_bad_param(body)
                    ):
                        self.extra_params = False
                        raise ClientError(
                            "extra params rejected; retrying plain", status=r.status_code
                        )
                    retryable = r.status_code in (408, 429, 500, 502, 503, 504)
                    raise ClientError(
                        f"HTTP {r.status_code}: {body}",
                        status=r.status_code,
                        retryable=retryable,
                    )

                ttft_ms: float | None = None
                tok_times: list[float] = []
                usage: dict = {}
                finish_reason = ""
                reasoning_tokens = 0
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        # server reported an abort/error mid-stream (e.g. omlx
                        # killing a long prefill) — surface it as an API error
                        raise ClientError(
                            f"server error: {chunk['error']}", status=500, retryable=True,
                        )
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        # DeepSeek-style reasoning models stream under `reasoning`
                        # (vLLM) or `reasoning_content` (some gateways); both are
                        # generated tokens and the first one defines TTFT.
                        content = delta.get("content")
                        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                        piece = content or reasoning
                        if piece:
                            now = time.monotonic()
                            if ttft_ms is None:
                                ttft_ms = (now - t0) * 1000.0
                            tok_times.append((now - t0) * 1000.0)
                            if reasoning:
                                reasoning_tokens += 1
                        if ch.get("finish_reason"):
                            finish_reason = ch.get("finish_reason")
                t_end = time.monotonic()
        except httpx.TimeoutException as e:
            raise ClientError(f"timeout: {e}", retryable=True)
        except httpx.HTTPError as e:
            raise ClientError(str(e), retryable=True)

        total_ms = (t_end - t0) * 1000.0
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        tg_tokens = int(usage.get("completion_tokens", 0) or len(tok_times) or 0)
        if tg_tokens == 0:
            # Stream ended with no output tokens (server aborted prefill /
            # dropped the connection mid-flight). Treat as a failure so the
            # sample is not recorded as a bogus success and resume retries it.
            return Result(
                ok=False,
                error="stream ended with no output tokens (server abort?)",
                total_ms=total_ms,
            )
        tg_ms = max(total_ms - (ttft_ms or 0.0), 0.0)
        tpot_ms = tg_ms / tg_tokens if tg_tokens else 0.0
        out_tps = tg_tokens / (tg_ms / 1000.0) if tg_ms > 0 else 0.0
        prompt_tps = prompt_tokens / ((ttft_ms or 0.0) / 1000.0) if ttft_ms else 0.0

        return Result(
            ok=True,
            ttft_ms=ttft_ms or 0.0,
            prompt_tps=prompt_tps,
            tpot_ms=tpot_ms,
            tg_ms=tg_ms,
            tg_tokens=tg_tokens,
            out_tps=out_tps,
            prompt_tokens=prompt_tokens,
            total_ms=total_ms,
            inter_token_ms=tok_times,
            reasoning_tokens=reasoning_tokens,
            status=200,
            finish_reason=finish_reason,
            usage=usage,
            extra_params_used=self.extra_params,
        )


def _looks_like_bad_param(body: str) -> bool:
    lowered = body.lower()
    needles = (
        "stream_options",
        "ignore_eos",
        "unexpected field",
        "unknown parameter",
        "extra_for_body",
        "additional properties",
        "unexpected keyword",
    )
    return any(n in lowered for n in needles)
