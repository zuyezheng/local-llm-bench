"""Real-world corpora (code + writing) and tiktoken-based prompt sizing.

We download real source files and public-domain books once, tokenize them
with a tiktoken encoding, cache the token pool, and then select contiguous
slices of tokens sized to hit a target context length. Decoded slices are the
prompts sent to the server, so benchmarks measure real prose/code instead of
random token strings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import tiktoken

DEFAULT_CODE_URLS = [
    ("pandas_frame.py", "https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/core/frame.py"),
    ("django_query.py", "https://raw.githubusercontent.com/django/django/main/django/db/models/query.py"),
    ("numpy_funcbase.py", "https://raw.githubusercontent.com/numpy/numpy/main/numpy/lib/_function_base_impl.py"),
    ("pytorch_linear.py", "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/modules/linear.py"),
    ("rustc_middle.rs", "https://raw.githubusercontent.com/rust-lang/rust/master/compiler/rustc_middle/src/ty/mod.rs"),
    ("rustc_context.rs", "https://raw.githubusercontent.com/rust-lang/rust/master/compiler/rustc_middle/src/ty/context.rs"),
    ("git_pack_objects.c", "https://raw.githubusercontent.com/git/git/master/builtin/pack-objects.c"),
    ("llvm_seldag.cpp", "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/CodeGen/SelectionDAG/SelectionDAGISel.cpp"),
    ("kubernetes_scheduler.go", "https://raw.githubusercontent.com/kubernetes/kubernetes/master/pkg/scheduler/schedule_one.go"),
    ("cpython_listobject.c", "https://raw.githubusercontent.com/python/cpython/main/Objects/listobject.c"),
    ("scipy_continuous_distns.py", "https://raw.githubusercontent.com/scipy/scipy/main/scipy/stats/_continuous_distns.py"),
    ("torch_tensor.py", "https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_tensor.py"),
    ("transformers_modeling_utils.py", "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_utils.py"),
    ("cpython_ceval.c", "https://raw.githubusercontent.com/python/cpython/main/Python/ceval.c"),
    ("typescript_lib_dom.d.ts", "https://raw.githubusercontent.com/microsoft/TypeScript/main/tsc/internal/bundled/libs/lib.dom.d.ts"),
    ("typescript_checker.go", "https://raw.githubusercontent.com/microsoft/TypeScript/main/tsc/internal/checker/checker.go"),
    ("linux_bnxt.c", "https://raw.githubusercontent.com/torvalds/linux/master/drivers/net/ethernet/broadcom/bnxt/bnxt.c"),
    ("postgres_planner.c", "https://raw.githubusercontent.com/postgres/postgres/master/src/backend/optimizer/plan/planner.c"),
    ("wireshark_http2.c", "https://raw.githubusercontent.com/wireshark/wireshark/master/epan/dissectors/packet-http2.c"),
    ("llvm_aarch64_dag.cpp", "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AArch64/AArch64ISelDAGToDAG.cpp"),
]

DEFAULT_WRITING_URLS = [
    ("war_and_peace.txt", "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"),
    ("moby_dick.txt", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    ("ulysses.txt", "https://www.gutenberg.org/cache/epub/4300/pg4300.txt"),
    ("les_miserables.txt", "https://www.gutenberg.org/cache/epub/135/pg135.txt"),
    ("brothers_karamazov.txt", "https://www.gutenberg.org/cache/epub/28054/pg28054.txt"),
    ("tale_of_two_cities.txt", "https://www.gutenberg.org/cache/epub/98/pg98.txt"),
]

KIND_WRITING = 0
KIND_CODE = 1
KIND_NAME = {KIND_WRITING: "writing", KIND_CODE: "code"}


@dataclass
class Pool:
    tokens: np.ndarray          # (N,) uint32 token ids
    offsets: np.ndarray         # (M+1,) boundaries of each file's slice
    kinds: np.ndarray           # (M,) kind per file
    names: list[str]
    encoding: str

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.shape[0])

    def kind_ranges(self, kind: int) -> list[tuple[int, int]]:
        """[start, end) token ranges belonging to a kind."""
        out = []
        for i, k in enumerate(self.kinds):
            if k == kind:
                out.append((int(self.offsets[i]), int(self.offsets[i + 1])))
        return out


class Corpus:
    """Download, cache, and sample real-world corpora."""

    def __init__(
        self,
        corpus_dir: Path,
        encoding: str = "o200k_base",
        code_urls: Optional[list[str]] = None,
        writing_urls: Optional[list[str]] = None,
    ) -> None:
        self.corpus_dir = corpus_dir
        self.raw_dir = corpus_dir / "raw"
        self.pool_path = corpus_dir / f"pool_{encoding}.npz"
        self.encoding_name = encoding
        self.enc = tiktoken.get_encoding(encoding)
        self.code_urls = code_urls or [u for _, u in DEFAULT_CODE_URLS]
        self.writing_urls = writing_urls or [u for _, u in DEFAULT_WRITING_URLS]
        self._pool: Optional[Pool] = None

    # ------------------------------------------------------------------ fetch

    def fetch(self) -> int:
        """Download any missing corpus files. Returns number of files fetched.

        Individual failures are warned about and skipped so one dead upstream
        link cannot block the whole corpus build.
        """
        sources = [(KIND_CODE, n, u) for n, u in DEFAULT_CODE_URLS] + [
            (KIND_WRITING, n, u) for n, u in DEFAULT_WRITING_URLS
        ]
        got = 0
        for kind, name, url in sources:
            dest = self.raw_dir / KIND_NAME[kind] / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"[corpus] fetching {name} ...", flush=True)
            try:
                with httpx.Client(follow_redirects=True, timeout=180.0) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
            except Exception as e:  # noqa: BLE001
                print(f"[corpus] WARN failed {name}: {e}", flush=True)
                continue
            got += 1
        return got

    # ------------------------------------------------------------------- pool

    def build_pool(self, force: bool = False) -> Pool:
        if self._pool is not None and not force:
            return self._pool
        if self.pool_path.exists() and not force:
            with np.load(self.pool_path, allow_pickle=False) as z:
                tokens = z["tokens"]
                offsets = z["offsets"]
                kinds = z["kinds"]
                names = [str(n) for n in z["names"].tolist()] if "names" in z else []
            self._pool = Pool(tokens, offsets, kinds, names, self.encoding_name)
            return self._pool

        if not self.raw_dir.exists():
            self.fetch()

        chunks: list[np.ndarray] = []
        offsets: list[int] = [0]
        kinds: list[int] = []
        names: list[str] = []
        for kind, dirname in ((KIND_WRITING, "writing"), (KIND_CODE, "code")):
            d = self.raw_dir / dirname
            if not d.exists():
                continue
            for path in sorted(d.iterdir()):
                text = path.read_text(encoding="utf-8", errors="replace")
                ids = self.enc.encode(text)
                if not ids:
                    continue
                chunks.append(np.asarray(ids, dtype=np.uint32))
                kinds.append(kind)
                names.append(path.name)
                offsets.append(offsets[-1] + len(ids))

        tokens = np.concatenate(chunks).astype(np.uint32)
        pool = Pool(
            tokens=tokens,
            offsets=np.asarray(offsets, dtype=np.int64),
            kinds=np.asarray(kinds, dtype=np.int64),
            names=names,
            encoding=self.encoding_name,
        )
        self._pool = pool
        np.savez(
            self.pool_path,
            tokens=tokens,
            offsets=offsets,
            kinds=kinds,
            names=np.asarray(names),
        )
        print(
            f"[corpus] pool {self.encoding_name}: {pool.n_tokens:,} tokens "
            f"across {len(names)} files"
        )
        return pool

    def load_pool(self) -> Pool:
        return self.build_pool()

    # --------------------------------------------------------------- sampling

    def select_text(
        self,
        n_tokens: int,
        rng: np.random.Generator,
        kind: str = "mixed",
        code_ratio: float = 0.7,
    ) -> str:
        """Return a decoded text slice of ~n_tokens tokens of the given kind."""
        pool = self.load_pool()
        kind_id = {"code": KIND_CODE, "writing": KIND_WRITING}.get(kind)
        if kind_id is None:  # mixed
            code_ids = pool.kinds == KIND_CODE
            total_code = int(code_ids.sum())
            n_code = int(round(n_tokens * code_ratio))
            n_code = max(0, min(n_code, total_code - 1))
            n_writing = n_tokens - n_code
            piece_code = self._slice_kind(pool, KIND_CODE, n_code, rng)
            piece_writing = self._slice_kind(pool, KIND_WRITING, n_writing, rng)
            return (piece_code + piece_writing) if n_code else piece_writing

        return self._slice_kind(pool, kind_id, n_tokens, rng)

    def _slice_kind(
        self,
        pool: Pool, kind: int, n_tokens: int, rng: np.random.Generator,
    ) -> str:
        ranges = pool.kind_ranges(kind)
        if n_tokens <= 0 or not ranges:
            return ""
        # pick a random start within a kind region with enough room
        region = ranges[rng.integers(0, len(ranges))]
        size = region[1] - region[0]
        if size <= n_tokens:
            start = region[0]
            end = region[1]
        else:
            start = int(region[0] + rng.integers(0, size - n_tokens + 1))
            end = start + n_tokens
        ids = pool.tokens[start:end]
        text = self.enc.decode(ids.tolist())
        # safety: never exceed the intended context
        text = text[: math.ceil(n_tokens * 24)]
        return text

    def build_prefixes(
        self,
        n_max: int,
        steps: list[int],
        rng: np.random.Generator,
        kind: str = "code",
    ) -> list[tuple[int, str]]:
        """Decoded contiguous prefixes for progressive warm-cache sessions.

        Returns [(step_tokens, text), ...] where each text is the token-level
        prefix of every larger step, so consecutive turns share a byte-identical
        prefix — the property a real coding session has and that prefix-caching
        servers (vLLM) exploit.
        """
        pool = self.load_pool()
        kind_id = {"code": KIND_CODE, "writing": KIND_WRITING}[kind]
        ranges = pool.kind_ranges(kind_id)
        if not ranges:
            raise RuntimeError(f"no '{kind}' corpus available for warm sessions")
        # Use the largest region so every warm step fits and prefixes stay as
        # long as possible (realistic large-project coding session).
        region = max(ranges, key=lambda r: r[1] - r[0])
        size = region[1] - region[0]
        if size < n_max:
            start = int(region[0])
            n_max = size
        else:
            start = int(region[0] + rng.integers(0, size - n_max + 1))
        ids = pool.tokens[start : start + n_max].tolist()
        out: list[tuple[int, str]] = []
        for st in sorted(set(steps)):
            if st <= n_max:
                out.append((st, self.enc.decode(ids[:st])))
        return out

    def build_user_prefixes(
        self,
        n_users: int,
        n_max: int,
        steps: list[int],
        rng: np.random.Generator,
        kind: str = "code",
    ) -> list[list[tuple[int, str]]]:
        """Per-user progressive warm bases for multi-user sessions.

        Each of the ``n_users`` sessions gets a DIFFERENT underlying slice
        (different corpus region / offset), so users never share prefixes.
        Returns ``n_users`` lists of [(step_tokens, decoded_text), ...].
        """
        pool = self.load_pool()
        kind_id = {"code": KIND_CODE, "writing": KIND_WRITING}[kind]
        ranges = sorted(pool.kind_ranges(kind_id), key=lambda r: r[1] - r[0], reverse=True)
        if not ranges:
            raise RuntimeError(f"no '{kind}' corpus available for multi-user sessions")

        user_bases: list[list[tuple[int, str]]] = []
        used_starts: set[int] = set()
        for i in range(n_users):
            region = ranges[i % len(ranges)]
            size = region[1] - region[0]
            nmax = min(n_max, size)
            max_start = size - nmax
            start = int(region[0])
            if max_start > 0:
                slot = i // len(ranges)
                for _ in range(max_start + 1):
                    cand = int(region[0] + (slot % (max_start + 1)))
                    slot += 1
                    if cand not in used_starts:
                        start = cand
                        break
            used_starts.add(start)
            ids = pool.tokens[start : start + nmax].tolist()
            user_bases.append(
                [(st, self.enc.decode(ids[:st])) for st in sorted(set(steps)) if st <= nmax]
            )
        return user_bases
