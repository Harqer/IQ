# IQ Local Streaming Inference (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the IQ checkpoint `glm53-full78-local-20261025a` (539.6 GB in HF bucket `kofecheckmate/iq-glm53-all256`) on the local N95 machine via three-tier streaming (RAM → NVMe cache → network bucket fetch), producing greedy generations through an OpenAI-compatible HTTP server, with numerics validated against a billed H200 parity fixture.

**Architecture:** A streaming weight provider materializes per-layer tensors from the checkpoint's safetensors index and bucket files; non-expert shards are cached whole on NVMe (LRU, 44 GB budget), routed experts are int4-quantized on first fetch (group-64 symmetric) and dequantized on demand. The model runs on CPU using the IQ repo's own device-agnostic layer classes, with a new CPU Mamba-3 MIMO runtime (`mamba3_cpu.py`) ported from upstream `state-spaces/mamba @ e9594ce1c732d97440f0332fdc43170a2294dbfa` (decode math taken from upstream's own PyTorch reference implementations; prefill = looped decode steps). Routed experts are computed functionally (no parameter copies). Production CUDA files are never modified.

**Tech Stack:** Python 3.11+ venv, torch (CPU build) >= 2.10, transformers 5.17.0, safetensors, huggingface_hub 1.14 (native bucket Python API), pytest 9.

**Spec:** `/home/shaolin/IQ/docs/IQ_LOCAL_STREAMING_INFERENCE_DESIGN.md` (approved 2026-11-15). Read it before executing this plan.

## Global Constraints

- All work happens in `/home/shaolin/IQ` on the existing branch. Do not touch `/home/shaolin/colibri` (Phase 2 only).
- HF token comes from environment variable `HF_TOKEN` only. Never print it, never write it to any file, never put it in `.env*`, never commit it.
- No edits to production CUDA-gated files: `iq_model/state/mamba3.py` stays CUDA-only; `IQHybridForCausalLM.__init__`'s CUDA check stays. The CPU runtime lives in new modules (`iq_model/state/mamba3_cpu.py`, `iq_local/`).
- The CPU runtime must not import `mamba_ssm`, `tilelang`, or any CUDA-only package. Only torch/stdlib/transformers/safetensors/huggingface_hub.
- Upstream source of truth for Mamba-3 math: commit `e9594ce1c732d97440f0332fdc43170a2294dbfa` of `state-spaces/mamba`. Cite that commit in every ported function's docstring.
- Model constants (from run `iq_config.json`): vocab 128256, hidden 4096, 235 layers, 79 Mamba-3 + 78 CSA/HCA + 78 MoE, latent 2048, 256 experts top-8, experts `[2048, 2048]` BF16, tokenizer = Llama-3.1 (`unsloth/Meta-Llama-3.1-8B`).
- Checkpoint key-set contract: the run's `model.safetensors.index.json` `weight_map` (63,164 tensors / 20,283 files) is the exact state-dict key oracle. The CPU model's `state_dict()` key set must equal it.
- Cache budget: single 44 GB NVMe cache for everything cached (non-expert shards + int4 experts). Refuse to write when free disk on the cache filesystem < 12 GB. Default cache dir: `~/.iq-local-cache`.
- Storage dtype bf16 (matches checkpoint); compute dtype fp32 inside the Mamba-3 port and attention softmax; int4 group-64 symmetric quantization for cold experts.
- Tests: pytest. Tests that touch the network require `IQ_NET_TESTS=1` in the environment and are skipped otherwise. Default suite must pass offline.
- Bucket access via huggingface_hub 1.14 Python API only (signatures verified): `list_bucket_tree(bucket_id, prefix=None, *, recursive=None, token=None)`, `get_bucket_file_metadata(bucket_id, remote_path, *, token=None) -> BucketFileMetadata`, `download_bucket_files(bucket_id, files: list[tuple[str, str|Path]], *, raise_on_missing_files=False, token=None) -> None`.
- Bucket/run constants: bucket `kofecheckmate/iq-glm53-all256`, run prefix `IQ/runs/glm53-full78-local-20261025a`. Expert files: `model-moe-<NNN:03d>-expert-<EEE:03d>.safetensors`, 25.2 MB each, BF16, keys `layers.<L>.moe.routed_experts.<E>.{gate_proj,up_proj,down_proj}.weight`.
- Commit per task with conventional prefixes (`feat:`, `test:`, `docs:`). Never commit beyond that task's commit step.
- venv: `/home/shaolin/IQ/.venv-local`. All plan commands assume `source /home/shaolin/IQ/.venv-local/bin/activate`.

---

### Task 1: Local venv, local requirements, network-gated test harness

**Files:**
- Create: `requirements-local.txt`
- Create: `tests/iq_local/__init__.py` (empty)
- Create: `tests/iq_local/conftest.py`
- Create: `iq_local/__init__.py` (empty)

**Interfaces:**
- Consumes: nothing.
- Produces: venv at `.venv-local` with torch/transformers/safetensors/huggingface_hub/pytest; pytest net-gate in `tests/iq_local/conftest.py` (test marked `@pytest.mark.net` skipped unless `IQ_NET_TESTS=1`).

- [ ] **Step 1: Create the venv and install dependencies**

```bash
cd /home/shaolin/IQ
python3.11 -m venv .venv-local || python3 -m venv .venv-local
source .venv-local/bin/activate
pip install --upgrade pip
pip install "torch>=2.10" --index-url https://download.pytorch.org/whl/cpu
pip install "transformers==5.17.0" "safetensors>=0.6" "huggingface_hub==1.14.0" "pytest==9.1.1"
python -c "import torch; print(torch.__version__)"
```

Expected: install succeeds; torch prints >= 2.10 with no CUDA requirement.

- [ ] **Step 2: Write `requirements-local.txt`**

```
torch>=2.10
transformers==5.17.0
safetensors>=0.6
huggingface_hub==1.14.0
pytest==9.1.1
```

- [ ] **Step 3: Write the network gate fixture**

`tests/iq_local/conftest.py`:

```python
import os

import pytest


def pytest_collection_modifyitems(config, items):
    net_on = os.environ.get("IQ_NET_TESTS") == "1"
    for item in items:
        if "net" in item.keywords and not net_on:
            item.add_marker(pytest.mark.skip(reason="IQ_NET_TESTS=1 required"))
```

- [ ] **Step 4: Verify the harness runs offline**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local -q`
Expected: exit 0, no tests collected.

- [ ] **Step 5: Commit**

```bash
cd /home/shaolin/IQ
git add requirements-local.txt tests/iq_local/__init__.py tests/iq_local/conftest.py iq_local/__init__.py
git commit -m "feat: local streaming venv requirements and net-gated test harness"
```

---

### Task 2: BucketClient wrapper (typed errors, metadata, batched download)

**Files:**
- Create: `iq_local/bucket.py`
- Test: `tests/iq_local/test_bucket.py`

**Interfaces:**
- Consumes: `huggingface_hub` bucket API signatures (Global Constraints).
- Produces:
  - `BUCKET_ID = "kofecheckmate/iq-glm53-all256"`, `RUN_PREFIX = "IQ/runs/glm53-full78-local-20261025a"` (module constants)
  - `class BucketError(RuntimeError)`, `class BucketAuthError(BucketError)`, `class BucketObjectMissing(BucketError)`
  - `class BucketClient(bucket_id: str, run_prefix: str)` with:
    - `metadata(self, path: str) -> BucketFileMetadata` (`path` relative to run prefix)
    - `download(self, files: list[tuple[str, Path]]) -> None` (batched; paths relative to run prefix)
    - `exists(self, path: str) -> bool`

- [ ] **Step 1: Write the failing test**

`tests/iq_local/test_bucket.py`:

```python
import os

import pytest

from iq_local.bucket import (
    BUCKET_ID,
    RUN_PREFIX,
    BucketAuthError,
    BucketClient,
    BucketObjectMissing,
)


def test_constants_match_checkpoint_run():
    assert BUCKET_ID == "kofecheckmate/iq-glm53-all256"
    assert RUN_PREFIX == "IQ/runs/glm53-full78-local-20261025a"


def test_missing_token_raises_bucket_auth_error(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    with pytest.raises(BucketAuthError):
        client.metadata("COMPILE_COMPLETE.json")


@pytest.mark.net
def test_metadata_and_download_real_file(tmp_path):
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    meta = client.metadata("COMPILE_COMPLETE.json")
    assert meta.size > 0
    dest = tmp_path / "iq_compile_complete_probe.json"
    client.download([("COMPILE_COMPLETE.json", dest)])
    assert dest.read_bytes()[:2] == b"{"


@pytest.mark.net
def test_exists_false_for_absent_path():
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    assert not client.exists("definitely/not/here.safetensors")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_bucket.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'iq_local.bucket'`.

- [ ] **Step 3: Write the implementation**

`iq_local/bucket.py`:

```python
"""Typed wrapper around the huggingface_hub 1.14 bucket API.

Token policy: read HF_TOKEN from the environment, never print it, never
persist it. huggingface_hub also reads HF_TOKEN itself; we use it for
presence checks and explicit passing.
"""

from __future__ import annotations

import os
from pathlib import Path

BUCKET_ID = "kofecheckmate/iq-glm53-all256"
RUN_PREFIX = "IQ/runs/glm53-full78-local-20261025a"


class BucketError(RuntimeError):
    """Base error for bucket access failures."""


class BucketAuthError(BucketError):
    """HF_TOKEN missing or rejected by the bucket."""


class BucketObjectMissing(BucketError):
    """Requested object does not exist in the run."""


class BucketClient:
    def __init__(self, bucket_id: str, run_prefix: str) -> None:
        self.bucket_id = bucket_id
        self.run_prefix = run_prefix.rstrip("/")

    @staticmethod
    def _token() -> str | None:
        token = os.environ.get("HF_TOKEN")
        return token or None

    def _remote(self, path: str) -> str:
        return f"{self.run_prefix}/{path.lstrip('/')}"

    def metadata(self, path: str):
        from huggingface_hub import get_bucket_file_metadata

        token = self._token()
        if token is None:
            raise BucketAuthError("HF_TOKEN is not set in the environment")
        remote = self._remote(path)
        try:
            return get_bucket_file_metadata(self.bucket_id, remote, token=token)
        except BucketAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize hub errors
            message = str(exc).lower()
            if "401" in message or "403" in message or "credential" in message:
                raise BucketAuthError(f"bucket rejected credentials: {exc}") from exc
            if "404" in message or "not found" in message or "nosuchkey" in message:
                raise BucketObjectMissing(remote) from exc
            raise BucketError(f"metadata failed for {remote}: {exc}") from exc

    def exists(self, path: str) -> bool:
        try:
            self.metadata(path)
        except BucketObjectMissing:
            return False
        return True

    def download(self, files: list[tuple[str, Path]]) -> None:
        token = self._token()
        if token is None:
            raise BucketAuthError("HF_TOKEN is not set in the environment")
        if not files:
            return
        for _, dest in files:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import download_bucket_files

        payload = [(self._remote(remote), str(dest)) for remote, dest in files]
        try:
            download_bucket_files(
                self.bucket_id, payload, raise_on_missing_files=True, token=token
            )
        except Exception as exc:  # noqa: BLE001 - normalize hub errors
            message = str(exc).lower()
            if "401" in message or "403" in message or "credential" in message:
                raise BucketAuthError(f"bucket rejected credentials: {exc}") from exc
            if "404" in message or "not found" in message or "missing" in message:
                raise BucketObjectMissing(f"batch download missing objects: {exc}") from exc
            raise BucketError(f"batch download failed: {exc}") from exc
```

- [ ] **Step 4: Run tests — offline pass, then net pass**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_bucket.py -q`
Expected: 2 passed, 2 skipped (net tests skipped offline).

Then with `HF_TOKEN` exported in the shell (never echo it):
Run: `cd /home/shaolin/IQ && IQ_NET_TESTS=1 .venv-local/bin/python -m pytest tests/iq_local/test_bucket.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/shaolin/IQ
git add iq_local/bucket.py tests/iq_local/test_bucket.py
git commit -m "feat: typed bucket client over hf hub python bucket api"
```

---

### Task 3: RunIndex + verify (manifests, weight_map parse, expert stage map)

**Files:**
- Create: `iq_local/index.py`
- Test: `tests/iq_local/test_index.py`

**Interfaces:**
- Consumes: `iq_local/bucket.py` (`BucketClient`), repo's `IQHybridConfig.from_json` (`iq_model/hybrid.py:273`).
- Produces:
  - `_EXPERT_FILE` regex: `^model-moe-(\d{3})-expert-(\d{3})\.safetensors$` (module-level, reused by Task 6)
  - `class RunIndex(client: BucketClient | None, work_dir: Path)` with:
    - Downloads run `iq_config.json`, `model.safetensors.index.json`, `COMPILE_COMPLETE.json`, `transplant_manifest.json` into `work_dir` on first use (skips when already present).
    - `config: IQHybridConfig` (via `IQHybridConfig.from_json(work_dir / "iq_config.json")`)
    - `weight_map: dict[str, str]` (63,164 entries: tensor key -> shard filename)
    - `files: list[str]` (20,283 distinct shard filenames, sorted)
    - `experts_by_stage: dict[int, list[str]]` (78 stages x 256 expert filenames)
    - `stage_of_layer(layer_idx: int) -> int` (memoized in `self._stage_cache`)
    - `keys_for_file(filename: str) -> list[str]`
  - `RunIndex.verify_against_local_checkpoint(run_index: RunIndex, local_dir: Path) -> None` — asserts run `iq_config.json` bytes and `weight_map` dict equal the local `checkpoints/iq` copies; checks the manifest recipient fingerprint prefix `6c292b47`.

- [ ] **Step 1: Write the failing test**

`tests/iq_local/test_index.py`:

```python
import json
from pathlib import Path

import pytest

from iq_local.bucket import BUCKET_ID, RUN_PREFIX, BucketClient
from iq_local.index import RunIndex

REPO = Path(__file__).resolve().parents[2]
LOCAL_IQ = REPO / "checkpoints" / "iq"


def test_expert_stage_index_is_complete_from_local_fixture():
    run_index = RunIndex.__new__(RunIndex)
    run_index.weight_map = json.loads(
        (LOCAL_IQ / "model.safetensors.index.json").read_text()
    )["weight_map"]
    run_index.files = sorted(set(run_index.weight_map.values()))
    run_index._build_experts_by_stage()
    assert len(run_index.experts_by_stage) == 78
    for stage, experts in run_index.experts_by_stage.items():
        assert len(experts) == 256
        assert all(
            name.startswith(f"model-moe-{stage:03d}-expert-") for name in experts
        )


@pytest.mark.net
def test_run_index_downloads_and_matches_local(tmp_path):
    client = BucketClient(BUCKET_ID, RUN_PREFIX)
    run_index = RunIndex(client, tmp_path)
    assert len(run_index.weight_map) == 63164
    assert len(run_index.files) == 20283
    RunIndex.verify_against_local_checkpoint(run_index, LOCAL_IQ)
    assert run_index.stage_of_layer(2) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_index.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'iq_local.index'`.

- [ ] **Step 3: Write the implementation**

`iq_local/index.py`:

```python
"""Run-side index: fetch run manifests, parse weight_map, build expert map."""

from __future__ import annotations

import json
import re
from pathlib import Path

from iq_local.bucket import BucketClient

_MANIFESTS = (
    "iq_config.json",
    "model.safetensors.index.json",
    "COMPILE_COMPLETE.json",
    "transplant_manifest.json",
)

_EXPERT_FILE = re.compile(r"^model-moe-(\d{3})-expert-(\d{3})\.safetensors$")


class RunIndex:
    def __init__(self, client: BucketClient | None, work_dir: Path) -> None:
        self.client = client
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for name in _MANIFESTS:
            self._ensure(name)
        self.config = self._load_config()
        index_payload = json.loads(
            (self.work_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        self.weight_map: dict[str, str] = dict(index_payload["weight_map"])
        self.files: list[str] = sorted(set(self.weight_map.values()))
        self._build_experts_by_stage()
        self._stage_cache: dict[int, int] = {}

    def _ensure(self, name: str) -> None:
        dest = self.work_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            return
        if self.client is None:
            raise FileNotFoundError(f"{dest} missing and no bucket client configured")
        self.client.download([(name, dest)])

    def _load_config(self):
        from iq_model.hybrid import IQHybridConfig

        return IQHybridConfig.from_json(str(self.work_dir / "iq_config.json"))

    def _build_experts_by_stage(self) -> None:
        experts_by_stage: dict[int, list[str]] = {}
        for filename in self.files:
            match = _EXPERT_FILE.match(filename)
            if match:
                experts_by_stage.setdefault(int(match.group(1)), []).append(filename)
        self.experts_by_stage = experts_by_stage

    def stage_of_layer(self, layer_idx: int) -> int:
        """Resolve MoE stage for a schedule layer via one routed-expert key.

        Keys: `layers.<layer_idx>.moe.routed_experts.<E>...`; files:
        `model-moe-<stage>-expert-<E>.safetensors`. Memoized per layer.
        """
        if layer_idx in self._stage_cache:
            return self._stage_cache[layer_idx]
        prefix = f"layers.{layer_idx}.moe.routed_experts."
        for key, filename in self.weight_map.items():
            if key.startswith(prefix):
                match = _EXPERT_FILE.match(filename)
                if match:
                    stage = int(match.group(1))
                    self._stage_cache[layer_idx] = stage
                    return stage
                break
        raise KeyError(f"layer {layer_idx} has no routed expert keys")

    def keys_for_file(self, filename: str) -> list[str]:
        return sorted(key for key, file in self.weight_map.items() if file == filename)

    @staticmethod
    def verify_against_local_checkpoint(run_index: RunIndex, local_dir: Path) -> None:
        local_dir = Path(local_dir)
        local_config = (local_dir / "iq_config.json").read_bytes()
        run_config = (run_index.work_dir / "iq_config.json").read_bytes()
        if local_config != run_config:
            raise ValueError("run iq_config.json differs from local checkpoints/iq copy")
        local_index = json.loads(
            (local_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        if dict(local_index["weight_map"]) != run_index.weight_map:
            raise ValueError("run weight_map differs from local checkpoints/iq index")
        manifest = json.loads(
            (run_index.work_dir / "transplant_manifest.json").read_text(encoding="utf-8")
        )
        recipient = manifest.get("recipient", {})
        fingerprint = recipient.get("config_fingerprint", "")
        if fingerprint and not str(fingerprint).startswith("6c292b47"):
            raise ValueError(f"unexpected recipient config fingerprint: {fingerprint}")
```

Implementer note: if the manifest's recipient fingerprint field lives at a different JSON path, read `/tmp/opencode/iqmeta/transplant_manifest.json` to find the real field and adjust; document the deviation in the commit message.

- [ ] **Step 4: Run tests**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_index.py -q`
Expected: 1 passed, 1 skipped offline.

Run: `cd /home/shaolin/IQ && IQ_NET_TESTS=1 .venv-local/bin/python -m pytest tests/iq_local/test_index.py -q`
Expected: 2 passed (downloads ~4 small manifests; exercises `IQHybridConfig.from_json`).

- [ ] **Step 5: Commit**

```bash
cd /home/shaolin/IQ
git add iq_local/index.py tests/iq_local/test_index.py
git commit -m "feat: run index with weight_map parse and expert stage map"
```

---

### Task 4: LocalCache (content-addressed LRU store with disk guard)

**Files:**
- Create: `iq_local/cache.py`
- Test: `tests/iq_local/test_cache.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class CacheDiskFullError(RuntimeError)`
  - `class LocalCache(root: Path, budget_bytes: int, *, min_free_bytes: int = 12 * (1 << 30))` with:
    - `path_for(self, key: str) -> Path` (slashes replaced with `_`)
    - `contains(self, key: str) -> bool`
    - `commit(self, key: str, tmp_path: Path) -> Path` — moves a fully-written temp file into the store, records size, evicts oldest-unpinned beyond budget, raises `CacheDiskFullError` when free disk on `root`'s filesystem < `min_free_bytes`.
    - `pin(self, key: str) -> None` / `unpin(self, key: str) -> None`
    - `usage(self) -> tuple[int, int]` (bytes used, entry count)

- [ ] **Step 1: Write the failing test**

`tests/iq_local/test_cache.py`:

```python
from pathlib import Path

import pytest

from iq_local.cache import CacheDiskFullError, LocalCache


def test_commit_moves_file_and_counts(tmp_path: Path):
    cache = LocalCache(tmp_path / "store", budget_bytes=1 << 20)
    src = tmp_path / "staged.bin"
    src.write_bytes(b"x" * 100)
    path = cache.commit("run/model-global.safetensors", src)
    assert path.exists()
    assert not src.exists()
    assert cache.contains("run/model-global.safetensors")
    assert cache.usage() == (100, 1)


def test_lru_eviction_respects_budget(tmp_path: Path):
    cache = LocalCache(tmp_path / "store", budget_bytes=250)
    for i in range(3):
        src = tmp_path / f"s{i}.bin"
        src.write_bytes(b"y" * 100)
        cache.commit(f"file{i}", src)
    assert not cache.contains("file0")
    assert cache.contains("file1")
    assert cache.contains("file2")


def test_pinned_entry_survives_eviction(tmp_path: Path):
    cache = LocalCache(tmp_path / "store", budget_bytes=250)
    for i in range(3):
        src = tmp_path / f"s{i}.bin"
        src.write_bytes(b"y" * 100)
        cache.commit(f"file{i}", src)
        if i == 0:
            cache.pin("file0")
    assert cache.contains("file0")


def test_disk_guard_raises(tmp_path: Path, monkeypatch):
    import iq_local.cache as cache_module

    cache = LocalCache(tmp_path / "store", budget_bytes=1 << 30, min_free_bytes=1)
    monkeypatch.setattr(
        cache_module.shutil, "disk_usage", lambda _p: type("U", (), {"free": 0})()
    )
    src = tmp_path / "s.bin"
    src.write_bytes(b"z" * 10)
    with pytest.raises(CacheDiskFullError):
        cache.commit("guarded", src)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_cache.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'iq_local.cache'`.

- [ ] **Step 3: Write the implementation**

`iq_local/cache.py`:

```python
"""Content-addressed LRU byte store on NVMe with a budget and a disk guard."""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from pathlib import Path


class CacheDiskFullError(RuntimeError):
    """Writing would leave the cache filesystem under the free-space floor."""


class LocalCache:
    def __init__(
        self,
        root: Path,
        budget_bytes: int,
        *,
        min_free_bytes: int = 12 * (1 << 30),
    ) -> None:
        self.root = Path(root)
        self.budget_bytes = int(budget_bytes)
        self.min_free_bytes = int(min_free_bytes)
        self.root.mkdir(parents=True, exist_ok=True)
        self._entries: OrderedDict[str, int] = OrderedDict()  # key -> size, LRU order
        self._pinned: set[str] = set()
        self._bytes = 0

    def path_for(self, key: str) -> Path:
        return self.root / key.replace("/", "_")

    def contains(self, key: str) -> bool:
        return key in self._entries

    def commit(self, key: str, tmp_path: Path) -> Path:
        size = os.path.getsize(tmp_path)
        free = shutil.disk_usage(self.root).free
        if free < self.min_free_bytes:
            raise CacheDiskFullError(
                f"free disk {free} bytes < guard {self.min_free_bytes}; refusing write"
            )
        self._evict_until(self.budget_bytes - size)
        dest = self.path_for(key)
        if dest.exists():
            dest.unlink()
        os.replace(tmp_path, dest)
        self._entries[key] = size
        self._entries.move_to_end(key)
        self._bytes += size
        return dest

    def pin(self, key: str) -> None:
        self._pinned.add(key)

    def unpin(self, key: str) -> None:
        self._pinned.discard(key)

    def usage(self) -> tuple[int, int]:
        return self._bytes, len(self._entries)

    def _evict_until(self, target: int) -> None:
        while self._bytes > target:
            victim = None
            for key in self._entries:
                if key not in self._pinned:
                    victim = key
                    break
            if victim is None:
                return  # all pinned: overshoot allowed rather than deadlock
            size = self._entries.pop(victim)
            self._bytes -= size
            path = self.path_for(victim)
            if path.exists():
                path.unlink()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_cache.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/shaolin/IQ
git add iq_local/cache.py tests/iq_local/test_cache.py
git commit -m "feat: budgeted lru nvme cache with disk guard"
```

---

### Task 5: ExpertQuantizer (int4 group-64 symmetric)

**Files:**
- Create: `iq_local/quant.py`
- Test: `tests/iq_local/test_quant.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `GROUP = 64`
  - `class PackedExpert(NamedTuple)`: `packed: torch.Tensor` (`uint8`, `[rows, cols//2]`), `scales: torch.Tensor` (`float32`, `[rows, cols//GROUP]`)
  - `quantize_int4(tensor: torch.Tensor) -> PackedExpert` (2-D input, last dim divisible by 64; grouped along last dim; raises `ValueError` otherwise)
  - `dequantize_int4(packed: PackedExpert, out_shape: tuple[int, int], out_dtype: torch.dtype) -> torch.Tensor`
  - Both deterministic, CPU-only.

- [ ] **Step 1: Write the failing test**

`tests/iq_local/test_quant.py`:

```python
import pytest
import torch

from iq_local.quant import GROUP, dequantize_int4, quantize_int4


def test_roundtrip_error_is_small():
    torch.manual_seed(0)
    tensor = torch.randn(256, 128, dtype=torch.float32) * 0.05
    packed = quantize_int4(tensor)
    assert packed.packed.dtype == torch.uint8
    assert packed.packed.shape == (256, 64)
    assert packed.scales.shape == (256, 2)
    out = dequantize_int4(packed, (256, 128), torch.bfloat16).float()
    error = (out - tensor).abs().max().item()
    assert error <= tensor.abs().max().item() / 8.0


def test_group_constant():
    assert GROUP == 64


def test_rejects_non_divisible_last_dim():
    tensor = torch.randn(8, 65, dtype=torch.float32)
    with pytest.raises(ValueError):
        quantize_int4(tensor)


def test_rejects_non_2d():
    tensor = torch.randn(8, 8, 8, dtype=torch.float32)
    with pytest.raises(ValueError):
        quantize_int4(tensor)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_quant.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'iq_local.quant'`.

- [ ] **Step 3: Write the implementation**

`iq_local/quant.py`:

```python
"""Group-64 symmetric int4 quantization for cold expert shards."""

from __future__ import annotations

from typing import NamedTuple

import torch

GROUP = 64


class PackedExpert(NamedTuple):
    packed: torch.Tensor
    scales: torch.Tensor


def quantize_int4(tensor: torch.Tensor) -> PackedExpert:
    if tensor.ndim != 2:
        raise ValueError("quantize_int4 expects a 2-D tensor")
    rows, cols = tensor.shape
    if cols % GROUP != 0:
        raise ValueError(f"last dim {cols} not divisible by group {GROUP}")
    x = tensor.detach().to(torch.float32)
    grouped = x.reshape(rows, cols // GROUP, GROUP)
    max_abs = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scales = (max_abs / 7.0).squeeze(-1)  # [rows, cols // GROUP]
    q = torch.round(grouped / scales.unsqueeze(-1)).clamp_(-8, 7).to(torch.int8)
    low = (q & 0x0F).to(torch.uint8)
    high = ((q >> 4) & 0x0F).to(torch.uint8)
    packed = low | (high << 4)
    return PackedExpert(packed.reshape(rows, cols // 2), scales)


def dequantize_int4(
    packed: PackedExpert, out_shape: tuple[int, int], out_dtype: torch.dtype
) -> torch.Tensor:
    rows, cols = out_shape
    if cols % GROUP != 0:
        raise ValueError(f"last dim {cols} not divisible by group {GROUP}")
    packed3d = packed.packed.reshape(rows, cols // GROUP, GROUP // 2)
    low = (packed3d & 0x0F).to(torch.int8)
    high = ((packed3d >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)
    q = torch.stack((low, high), dim=-1).reshape(rows, cols // GROUP, GROUP)
    values = q.to(torch.float32) * packed.scales.to(torch.float32).unsqueeze(-1)
    return values.reshape(rows, cols).to(out_dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_quant.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/shaolin/IQ
git add iq_local/quant.py tests/iq_local/test_quant.py
git commit -m "feat: int4 group-64 expert quantizer with roundtrip tests"
```

---

### Task 6: StreamingWeightProvider (RAM LRU -> NVMe cache -> bucket)

**Files:**
- Create: `iq_local/weights.py`
- Test: `tests/iq_local/test_weights.py`

**Interfaces:**
- Consumes: `RunIndex` (Task 3), `LocalCache` (Task 4), `BucketClient` (Task 2), `iq_local/quant.py` (Task 5).
- Produces:
  - `class StreamingWeightProvider(index, cache, client, *, max_ram_experts: int = 64)` with:
    - `load_global(key: str) -> torch.Tensor` (bf16)
    - `load_layer_base(layer_idx: int) -> dict[str, torch.Tensor]` — every non-`moe.routed_experts.` key for the layer (includes router/latent MoE tensors)
    - `load_expert(stage: int, expert_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]` — dequantized bf16 `(gate, up, down)`
    - `counters: dict[str, int]` — `bucket_files`, `bucket_bytes`, `nvme_hits`, `ram_hits`, `int4_converted`
  - Tier order per expert: RAM LRU (dequantized bf16 triple) -> NVMe int4 copy -> bucket bf16 fetch (quantize once, persist int4, do not keep the bf16 expert file in cache).
  - Whole non-expert shards are cached bf16; tensors are read per-key with `safetensors.safe_open`.

- [ ] **Step 1: Write the failing test**

`tests/iq_local/test_weights.py` builds a fake run: a local fixture dir with two tiny safetensors shards (`model-global.safetensors` holding `lm_head.weight`, `model-moe-000-expert-000.safetensors` holding three `[64, 64]` fp32->bf16 expert weights), a hand-built `RunIndex` (same pattern as `test_index.py`), a `FakeClient` that copies fixture files to requested destinations, and a `LocalCache` in tmp_path. Tests:

1. `load_global` returns the tensor value; second call increments `nvme_hits` and does not touch the client (counter `bucket_files` unchanged).
2. `load_expert` first call: downloads bf16 once, persists an int4 copy under cache key `int4/model-moe-000-expert-000.safetensors`, and the bf16 expert file is NOT left in the cache (its cache key absent). Dequantized values approximate the bf16 originals within `1/8` of the max abs weight.
3. Second `load_expert` call after eviction from RAM: loads from the persisted int4 copy (counter `bucket_files` unchanged, `ram_hits` or reload path taken).
4. With `max_ram_experts=1`, loading expert A then expert B evicts A (next A load is a cache/int4 load, not a bucket fetch).
5. `load_layer_base(0)` returns exactly the layer-0 keys that are not `moe.routed_experts.*`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_weights.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'iq_local.weights'`.

- [ ] **Step 3: Write the implementation**

`iq_local/weights.py`:

```python
"""Three-tier streaming weight provider: RAM LRU, NVMe cache, HF bucket."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open

from iq_local.cache import LocalCache
from iq_local.index import RunIndex
from iq_local.quant import dequantize_int4, quantize_int4


class StreamingWeightProvider:
    def __init__(
        self,
        index: RunIndex,
        cache: LocalCache,
        client,
        *,
        max_ram_experts: int = 64,
    ) -> None:
        self.index = index
        self.cache = cache
        self.client = client
        self.max_ram_experts = max_ram_experts
        self._ram: OrderedDict[str, tuple[torch.Tensor, ...]] = OrderedDict()
        self.counters = {
            "bucket_files": 0,
            "bucket_bytes": 0,
            "nvme_hits": 0,
            "ram_hits": 0,
            "int4_converted": 0,
        }

    # -- shard tier ---------------------------------------------------------

    def _shard_path(self, filename: str) -> Path:
        key = f"shards/{filename}"
        if self.cache.contains(key):
            self.counters["nvme_hits"] += 1
            return self.cache.path_for(key)
        tmp = self.cache.path_for("staging.tmp")
        self.client.download([(filename, tmp)])
        self.counters["bucket_files"] += 1
        self.counters["bucket_bytes"] += tmp.stat().st_size
        return self.cache.commit(key, tmp)

    def _read_keys(self, path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in keys:
                out[key] = handle.get_tensor(key)
        return out

    def load_global(self, key: str) -> torch.Tensor:
        filename = self.index.weight_map[key]
        path = self._shard_path(filename)
        return self._read_keys(path, [key])[key]

    def load_layer_base(self, layer_idx: int) -> dict[str, torch.Tensor]:
        prefix = f"layers.{layer_idx}."
        excluded = f"{prefix}moe.routed_experts."
        keys = [
            key
            for key in self.index.weight_map
            if key.startswith(prefix) and not key.startswith(excluded)
        ]
        by_file: dict[str, list[str]] = {}
        for key in keys:
            by_file.setdefault(self.index.weight_map[key], []).append(key)
        tensors: dict[str, torch.Tensor] = {}
        for filename, file_keys in by_file.items():
            path = self._shard_path(filename)
            tensors.update(self._read_keys(path, file_keys))
        return tensors

    # -- expert tier --------------------------------------------------------

    def _expert_key(self, stage: int, expert_idx: int) -> str:
        return f"{stage}:{expert_idx:03d}"

    def _expert_filename(self, stage: int, expert_idx: int) -> str:
        return f"model-moe-{stage:03d}-expert-{expert_idx:03d}.safetensors"

    def _ram_get(self, key: str):
        if key not in self._ram:
            return None
        self._ram.move_to_end(key)
        self.counters["ram_hits"] += 1
        return self._ram[key]

    def _ram_put(self, key: str, triple: tuple[torch.Tensor, ...]) -> None:
        self._ram[key] = triple
        self._ram.move_to_end(key)
        while len(self._ram) > self.max_ram_experts:
            self._ram.popitem(last=False)

    def load_expert(self, stage: int, expert_idx: int):
        key = self._expert_key(stage, expert_idx)
        cached = self._ram_get(key)
        if cached is not None:
            return cached
        filename = self._expert_filename(stage, expert_idx)
        int4_key = f"int4/{filename}"
        if self.cache.contains(int4_key):
            payload = torch.load(
                self.cache.path_for(int4_key), map_location="cpu", weights_only=True
            )
            triple = self._dequant_payload(payload)
        else:
            tmp = self.cache.path_for("expert.tmp")
            self.client.download([(filename, tmp)])
            self.counters["bucket_files"] += 1
            self.counters["bucket_bytes"] += tmp.stat().st_size
            layer_keys = self.index.keys_for_file(filename)
            with safe_open(str(tmp), framework="pt", device="cpu") as handle:
                raw = {k: handle.get_tensor(k).to(torch.float32) for k in layer_keys}
            tmp.unlink()
            packed = {
                name: quantize_int4(tensor)
                for name, tensor in (
                    ("gate", raw[f"layers.{self._expert_layer(stage)}.moe.routed_experts.{expert_idx}.gate_proj.weight"]),
                    ("up", raw[f"layers.{self._expert_layer(stage)}.moe.routed_experts.{expert_idx}.up_proj.weight"]),
                    ("down", raw[f"layers.{self._expert_layer(stage)}.moe.routed_experts.{expert_idx}.down_proj.weight"]),
                )
            }
            payload = {
                name: {
                    "packed": expert.packed,
                    "scales": expert.scales,
                    "shape": tuple(expert.packed.shape[0], expert.packed.shape[1] * 2),
                }
                for name, expert in packed.items()
            }
            staged = self.cache.path_for("int4.tmp")
            torch.save(payload, staged)
            self.cache.commit(int4_key, staged)
            self.counters["int4_converted"] += 1
            triple = self._dequant_payload(payload)
        self._ram_put(key, triple)
        return triple

    def _expert_layer(self, stage: int) -> int:
        if not hasattr(self, "_stage_layers"):
            self._stage_layers = {}
        if stage not in self._stage_layers:
            sample = self.index.experts_by_stage[stage][0]
            first_key = self.index.keys_for_file(sample)[0]
            self._stage_layers[stage] = int(first_key.split(".")[1])
        return self._stage_layers[stage]

    @staticmethod
    def _dequant_payload(payload) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = []
        for name in ("gate", "up", "down"):
            entry = payload[name]
            out.append(
                dequantize_int4(
                    type(
                        "P",
                        (),
                        {"packed": entry["packed"], "scales": entry["scales"]},
                    )(),
                    tuple(entry["shape"]),
                    torch.bfloat16,
                )
            )
        return tuple(out)
```

- [ ] **Step 4: Run tests**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_weights.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/shaolin/IQ
git add iq_local/weights.py tests/iq_local/test_weights.py
git commit -m "feat: three-tier streaming weight provider with int4 expert cache"
```

---

### Task 7: CPU Mamba-3 MIMO runtime port (`mamba3_cpu.py`)

**Files:**
- Create: `iq_model/state/mamba3_cpu.py`
- Test: `tests/iq_local/test_mamba3_cpu.py`
- Create: `iq_model/state/__init__.py` only if missing.

**Interfaces:**
- Consumes: upstream reference math at commit `e9594ce1c732d97440f0332fdc43170a2294dbfa` of `state-spaces/mamba`:
  - `https://raw.githubusercontent.com/state-spaces/mamba/e9594ce1c732d97440f0332fdc43170a2294dbfa/ops/cute/mamba3/mamba3_step_fn.py` — `selective_state_update_fused_ref_v2` (decode-step recurrence; mirror verbatim)
  - `https://raw.githubusercontent.com/state-spaces/mamba/e9594ce1c732d97440f0332fdc43170a2294dbfa/ops/triton/mamba3/mamba3_mimo_rotary_step.py` — `apply_rotary_qk_inference_reference` (halved-branch rotary; mirror verbatim)
  - `https://raw.githubusercontent.com/state-spaces/mamba/e9594ce1c732d97440f0332fdc43170a2294dbfa/ops/triton/layernorm_gated.py` — `rms_norm_ref` semantics (norm-then-gate; mirror verbatim)
  If a URL 404s, list the repo tree at that commit (GitHub API `git/trees/e9594ce...?recursive=1`) and locate the same file; record the real path in the docstring.
- Produces:
  - `MAMBA3_CPU_SOURCE_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"` (module constant)
  - `class Mamba3MIMOConfigCPU` — field names/defaults MUST be copied from production `iq_model/state/mamba3.py` `Mamba3MIMOConfig` (read the file first; do not invent names).
  - `class Mamba3CPU(nn.Module)` with `self.core = _Mamba3MIMOCore(nn.Module)` so state_dict keys mirror production (`layers.<i>.mamba.core.*`).
  - `class CPUMamba3State` (dataclass): `angle` fp32 `[B, nheads, num_rope_angles]`, `ssm` fp32 `[B, nheads, headdim, d_state]`, `k` bf16 `[B, mimo_rank, nheads, d_state]`, `v` bf16 `[B, nheads, headdim]`; `CPUMamba3State.create(batch, config, device="cpu")` zeros.
  - `Mamba3CPU.step(x_b: [B, d_model], state: CPUMamba3State) -> (y_b: [B, d_model], CPUMamba3State)`
  - `Mamba3CPU.prefill(x: [B, T, d_model], state=None) -> (y: [B, T, d_model], CPUMamba3State)` — v1 implementation is a loop of `step` calls (same recurrence; fp accumulation order differs from the CUDA chunk kernel; H200 parity fixture in Task 9 quantifies drift).
  - Core parameter name set derived from the checkpoint (Step 1 below), not guessed.

**Wiring table (from upstream refs; each row must be cross-checked against the reference code before porting):**

| Stage | Formula |
| --- | --- |
| in_proj split | `[d_inner, d_inner, d_state*mimo_rank, d_state*mimo_rank, nheads, nheads, nheads, num_rope_angles]` -> `(x, xs, B_bias, C_bias, z, dt_proj, angle_proj, angle_state)`; d_model=4096, d_inner=8192, headdim=64, nheads=128, mimo_rank=4, num_rope_angles=32, d_in_proj=17824 |
| decay params | `DT = softplus(dt_proj + dt_bias)`; `trap = <mirror reference exactly>`; `A = -heavy_tail(dd_A)`, clamp `max=-A_floor` (A_floor=1e-4); `alpha = exp(A * DT)` |
| update coeffs | `beta = (1 - trap) * DT * alpha`; `gamma = trap * DT` |
| B/C norms | reshape `(B, R, S)` -> plain RMSNorm eps 1e-5 -> expand to nheads |
| B matrix | `B = B_norm.unsqueeze(2)` broadcast over headdim -> `(B, R, nheads, S)` |
| rotary | halved branch (`rotate_pairwise=False`); `bias_q = C_bias.permute(1, 0, 2)`, `bias_k = B_bias.permute(1, 0, 2)`; `angle = tanh(angle_proj) * DT * pi + angle_state`; cache `angle` fp32 |
| step recurrence | `x_vals = x * xproj` (both `[B, nheads, headdim]` via reshape of the two d_inner halves); `xBt_state = einsum('brnh,brns->bnhs', x_vals * gamma, B)`; `xBt_prev = einsum('brnh,brns->bnhs', xs_vals * beta, B_state)`; `new_state = state * alpha + xBt_state + xBt_prev`; `out_r = einsum('bnhs,brns->brnh', new_state, C) + x_vals * D`; step core called with `outproj=None, z=None` |
| postprocess | `z_r = einsum('bhp,rhp->brhp', z, mimo_z)`; `y = rms_norm_gated(y_r, z_r, norm.weight, groups=headdim, eps=1e-5)` (norm-then-gate: multiply by `silu(z_r)`); `out = einsum('brhp,rhp->bhp', y, mimo_o)`; `out_proj(out)` |

- [ ] **Step 1: Extract the exact core parameter name set**

```bash
cd /home/shaolin/IQ && .venv-local/bin/python - <<'PY'
import json, collections
wm = json.load(open("checkpoints/iq/model.safetensors.index.json"))["weight_map"]
names = collections.Counter()
for key in wm:
    if key.startswith("layers.0.mamba.core."):
        names[key.removeprefix("layers.0.mamba.core.")] += 1
for name, count in sorted(names.items()):
    print(count, name)
PY
```

Record the printed list. The `_Mamba3MIMOCore` must produce exactly these parameter names (plus any buffer-only entries as buffers). This list is also asserted in the test, so it becomes a fixture constant in the test file.

- [ ] **Step 2: Write the failing tests**

`tests/iq_local/test_mamba3_cpu.py`:

1. `test_core_param_names_match_checkpoint` — builds `Mamba3CPU` with config from `checkpoints/iq/iq_config.json` (via `IQHybridConfig.from_json` -> mamba section), asserts `set(state_dict keys of core)` equals the name set from Step 1 (parameter names, ignoring dtypes/shapes).
2. `test_zero_weights_zero_output` — zero-init linears, ones RMSNorm weights: `step` returns zeros and state stays zero.
3. `test_cache_shapes` — `CPUMamba3State.create(batch=3)` shapes/dtypes per Interfaces.
4. `test_prefill_matches_incremental` — random weights (seed 0): `prefill(x[B,T])` equals T sequential `step` calls on the same initial state, `torch.allclose` rtol 0 atol 1e-5 (identical code path, guards future refactors).

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_mamba3_cpu.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'iq_model.state.mamba3_cpu'`.

- [ ] **Step 4: Write the implementation**

Port procedure (mandatory):
1. `curl` the three upstream files (or locate via tree API) into `/tmp/opencode/mamba3ref/`.
2. Copy the reference functions into `mamba3_cpu.py` as private helpers, renamed with `_cpu` suffix, each with a docstring citing the source file + commit. Keep tensor ops identical (einsums, exp/softplus/sigmoid, RMSNorm eps, silu gating).
3. Assemble `_Mamba3MIMOCore` with parameters named from the Step 1 list; initialize: RMSNorm weights `ones`, rope/dt/A params per production semantics where derivable, linear weights `zeros` (weights are loaded from the checkpoint later; init only needs determinism — note this in the docstring).
4. Implement `step` as: preprocessing (in_proj split, DT/trap/B/C norms/rotary) -> reference step core -> postprocess (gated RMSNorm, out einsum, out_proj) -> new state return.
5. Implement `prefill` as the looped-step wrapper. Do NOT import `mamba_ssm` or `tilelang` anywhere.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_mamba3_cpu.py -q`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
cd /home/shaolin/IQ
git add iq_model/state/mamba3_cpu.py tests/iq_local/test_mamba3_cpu.py
git commit -m "feat: cpu mamba-3 mimo runtime ported from upstream reference math"
```

---

### Task 8: CPUHybridForCausalLM (schedule composition, functional MoE, key parity)

**Files:**
- Create: `iq_local/cpu_hybrid.py`
- Test: `tests/iq_local/test_cpu_hybrid.py`

**Interfaces:**
- Consumes: `IQHybridConfig` (hybrid.py:48), `Mamba3CPU` (Task 7), production device-agnostic classes `DenseContextResidualLayer` (hybrid.py:631), `CompressedContextResidualLayer` (hybrid.py:662), `StableLatentMoE` (latent_moe.py:91), `SiTUExpert`/`SiTUAndMul` (latent_moe.py:63-90), `Mamba3ResidualLayer` wiring pattern (hybrid.py:587-630).
- Produces:
  - `class FunctionalStableLatentMoELayer(nn.Module)` — mirrors production StableLatentMoE structure (router, latent_down, latent_up, latent_norm, shared_experts as real loadable params) but computes routed experts FUNCTIONALLY: `forward(x, expert_fn)` where `expert_fn(stage, expert_idx) -> (gate, up, down)` bf16 tensors; routed-expert math mirrors `SiTUExpert.forward` + `SiTUAndMul` exactly (read the file; mirror verbatim). No routed-expert `nn.Module`s, no `copy_`.
  - `class CPUHybridForCausalLM(nn.Module)` — composes the 235-layer schedule from `config` exactly as production `IQHybridForCausalLM.__init__` (hybrid.py:702-780) minus the CUDA gate: mamba layers wrap `Mamba3CPU` inside the production residual-layer pattern (`norm -> pack -> mamba -> unpack -> residual`, incl. `mix_logit` etc.); CSA/HCA layers instantiate production classes directly; MoE layers use `FunctionalStableLatentMoELayer`; `embed_tokens`, `norm`, `lm_head` mirror production naming.
  - `CPUHybridForCausalLM.forward(input_ids, attention_mask=None) -> logits` (plain, non-streaming; used by parity tests; the streaming runner in Task 10 drives layers directly).

- [ ] **Step 1: Read production structure (do this first, cite lines in commit message)**

Open and note: `IQHybridForCausalLM.__init__` module/attr names, `forward` composition order, `Mamba3ResidualLayer` exact wiring, `StableLatentMoE.forward` and `SiTUAndMul` formula, and the routed-experts attribute name in `StableLatentMoE.__init__`.

- [ ] **Step 2: Write the failing key-parity test**

`tests/iq_local/test_cpu_hybrid.py`:

```python
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_state_dict_keys_match_checkpoint_index():
    from iq_local.cpu_hybrid import CPUHybridForCausalLM
    from iq_model.hybrid import IQHybridConfig

    config = IQHybridConfig.from_json(str(REPO / "checkpoints" / "iq" / "iq_config.json"))
    model = CPUHybridForCausalLM(config)
    produced = set(model.state_dict().keys())
    expected = set(
        json.loads(
            (REPO / "checkpoints" / "iq" / "model.safetensors.index.json").read_text()
        )["weight_map"]
    )
    missing = sorted(expected - produced)
    extra = sorted(produced - expected)
    assert not missing, f"missing {len(missing)}: {missing[:20]}"
    assert not extra, f"extra {len(extra)}: {extra[:20]}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_cpu_hybrid.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'iq_local.cpu_hybrid'`.

- [ ] **Step 4: Implement**

1. `FunctionalStableLatentMoELayer`: copy production `StableLatentMoE.__init__`/`_route`/`forward` into a subclass, replace routed-expert ModuleList with `expert_fn` calls (one call per selected expert per token row; gather weights as local tensors, run the SiTU math with `torch.nn.functional.linear`), keep every other param/module and key name identical. Add a `_stage` attribute so `expert_fn` knows which stage to fetch.
2. Attention + residual layers: instantiate production classes; for mamba layers build a thin CPU wrapper mirroring `Mamba3ResidualLayer` field-for-field but holding `Mamba3CPU` instead of the CUDA module.
3. `CPUHybridForCausalLM`: mirror production init order and attribute names exactly; remove only the CUDA check; keep config-driven schedule logic (mamba/attn/moe placement, attnres groups).
4. Do not import anything CUDA-only.

- [ ] **Step 5: Add the functional-MoE parity test (same file)**

Build a tiny `StableLatentMoE` on CPU with random weights (seed 0); run production `forward` and `FunctionalStableLatentMoELayer.forward` with `expert_fn` reading the production module's routed-expert tensors directly; assert `allclose` atol 1e-4 (bf16 tolerance).

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_cpu_hybrid.py -q`
Expected: 2 passed. Iterate on the key-parity test until exact equality; it is the structural gate for every later task.

- [ ] **Step 7: Commit**

```bash
cd /home/shaolin/IQ
git add iq_local/cpu_hybrid.py tests/iq_local/test_cpu_hybrid.py
git commit -m "feat: cpu hybrid model with functional moe and checkpoint key parity"
```

---

### Task 9: H200 parity fixture job (billed verification gate)

**Files:**
- Create: `scripts/iq_mamba3_parity_job.py`
- Test: `tests/iq_local/test_parity_preflight.py`

**Interfaces:**
- Consumes: `Mamba3CPU` (Task 7), run `iq_config.json` (Task 3), conventions from `scripts/iq_hf_inference.py` (env `HF_TOKEN`, argparse, JSON summary).
- Produces: fixture JSONs under `iq-transfer/parity/`: `prefill_l{L}.json` and `decode_l{L}.json` for layers {0, first mamba layer after an attn block}, each with input/output/state tensors and `max_abs_diff` per compared tensor. Script modes: `--execute` (runs ON the H200 billed job), default = local preflight.

- [ ] **Step 1: Write the preflight test**

`tests/iq_local/test_parity_preflight.py` — runs the script's preflight path in-process: tiny synthetic config, CPU-vs-CPU comparison must produce max_abs_diff == 0 and valid summary structure. Assert summary JSON serializes with float diffs and the two fixture cases (prefill `[4, 64]`, decode `[1, 1]`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_parity_preflight.py -q`
Expected: FAIL (script missing).

- [ ] **Step 3: Implement the script**

`scripts/iq_mamba3_parity_job.py`:

1. Argparse: `--execute`, `--work-dir` (default `iq-transfer/parity`), `--seed-base 20261025`.
2. Local preflight (default): synthetic tiny config -> CPU port vs itself -> asserts exact-zero diffs -> writes summary skeleton (proves wiring without billing).
3. `--execute` (run on H200 job): loads real weights via the production CUDA runtime for selected layers, runs production prefill/decode; then runs `Mamba3CPU` prefill (looped steps) and decode on the same inputs (seed `20261025 + layer_idx`); computes and records `max_abs_diff` for outputs and state tensors; writes `prefill_l{L}.json` / `decode_l{L}.json`.
4. Mirrors `scripts/iq_hf_inference.py` job submission conventions (read that file and match its env/arg pattern).
5. Summary prints a table: tensor, shape, max_abs_diff, and a PASS/FAIL verdict against tolerance (tolerance documented in the script docstring: bf16 inputs through fp32 math, expected < 1e-3 for outputs, < 1e-3 for state after 64 prefill steps).

- [ ] **Step 4: Run preflight test**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_parity_preflight.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit (execution on H200 is a separate, user-triggered step)**

```bash
cd /home/shaolin/IQ
git add scripts/iq_mamba3_parity_job.py tests/iq_local/test_parity_preflight.py
git commit -m "feat: mamba3 cpu-vs-cuda parity fixture job with local preflight"
```

---

### Task 10: HybridRunner (greedy streaming decode with router mirror)

**Files:**
- Create: `iq_local/runner.py`
- Test: `tests/iq_local/test_runner.py`

**Interfaces:**
- Consumes: `CPUHybridForCausalLM` (Task 8), `StreamingWeightProvider` (Task 6), `CPUMamba3State` (Task 7), `RunIndex` (Task 3).
- Produces:
  - `class HybridRunner(model, provider)` with:
    - `select_experts(x: torch.Tensor, stage: int) -> tuple[torch.Tensor, torch.Tensor]` — router mirror: `scores = torch.sigmoid(router(x).float())`, `topk(scores, k=8, dim=-1)`. Must produce identical results to production `StableLatentMoE._route` on CPU (tested).
    - `generate(prompt_ids: list[int], max_new_tokens: int) -> list[int]` — greedy; drives the schedule layer-by-layer (not `model.forward`), interleaving: for each MoE layer, compute router scores from that layer's hidden state, call `provider.load_expert(stage, expert)` for the selected experts, run expert math functionally, continue.
    - Maintains `CPUMamba3State` per mamba layer across decode steps; bf16 activations, fp32 Mamba internals (per Task 7).
    - Layer schedule and residual wiring come from the model module itself (walk `model.layers` in order, dispatch by layer type), mirroring production `forward` composition (hybrid.py:780-850).

- [ ] **Step 1: Write the failing tests**

`tests/iq_local/test_runner.py`:

1. `test_select_experts_matches_production_route` — build a small production `StableLatentMoE` on CPU; compute its `_route(x)`; compute `runner.select_experts` on the same router weights; assert identical indices and values (exact equality on fp32 scores).
2. `test_generate_matches_model_forward` — tiny synthetic `CPUHybridForCausalLM` (all-agnostic config, zero or seeded weights); `runner.generate` over a 3-token prompt producing 2 new tokens must equal greedy argmax of `model.forward` on the same prefix computed autoregressively (bf16 tolerance atol 1e-3).
3. `test_expert_fetch_happens_between_layers` — with a counting `expert_fn`, assert during generation each MoE stage fetches exactly 8 distinct experts per token step and no more (router topk contract).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_runner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'iq_local.runner'`.

- [ ] **Step 3: Implement**

`iq_local/runner.py`: implement per Interfaces. Key points:
- The runner owns the decode loop: embed -> for layer in schedule -> mamba/attn/moe branch -> final norm -> lm_head -> argmax -> append.
- MoE branch: `gate, indices = select_experts(hidden, stage)`; build `expert_fn = lambda stage, e: provider.load_expert(stage, int(e))`; call the functional layer forward (from Task 8) with that closure so the same code path is used in runner and parity tests.
- Mamba branch: one `step` call per decode token; states kept in a dict keyed by layer index.
- Attention branch: production layers are stateless per call but cache projections between prefill and decode — for v1, recompute the full prefix inside each decode step only for attention layers if the production classes require KV; read the production classes' forward signature first and mirror their intended use; document the chosen approach in the runner docstring.
- No global mutable state; states reset via `reset()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_runner.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/shaolin/IQ
git add iq_local/runner.py tests/iq_local/test_runner.py
git commit -m "feat: streaming greedy runner with router mirror and layer schedule"
```

---

### Task 11: OpenAI-compatible local HTTP server

**Files:**
- Create: `iq_local/server.py`
- Create: `scripts/serve_iq_local.py`
- Test: `tests/iq_local/test_server.py`

**Interfaces:**
- Consumes: `HybridRunner` (Task 10), `StreamingWeightProvider` (Task 6), `RunIndex` (Task 3), tokenizer via transformers 5.17 Llama-3.1.
- Produces: `scripts/serve_iq_local.py` CLI mirroring `scripts/serve_iq_hf.py` (read it first and match its arg names where sensible): `--host`, `--port` (default 8000), `--cache-dir` (default `~/.iq-local-cache`), `--max-ram-experts`.
  - `GET /health` -> `{"status": "ok", "model": "<run prefix>", "counters": {...}}`
  - `POST /v1/chat/completions` -> OpenAI-compatible JSON: `{choices: [{message: {role, content}, finish_reason}]}`, greedy, `max_tokens` honored. Non-streaming only for v1.
- Stdlib `http.server.ThreadingHTTPServer` (no new dependencies).

- [ ] **Step 1: Write the failing test (offline synthetic model)**

`tests/iq_local/test_server.py` boots the server in-process (thread) over a tiny synthetic `CPUHybridForCausalLM` and a `FakeClient`-backed provider (pattern from Task 6 tests), then: `/health` returns 200 with `status=ok`; `/v1/chat/completions` with a 2-message chat returns 200, `choices[0].message.role == "assistant"`, string content, `finish_reason == "stop"`; unknown path returns 404.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_server.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'iq_local.server'`.

- [ ] **Step 3: Implement**

1. `iq_local/server.py`: `build_handler(runner, tokenizer, model_name)` returning a `BaseHTTPRequestHandler` subclass; JSON encode/decode; chat template via `tokenizer.apply_chat_template(..., add_generation_prompt=True)`; greedy generate via runner; errors -> JSON `{"error": {...}}` with 4xx/5xx; never logs or echoes `HF_TOKEN`.
2. `scripts/serve_iq_local.py`: argparse per Interfaces; wires `RunIndex` (with real `BucketClient`) + `LocalCache` + `StreamingWeightProvider` + `CPUHybridForCausalLM` + `HybridRunner`; loads checkpoint tensors through the provider lazily at first request (runner drives it); tokenizer `AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")`.
3. Match `scripts/serve_iq_hf.py` conventions for arg names and log lines (read the file; mirror, don't diverge).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local/test_server.py -q`
Expected: 1 passed (all offline).

- [ ] **Step 5: Commit**

```bash
cd /home/shaolin/IQ
git add iq_local/server.py scripts/serve_iq_local.py tests/iq_local/test_server.py
git commit -m "feat: openai-compatible local streaming server"
```

---

### Task 12: End-to-end smoke against the real bucket + validation doc

**Files:**
- Create: `docs/IQ_LOCAL_STREAMING_VALIDATION.md`
- Test: `tests/iq_local/test_end_to_end_smoke.py`

**Interfaces:**
- Consumes: everything wired in Tasks 1-11.
- Produces: a net-gated smoke test generating one short real completion through the real bucket + validation document recording startup, token rate, counters, and the H200 parity verdict placeholder.

- [ ] **Step 1: Write the net-gated smoke test**

`tests/iq_local/test_end_to_end_smoke.py` (marked `@pytest.mark.net`): with `HF_TOKEN` set, tiny `--max-ram-experts 4`-style config via provider kwargs; prompt "The capital of France is"; `max_new_tokens=6`; assert: generation returns >= 1 token; provider counters show `bucket_files > 0`; wall-clock of first token recorded; then `pytest.mark.net` skip logic from Task 1 keeps offline runs green.

- [ ] **Step 2: Run offline suite green, then net suite manually**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local -q`
Expected: all offline tests pass (net skipped).

Run (user must have `HF_TOKEN` exported): `cd /home/shaolin/IQ && IQ_NET_TESTS=1 .venv-local/bin/python -m pytest tests/iq_local/test_end_to_end_smoke.py -q`
Expected: 1 passed; record first-token latency + counters from output.

- [ ] **Step 3: Write the validation doc**

`docs/IQ_LOCAL_STREAMING_VALIDATION.md` sections: machine profile (N95, RAM/disk), environment (venv, versions), cold-start timeline (first token), steady-state token rate, cache usage after smoke (`usage()` numbers), counters table, parity fixture status (pending H200 run; insert results when Task 9 `--execute` is run), deviations from spec (list any implementation-time deviations discovered, e.g. exact upstream reference paths or production attr names).

- [ ] **Step 4: Run the full suite one more time offline**

Run: `cd /home/shaolin/IQ && .venv-local/bin/python -m pytest tests/iq_local -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /home/shaolin/IQ
git add tests/iq_local/test_end_to_end_smoke.py docs/IQ_LOCAL_STREAMING_VALIDATION.md
git commit -m "test: end-to-end streaming smoke and validation document"
```

---

## Plan Self-Review Notes (already applied)

- Task 5 test file is complete as written (no placeholders); roundtrip bound `max_abs/8` matches group-64 int4 worst case.
- Task 7 rotary uses the halved-branch reference with pairwise rotation off; implementer must confirm `rotate_pairwise=False` and the bias_q/bias_k permutation against the upstream reference file before porting.
- Task 7 core param names come from the checkpoint index (Step 1), not from memory.
- Task 6 `_expert_layer` resolves the stage's layer index from the index (files carry no stage-layer map), memoized.
- Task 10 attention decode strategy is explicitly delegated to reading the production classes' forward signatures; the plan does not guess KV semantics.
- Task 11/12 keep the real net-touching steps explicit and user-visible; the offline suite must stay green at every task boundary.

## Execution Instructions

- Execute with superpowers:subagent-driven-development (one task per subagent, task-to-task review) or superpowers:executing-plans (in-session, task-by-task).
- Each task ends at its commit step; never batch commits across tasks.
- Any implementation-time discovery that contradicts this plan (production attr names, upstream paths, config field names) is recorded in the commit message and in `docs/IQ_LOCAL_STREAMING_VALIDATION.md` deviations section.
