# IQ Local Streaming Inference — Validation

Status: **implementation complete (Tasks 1–11 committed); live smoke and H200
parity execution pending user-run net steps.**

Plan: `docs/IQ_LOCAL_STREAMING_PLAN.md`. This document records the machine,
environment, measured timings, cache/counter behavior, the parity fixture
status, and every deviation found during implementation.

## 1. Machine profile

| Item | Value |
| --- | --- |
| CPU | Intel N95 (4 cores) |
| RAM | 15 GiB total (≈6 GiB available under normal load; base-weight load needs ~8–10 GiB bf16) |
| Disk | 914 GiB volume, 62 GiB free (93% used) — shard cache grows by the full non-expert shard set on first smoke |
| OS | Linux, bash |

## 2. Environment

| Item | Value |
| --- | --- |
| Venv | `/home/shaolin/IQ/.venv-local` |
| Python | 3.12.3 |
| torch | 2.15.0+cpu |
| transformers | 5.17.0 |
| safetensors | 0.8.0 |
| Bucket | `kofecheckmate/iq-glm53-all256`, run `IQ/runs/glm53-full78-local-20261025a` |
| Auth | `HF_TOKEN` from the environment only (never logged/persisted) |

## 3. Test surface (offline, all committed)

`pytest tests/iq_local -q` → **53 passed, 3 skipped, zero warnings** (skip =
3 net-marked pre-existing tests; the Task 12 smoke skips offline via the
same `IQ_NET_TESTS` gate). Coverage by task: bucket (2), cache (5), index
(5), provider (9), quant (6), mamba CPU port (10), CPU hybrid model (5),
parity preflight job (4), runner (4), server (2), plus integration/protocol
suites.

## 4. Cold-start timeline (first token)

**PENDING — requires `IQ_NET_TESTS=1` + `HF_TOKEN` run of
`tests/iq_local/test_end_to_end_smoke.py`.** The smoke prints
`IQ_SMOKE base_load_seconds=… first_token_seconds=… full_6_token_seconds=…
tokens_per_second=… counters=… cache_bytes=… cache_entries=…`; paste the
line here after the run.

Expected composition: manifest download (4 small files) → CPUHybrid module
construction (~150–280 s on this N95 for the full 235-layer schedule, measured
in Task 8) → base-weight shard fetch through the provider (full non-expert
shard set, NVMe-cached and pinned) → first token (prompt prefill + 1 decode
step; expert fetches quantize-on-first-use, then serve from NVMe/RAM).

## 5. Steady-state token rate

**PENDING — same smoke run.** N95 expectation from Task 7's CPU Mamba port
timings and Task 10's full-prefix attention decode: order of a few seconds
per token; the smoke's `tokens_per_second` is the recorded number.

## 6. Cache usage after smoke

**PENDING — same smoke run** (`LocalCache.usage()` → bytes/entries; experts
are stored int4-packed (~1/8 of bf16 expert bytes), non-expert shards bf16
pinned). After a full smoke with `max_ram_experts=4`, expect: entries =
non-expert shard files + 16 int4 expert files for the stages the prompt
touched; RAM LRU holds ≤ 4 dequantized expert triples.

## 7. Counters table

Provider counters (`StreamingWeightProvider.counters`) exposed at
`GET /health`:

| Counter | Meaning |
| --- | --- |
| `bucket_files` | Objects downloaded from the bucket (bf16 shards + raw expert files) |
| `bucket_bytes` | Bytes downloaded |
| `nvme_hits` | Shard reads served from the NVMe cache |
| `ram_hits` | Expert triples served from the RAM LRU |
| `int4_converted` | Raw expert files quantized to int4 and persisted |

**PENDING — smoke values.**

## 8. Parity fixture status

The Mamba-3 CPU-vs-CUDA parity job (`scripts/iq_mamba3_parity_job.py`) is
implemented and its offline preflight passes (exact-zero diffs on both
prefill `[4,64]` and decode `[1,1]` cases; fixtures written). The billed H200
execution (`--execute`, comparing production `Mamba3MIMOState` against
`Mamba3CPU` on real `layers.{0,3}.mamba.core.*` weights, tolerance 1e-3) is
**PENDING — user-triggered**. Insert the printed PASS/FAIL table and fixture
paths here when run.

## 9. Deviations from spec

Implementation-time deviations discovered against the plan/production
references (all verified against source before editing):

1. **Routed experts live in latent width, not hidden width.** Production
   `StableLatentMoE` routes and dispatches experts over `latent_size`
   features; routed `SiTUExpert` projections are `[inter, latent]` /
   `[latent, inter]` (`iq_model/mlp/latent_moe.py`). The plan's prose
   ("fetch gate/up/down for the expert") carried no widths; Task 8's bitwise
   parity test pinned the real shapes.
2. **`FunctionalStableLatentMoE._route` mirrors production routing by
   delegation**, not by re-implementing sigmoid→topk: `select_experts` and
   the functional layer both call the same `_route`, so Task 10's
   "identical results to production" is exact (`torch.equal`), not
   tolerance-based.
3. **Runner attention decode recomputes the full prefix per attention layer**
   (production layers are stateless per call and derive positions
   internally — `iq_model/attention/context_dense.py:51`,
   `iq_model/attention/compressed.py:1009`); the plan delegated this
   strategy to reading the production classes. O(T) attention work per layer
   per step, exact semantics.
4. **AttnRes decode rebuilds per-token state** instead of carrying one
   incremental mixer state: `BlockAttentionResidual` is per-position
   independent (`iq_model/residual/attnres.py:101-169`), so per-token
   init/read/advance/finalize equals the full-sequence path bit-for-bit.
5. **`HybridRunner.generate` grew a `stop_tokens` kwarg in Task 11** (not in
   the Task 10 interface list): the server needs `finish_reason`
   stop/length; default `None` keeps Task 10 behavior.
6. **`scripts/serve_iq_local.py` adds `--cache-budget-gb`** beyond the
   plan's arg list: `LocalCache` requires a byte budget; 512 GiB default is
   effectively unbounded on the 914 GiB volume.
7. **Tiny synthetic test configs are mamba-dominant** (2 mamba ≥ 1 attention
   + 1 extra): `HybridSchedule` requires the mamba count to EXCEED the
   attention count (`iq_model/architecture.py`), so a single mamba +
   single attention schedule is unconstructible; hidden 64 with
   latent/inter 64 in the server test because the int4 quantizer requires
   last dim divisible by group 64 (`iq_local/quant.py:22`).
8. **Whole-repo test suite has 3 pre-existing failures** outside this plan's
   scope (`tests/test_executor.py`,
   `tests/test_glm53_inspector.py::test_validates_real_glm_attention_and_moe_layout`,
   `tests/test_transfer_job.py::TransferJobTests`) — a
   `MappingTensorSource is not iterable` bug in `iq_transfer/executor.py:92`
   — plus 1 pre-existing UserWarning in `tests/test_glm53_mohawk.py:90`.
   Confirmed failing identically with all plan commits stashed. The
   `iq_local` suite itself is fully green.

## 10. Net steps remaining (user-run)

1. `IQ_NET_TESTS=1 HF_TOKEN=… .venv-local/bin/python -m pytest
   tests/iq_local/test_end_to_end_smoke.py -q -s` — then paste the printed
   `IQ_SMOKE` line into sections 4–7.
2. H200 parity execution per Task 9 report (`scripts/iq_mamba3_parity_job.py
   --execute …`) — then paste results into section 8.
