# GLM-5.3 → IQ full-78 GPU transfer: fail-closed runbook

Status: **preparation only; DO NOT launch a full paid transfer** until every launch gate below passes. This workflow is a weight projection into IQ, not a proof of model quality, tokenization compatibility, or end-to-end teacher equivalence.

## Immutable contract

- Teacher: official `zai-org/GLM-5.3-BF16` at pinned revision `abf42e6c0e53bc29a875511301e7b7fcd220315b`.
- Mamba donor: `state-spaces/mamba3-mimo-1.5b` at pinned revision `bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec`.
- All GLM source layers **0..77**, with **75 genuine MoE source stages × 256 experts = 19,200**. The first 3 dense stages have **768 function-neutral IQ routed slots**, hence **19,968 total on-disk expert-slot files**.
- IQ architecture has **78 context + 78 MoE + 79 Mamba placements**. Only 32 Mamba source layers are pretrained; do not label the other 47 as independently pretrained.
- Output in `hf://buckets/kofecheckmate/iq-glm53-all256/IQ/runs/<RUN_ID>`.
- Preserve prior run `glm53-full78-20261024-c` for forensic inspection only. It predates the run-provenance receipt, so **do not automatically reuse its saved experts** after calibration repair. Begin with a new RUN_ID on a fresh output; resume is permitted only for runs created with a matching provenance record. Never have two writers.

## State machine and mandatory gates

| Gate | Must pass before proceeding | On failure |
|---|---|---|
| G0 — identity | Code commit, pinned donor revisions, canonical target fingerprint, run ID, previous job stopped | Refuse to adopt old shards; investigate mismatch |
| G1 — capacity | Verify the actual output bucket with `scripts/check_glm53_storage_policy.py`: private requires current provider attestation for ~630 GB; public requires `IQ_ACKNOWLEDGE_PUBLIC_BEST_EFFORT=1` and accepts that HF **does not guarantee** 600 GB under free fair use | **STOP** if visibility cannot be confirmed, the chosen policy is not acknowledged, or durable writes fail |
| G2 — calibration | 78 strictly ordered source mappings, 75 usage vectors, expected **866 artifacts**, all nonempty, plus `calibration_sha256.json` verified for EVERY file from a separate read-only worker | **STOP**, run calibration-only repair and republish, never compile against a partial manifest |
| G3 — GPU preflight | CUDA present, tiny device math smoke, sufficient VRAM; bounded range and tensor checks are performed inline by the full compiler | Retry bounded transient range errors during the real job; for OOM reduce tile/L4→L40S→A100; **never silently switch to CPU** |
| G4 — transfer | One writer, exact historical source IDs, atomic shard writes, per-slot progress and a paid-worker deadline | Exit with error after safe shard boundary; keep completed files; resume only after job is confirmed stopped |
| G5 — stage durability | All 256 routed slots, shared/common/context, projection report per source layer independently checked; no missing or corrupt shards | Recompute ONLY the missing stage/slot, do not overwrite valid shards |
| G6 — final | Strict expected model state keys, 78 context, 78 MoE, 79 Mamba, **19,968 expert slots**, final index and manifest, independent re-read from a different worker | Fail release; fix missing files and resume; checkpoint is not yet inference-validated |

The old `COMPLETED` job status or a local `/out` file alone is **not durability evidence**. Public bucket visibility does **not** make Hugging Face GPU Jobs free. Hugging Face Jobs discard the worker filesystem at exit; bucket volumes are durable but their persistence must be independently confirmed.

## Before spending on any GPU

From the **repository root**, run:

```bash
python -m unittest discover -s tests -p 'test_glm53_full78_gate.py' -v
python -m unittest discover -s tests -p 'test_glm53_atomic_shards.py' -v
python -m unittest discover -s tests -p 'test_glm53_resume.py' -v
python -m unittest discover -s tests -p 'test_glm53_gpu_projection.py' -v
python -m compileall -q iq_transfer
bash -n scripts/repair_glm53_full78_calibration.sh
bash -n scripts/resume_glm53_full78_mounted_cloud.sh
```

Get an explicit **provider-verified free-space statement** for the target bucket. Record `checked_at` (UTC ISO-8601), `source` (billing quota/provider evidence), `bucket` and `confirmed_free_bytes` in a private JSON file. The required bytes must be computed, not guessed:

```bash
BYTES="$(python -c 'from scripts.full78_storage_preflight import required_private_bytes;print(required_private_bytes())')"
python scripts/glm53_full78_gate.py storage \
  --attestation /secure/private-bucket-capacity.json --required-bytes "$BYTES"
```

This attestation must be backed by an actual quota check; typing a large number is not proof.

## Calibration-only recovery (do this first)

Do **not** run the full transfer until all map files are restored. The previous bucket snapshot contained stages 50–77 but lacked stage 0–49 artifacts. Complete and validate the 78-layer calibration on a controlled GPU job, then create a cryptographic receipt from the complete *local* map directory:

```bash
python scripts/glm53_full78_gate.py calibration --root /tmp/full78-calibration
python scripts/glm53_calibration_sha256.py make --root /tmp/full78-calibration
hf buckets sync /tmp/full78-calibration \
  hf://buckets/kofecheckmate/iq-glm53-all256/calibration78/NEW_CALIBRATION_RUN
```

Use a **separate read-only worker / separately mounted bucket** after the publisher job ends:

```bash
python scripts/glm53_full78_gate.py calibration --root /verified-calibration
python scripts/glm53_calibration_sha256.py verify --root /verified-calibration
```

If any name/size/hash differs, do not start transfer. Never replace historical reference files silently. The existing `scripts/repair_glm53_full78_calibration.sh` prepares and syncs a repair, but its same-process remote listing is **not** an independent checksum audit.

## Full GPU resume (only after gates G0–G3)

For the first approved execution, use `scripts/start_glm53_full78_gpu_from_verified_calibration.sh` with a **new** RUN_ID and the verified `CALIBRATION_RUN_ID`. For subsequent segments of that *same provenanced run*, use `scripts/resume_glm53_full78_mounted_cloud.sh` only after independent inventory checks. Use pinned code and exactly one writable output mount. Job arguments, token permissions, storage volume and CUDA device must be reviewed before launch. Under the user's strict **$0** compute constraint, never launch a billable HF Jobs H200 (currently $5/hour) or any other GPU Job unless credits/grants are verified to cover the *entire scheduled run*. H200 (141 GB) and RTX Pro 6000 Blackwell (96 GB) are preferred if donated/credit-funded. Free ZeroGPU offers only short daily Gradio allocations, not a long-lived H200 worker. Record completed source stages and bytes written during the continuous full transfer, never isolated benchmarks. See `docs/GLM53_ZERO_COST_GPU_RUNBOOK.md`.

The guarded entry point is `scripts/resume_glm53_full78_mounted_cloud.sh`. It requires:

- `RUN_ID`: ID of the validated incomplete output.
- `IQ_CLOUD_WORKER=1`, `IQ_CONFIRM_PREVIOUS_JOB_STOPPED=1`, `IQ_PROJECTION_DEVICE=cuda`.
- For a private destination: the actual verified free capacity attestation. For the current **public** bucket: `IQ_ACKNOWLEDGE_PUBLIC_BEST_EFFORT=1`, acknowledging fair-use refusal risk. The script checks the actual provider bucket visibility.
- `IQ_STORAGE_ATTESTATION`: path to the private, recent attestation **inside** the worker.
- `IQ_MAX_RUN_SECONDS=14400` (4-hour segment; allowed 900–82800 seconds), independent of the Hugging Face job's hard timeout.
- Correct read-only source calibration mount `/hf-bucket`, writable output mount `/out`, and HF credentials with only required access.

The compiler checks `IQ_TRANSFER_DEADLINE_EPOCH` before every expert and stops with a clear checkpointed-budget message, leaving all committed expert files eligible for the next run. **It deliberately exits nonzero** on a budget stop. Human/operator checks status and files, verifies the prior job stopped, then explicitly launches the next segment.

## Multi-expert CUDA execution (implemented, not yet cloud-validated)

- Sparse MoE compilation uses a bounded CUDA batched GEMM route by default
  when `IQ_PROJECTION_DEVICE=cuda`. The actual output remains one atomic
  safetensors shard per expert for safe resume.
- `IQ_MOE_GPU_BATCH_EXPERTS=auto` automatically requests **24** experts per H200 batch, **16** on 96 GB Blackwell, **8** on L40S, or **4** on smaller GPUs. The corresponding bounded BF16 donor source staging budgets are **2048/1536/1024/512 MiB**. Geometry may reduce each batch, and CUDA OOM automatically splits already downloaded tensors without repeating network reads.
- `IQ_MOE_GPU_PREFETCH=1` (default) enables **one** future remote batch
  using a separate `requests.Session`; set to 0 to disable I/O overlap.
- Every GPU batch emits source-payload throughput and CUDA peak allocated
  memory as part of the **real full78 transfer**, not a standalone benchmark.
- Dense stages, common/router transforms and shared experts retain their
  original checked processing path. GPU runtime and full model inference
  correctness are not asserted merely because these tensor kernels run.

## Full-run efficiency policy

Do not launch separate one-expert or one-layer GPU benchmarks. Once preflight,
calibration hashes, source revisions, durable free capacity and provenance pass,
launch the **full 78-layer / 256-expert-per-sparse-layer** job immediately.
The compiler reads/validates actual donor tensors only as it transforms them;
there is no extra network read of stage-3 slot-0 or slot-41 on startup.
Observe stage-level throughput, GPU/CPU transfer time, network throughput and
peak memory in the same job. Resume only genuinely incomplete shards if a
worker budget expires. The sparse routed expert CUDA path now uses bounded multi-expert batched GEMMs;
other portions still use staged per-tensor work. Validate numerical parity
and sustained full-run progress before promoting the new path.

## Fallback classification

1. **Missing quota or calibration hash** → permanent PRECONDITION failure; no full GPU job. Fix storage/calibration only.
2. **HTTP 429/5xx, timeout, byte-range mismatch** → bounded retries for transient failures; if exhausted, stop and preserve completed shards. Never silently download unbounded full teacher shards.
3. **CUDA unavailable** → reschedule on a CUDA worker; no automatic costly CPU fallback.
4. **CUDA OOM** → tile halves down to minimum; if still OOM, safely exit and use a larger GPU without redoing completed experts.
5. **Write failed, incomplete `.partial`, corrupt or mismatched shard** → fail closed and require explicit recovery; no deletion or silent overwrite. Safetensors use same-directory temporary write followed by atomic rename and only then index entry.
6. **Job timeout / budget stop** → wait for STOPPED confirmation, run independent bucket inventory, resume missing slots in a new time-boxed worker. Never start parallel writers.
7. **Final missing files or invalid index** → no `COMPILE_COMPLETE` release; recompute missing stage(s) and rerun comprehensive index validation.
8. **Unverified tokenizer or no inference evaluation** → compiled weights may be durable, but don't call the model production/inference ready.

## Completion proof

Run independent `checkpoint-stage` gates for all layers and `checkpoint-complete` only after the compiler wrote its final index and manifest:

```bash
for i in $(seq 0 77); do
  python scripts/glm53_full78_gate.py checkpoint-stage --root /verified-IQ --layer "$i" || exit
done
python scripts/glm53_full78_gate.py checkpoint-complete --root /verified-IQ
```

These prove expected **file inventory**, not language-model functional correctness. A separate safetensors shape/key/dtype audit, tokenizer packaging and inference evaluation remain required prior to deployment. Do not use `scripts/audit_iq_checkpoint.py` as the only full78 verifier: its historical 24-stage schedule assumptions must be updated.

## Never again

No standalone single-expert GPU benchmarks, no 24-hour non-resumable worker, no inferred storage quota, no manifest-only calibration success, no silent CPU fallback, no writing an index before the shard exists, no overwrite of an apparently valid historical shard, no concurrent resume jobs, and no declaration of successful transplant without external durable file verification.
