# IQ235 full-model inference: consolidated blockers

This is a source-and-metadata audit, not a prepaid GPU launch or an inference
certificate. The physical model is 235 layers (79 Mamba3, 78 MoE, 21 CSA,
57 HCA), with 256 routed experts per MoE stage and 4096 hidden dimensions.

Source checkpoint: Hugging Face bucket kofecheckmate/iq-glm53-all256,
run IQ/runs/glm53-full78-local-20261025a.
- 63164 indexed tensors across 20283 shard files
- 539388468802 indexed total bytes; no full payload scan has been completed
- Original source tokenizer vocab: 128256 IDs
- Intended GLM-5.3 tokenizer vocab: 154880 IDs
- Original COMPILE_COMPLETE.json status: FULL78_COMPILED_NOT_YET_INFERENCE_VALIDATED

## Confirmed without GPUs

The source config and index hashes match pinned metadata. Six real remote
SafeTensors headers (global, native, Mamba3, context, MoE-common, MoE-expert)
match their indexed tensor names, dtype encodings, file lengths and offsets.
This samples SIX shards, not the entire 20283-shard checkpoint. No neural
tensor payloads have been read or validated.

Conservative lexical candidate alignment, frozen-backbone training loss,
source body loading primitives, remote SHA256 streaming, and dependency guards
have CPU tests. No successful model-quality evaluation exists.

## Execution blockers found together

1. Source data integrity: no approved full 20283-shard SHA256 reference
   exists. Generate a snapshot only with explicit bandwidth/cost approval,
   then review and pin its fingerprint. Reading all bytes proves consistency
   with a snapshot, not that compilation retained donor model capabilities.
2. Actual model/schema compatibility: the Mamba3 MIMO constructor requires
   CUDA, pinned upstream commit and a working GPU kernel. The full runtime
   63164-key model/index match is not independently demonstrated. Runtime
   shape, dtype, device placement and peak VRAM remain to be checked.
3. Runtime wiring: load_phase_a and deployed Baseten/Modal entrypoints
   explicitly use architecture_only. The stage excludes the new full235
   remote weight loader. A passing CPU test cannot load pretrained weights.
4. GLM lexical overlay: GLM token IDs differ from source IDs. Both
   154880x4096 untied matrices need actual pretrained initialization,
   training, and held-out validation. Candidate token correspondences are
   not interchangeable with pretrained outputs.
5. Quality proof: there is no end-to-end full-model pretrained generation
   test or calibrated teacher/student behavior evaluation. A single random
   architecture-only forward would NOT prove intelligence retention.
6. Compute: 8 H200 GPUs plus a pinned TileLang/Mamba kernel and a private
   provider secret are required by the existing deployment configuration.
   No account capacity, budget approval or successful remote execution
   has been verified.
7. Deployment provenance: the Baseten build previously installed Mamba3
   from a local path, losing pip's pinned Git VCS provenance. Source code
   now uses the exact pinned VCS install and validates direct_url.json, but
   that fix has not been tested in a real GPU build.

## Practical milestone split

A. SOURCE-tokenizer pretrained proof, optional: verify and inject the full
original body AND its original 128256x4096 embedding/head, then run with the
original Mamba/Llama tokenizer on a remote GPU. This avoids GLM lexical
retraining for the FIRST source-vocabulary test, but still requires the
complete trusted loader, exact model schema and full GPU/kernel checks.
This is NOT the requested GLM-tokenizer checkpoint and no quality claim
can be made until generation passes.

B. GLM-tokenizer pretrained proof: complete all of A's body/runtime gates,
then train and validate the two aligned GLM lexical matrices and run
full-model held-out generation.

## Commands: read-only, no model payloads or GPUs

Run from the repository root:
- python -m scripts.iq235_bulk_readiness
- python -m scripts.iq235_bulk_readiness --remote-header-samples
- python -m scripts.iq235_preflight

The remote-header-samples option needs Hugging Face metadata access. It
does not scan the full weight payload or start any GPU.

Do not paste API keys into GitHub, logs, or chat. Provision secret variables
in the provider only after explicit compute and billing authorization.

## Completion criterion

Actual pretrained inference cannot be announced until a trusted worker has
loaded verified pretrained weights, the source/target tokenization and tensor
geometry agree, full kernel execution is observed, and meaningful held-out
generation succeeds. This outcome has NOT been demonstrated.

## Separate original-Llama-ID runtime (code implemented, still blocked)

The Baseten entrypoint has an opt-in `IQ_PHASE=pretrained_source` path distinct
from the default `IQ_PHASE=architecture_only` probe. It loads the pinned
public Llama 3.1 mapping at the original source vocabulary capacity (128256),
and prepares to inject **all** indexed source tensors, including the two
original-ID lexical matrices. These are not GLM-tokenizer weights.

The path **cannot** run until explicit billable H200x8 approval is passed via
`IQ235_AUTHORIZE_H200X8=I_APPROVE_BILLABLE_H200X8` AND an independently
reviewed 20,283-shard SHA-256 manifest is anchored in
`checkpoint_metadata.py`; that pin remains `None`.
Once approved, a remote worker needs the reviewed manifest file at
`IQ235_APPROVED_MANIFEST_PATH` and scratch at `IQ235_SHARD_SCRATCH_DIR`.
Neither can be substituted for independent content verification. The default
Truss config remains architecture-only; no GPU is started by this change.

The existing OTel exporter is reused. Explicit source loading emits
`iq235.source.construct_full235`,
`iq235.source.install_verified_checkpoint`,
and `iq235.checkpoint.stream_verified_shards` spans, with bounded progress
(events at shard 1, every 250, and completion). Existing per-layer execution
events are preserved. Setting `IQ235_NUMERIC_DIAGNOSTICS=1` also checks
*each actual layer output* for nonfinite values and emits numeric aggregates
and host-side forward duration. The optional checks synchronize CUDA and are
slow; host duration is **not** GPU kernel latency. No raw text, token IDs,
weight data, or credentials are exported by the new instrumentation.

**Not yet proven:** worker topology allocation, verified 539GB payloads,
Mamba3 numerical correctness, actual GPU forward, and held-out generation.
No production-readiness assertion is implied by the new path.

## One-pass SHA snapshot during exploratory inference

`IQ_PHASE=pretrained_source_exploratory` enables a **separate** experimental
path only with explicit H200x8 authorization, 539GB source-read approval,
and unpinned source acceptance. An unused **absolute**
`IQ235_UNREVIEWED_MANIFEST_PATH` on a **durable/persistent worker mount**
is mandatory. The original baseline `IQ_PHASE=architecture_only` remains
the default and uses no pretrained checkpoint.

For a first exploratory run, `snapshot_remote_shard` computes a SHA-256 while
reading and staging each source shard **once**, validates the real file header,
and installs the corresponding model tensors after structural validation. Once
all 20,283 shards and 63,164 tensors are processed and sampled lexical row
checks pass, it creates a complete **unreviewed** manifest atomically without
overwriting any prior result. The manifest's SHA-256 is returned in the probe
response. Do not let a worker exit before persisting that output manifest.

A self-generated digest during that same download is **not independent proof**
of checkpoint provenance. It cannot detect a pre-existing tampered source.
After the experiment, review source provenance, exported telemetry, all SHA
digests, and generated outputs **in bulk**; only then independently pin the
manifest for subsequent `IQ_PHASE=pretrained_source` verified runs. Never
call a successful exploratory response a production/quality attestation.

Experimental mode requires: `IQ235_AUTHORIZE_H200X8=I_APPROVE_BILLABLE_H200X8`,
`IQ235_AUTHORIZE_539GB_READ=I_AUTHORIZE_539GB`,
`IQ235_EXPLORATORY_UNPINNED=I_ACCEPT_UNATTESTED_SOURCE`, and
`IQ235_UNREVIEWED_MANIFEST_PATH=/durable/mount/new_snapshot.json`.
No flags are set automatically, and neither the checkpoint bucket nor the
model serving configuration is modified to switch into this mode.

The experimental Baseten prediction also writes a one-time
`<snapshot>.inference.json` beside the manifest with outcome, generated IDs,
completion, failure type and verified-stage metadata (not the prompt or
raw exception message). The snapshot and report must be on durable storage;
filesystem persistence cannot be inferred from the provider's replica state.
