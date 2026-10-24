# GLM-5.3 weight expansion with frozen Mamba-3

Status: source coverage planning and opt-in expanded checkpoint compiler.
No new weights have been transferred yet.

## Actual original checkpoint

Source: `hf://buckets/kofecheckmate/mocha`.
The source-layer map selects 24 of GLM-5.3's 78 layers:
`0,3,6,10,14,17,22,23,26,30,34,37,42,44,46,50,54,57,62,64,66,70,74,77`.

The GLM donor has 3 dense and 75 sparse stages, with 256 routed
experts in each sparse stage. IQ has 32 Mamba, 24 context and 24 MoE
physical stages, with 32 experts per selected MoE stage.

- Mapped: 24/78 GLM stages (1 dense + 23 sparse)
- Omitted: 54 GLM stages (2 dense + 52 sparse)
- Original routed expert slots: 23 × 32 = 736 / 19,200
- Expanded candidate at 64: 23 × 64 = 1,472 / 19,200
- Expanded candidate at 128: 2,944 / 19,200
- Expanded candidate at 256: 5,888 / 19,200

These counts measure parameter source coverage, not language quality.
The old checkpoint does not record exact selected donor expert IDs;
do not claim a new compile preserves each of the previous 32 experts
until donor selection and serialized weights are compared.

## Original artifacts recovered in Hugging Face bucket

The local `find` output only returned the donor's
`~/IQ-checkpoints/glm53-1907c8ca8b85f895ce72/model.safetensors.index.json`.
It does **not** prove shard absence (the search did not include `*.safetensors`).

We confirmed the old reusable calibration on Hugging Face at:

```
hf://buckets/kofecheckmate/mocha/legacy/visionboxx-macha/calibration/
```

It includes `glm53_calibration.json`, stage-wise expert-usage `.npy`
and WARM coordinate-map `.safetensors` artifacts. It is the
original `weight_only` 24-stage/32-expert calibration, with canonical
target fingerprint. The official donor Mamba foundation and manifest
remain under:

```
hf://buckets/kofecheckmate/mocha/legacy/visionboxx-macha/mamba-foundation/
```

`scripts/reuse_glm53_calibration.py` can **rebind** this existing
weight-only calibration to a 64/128/256 expert recipient by changing
the target-config fingerprint and preserving every coordinate map,
expert-usage array and physical/source layer mapping. This avoids
redoing WARM Gram/eigendecomposition merely because expert capacity
changes. The script validates the source manifest's SHA-256 against
`.calibration_integrity.json`; `--verify-hashes` additionally verifies
all original files.

This requires the **original calibration folder accessible locally**,
with all map/usage files; it does **not** download a remote HF bucket.
The new directory contains symlinks to the old map files and therefore
is not self-contained. Keep the source folder unchanged and mounted.

```bash
python scripts/reuse_glm53_calibration.py \
  --source /path/to/ORIGINAL/calibration \
  --output /path/to/NEW/calibration-64 \
  --routed-experts 64 --verify-hashes
```

This operation copies **no weight tensors** and performs **no training**
or activation refinement. An expanded checkpoint must still be compiled
from the pinned GLM donor and original Mamba donor, then verified.

## Continue from the user's indexed donor shards

A local metadata check found **82/82 shards listed by the local index**,
but no `config.json`. That ratio alone does **not** prove this is the
complete GLM donor or contains the extra expert shards required by a
64-expert expansion. Compare against the original pinned GLM snapshot
metadata before expensive work:

```bash
DONOR="$HOME/IQ-checkpoints/glm53-1907c8ca8b85f895ce72"
hf download zai-org/GLM-5.3-BF16 config.json \
  --revision abf42e6c0e53bc29a875511301e7b7fcd220315b \
  --local-dir "$DONOR"
python scripts/check_glm53_source_inventory.py --local-donor "$DONOR"
```

Only the tiny `config.json` and official source index are downloaded by
these commands; the preflight reports actual original donor metadata
coverage. An incomplete 82-shard local subset is **not** automatically
enough for the expanded compile.

Official bucket CLI supports downloading only the calibration prefix
without syncing the entire `mocha` bucket:

```bash
hf buckets sync \
  hf://buckets/kofecheckmate/mocha/legacy/visionboxx-macha/calibration \
  "$HOME/IQ-calibration-original"
```

This **downloads potentially tens of GB of existing map tensors**;
use a filesystem with sufficient disk space and prefer remote mounted
bucket reuse if local download is unnecessary. Never pass `mocha`
bucket root to `sync` just to retrieve calibration.

Then build a candidate calibration that uses the original map files:

```bash
python scripts/reuse_glm53_calibration.py \
  --source "$HOME/IQ-calibration-original" \
  --output "$HOME/IQ-calibration-64" \
  --routed-experts 64 --verify-hashes
```

**New:** the expanded compiler accepts
`--reuse-foundation-checkpoint /path/to/original/mocha` instead of
`--mamba3-checkpoint /path/to/raw/mamba-donor`.
This copies the original 32 Mamba and lexical shards byte-for-byte,
verifying every SHA-256 without reloading the 1.5B original donor.
Use only one of the two flags. The original IQ checkpoint must be
accessible locally (or mounted on authorized compute); this change
does not download it. The compiler also requires genuine GLM donor
shards and does not trigger any Hugging Face jobs by itself.

## Important correction: local donor is a *subset or repackaged source*

The recent inventory check compared the local 82 indexed shards/3,457
tensor keys against the official pinned BF16 checkpoint: 282 shards
and 59,585 tensor keys. The filenames do not match and nearly every
official tensor key is absent. **Do not overwrite the 82-shard index**,
and do not use it as `--checkpoint` for official GLM streaming mode.

IQ already has `HubSafetensorsSource`, which downloads missing
**official shard names on demand** from the pinned donor revision.
Use a separate, metadata-only donor directory with the **official**
`config.json` and `model.safetensors.index.json`. The streamed
checkpoint need not contain all 282 original shards in advance.

### Plan exact required official shards, without downloading weights

```bash
cd ~/IQ-transfer-review
META="$HOME/IQ-checkpoints/glm53-official-metadata"
mkdir -p "$META"
hf download zai-org/GLM-5.3-BF16 \
  config.json model.safetensors.index.json \
  --revision abf42e6c0e53bc29a875511301e7b7fcd220315b \
  --local-dir "$META"

python scripts/plan_glm53_selective_shards.py \
  --donor-config "$META/config.json" \
  --donor-index "$META/model.safetensors.index.json" \
  --calibration hf://buckets/kofecheckmate/mocha/legacy/visionboxx-macha/calibration \
  --transplant-manifest hf://buckets/kofecheckmate/mocha/transplant_manifest.json \
  --routed-experts 64 \
  --output "$HOME/glm53-official-shards-for-iq64.json"
```

This reads only the original calibration manifest, its integrity file,
23 tiny expert-usage arrays, and the donor **metadata**. It uses the
compiler's existing `_compile_tensor_keys` plus `_shards_for_keys`
implementation so the results match the actual expert selection.
**It does not download any donor weights, upload or mutate a bucket,
or start GPU compute.**

The JSON plan includes exact official shard names and selected expert
IDs. When fully verified, `glm53-compile --streaming-source` can later
use the separate official metadata checkpoint path and download only
the required shards. Note that a shard can contain tensors that are
not selected; shard-level streaming still downloads the whole file.

## Estimate download and disk cost before any shard transfer

The official donor has 282 BF16 shards. The exact 64-expert
metadata plan from the original cloud calibration requires **103**
official shards containing **4,738** tensor keys, but there is no
assurance those 103 shard files are small. Official shard files can be
about 5.3 GB each. Fetch file sizes using `HfApi.model_info`
without downloading the weights:

```bash
python -m scripts.estimate_glm53_shard_bytes \
  --plan "$HOME/glm53-official-shards-for-iq64.json" \
  --disk "$HOME"
```

This gives the exact *once-per-shard* estimated transfer size from
current model metadata and reports available local disk. Network
retries, cache copies and additional IQ/calibration artifacts can
increase actual disk and traffic. It does not allocate a GPU or
download shard weights. CPU-only unit tests:

```bash
python -m unittest discover -s tests \
  -p 'test_glm53_shard_size_estimate.py' -v
```

A 64-expert selected sparse layer can load multiple gigabytes of
source expert matrices before nonlinear WARM refactor computations,
with potentially much larger peak RAM due to float32 intermediates.
A 16 GB RAM workstation should **not** be the default full-compile
environment; calculate requirements before selecting compute.
Streaming full shards reduces *peak* disk but does not remove
per-shard network transfer. Do not start paid compute implicitly.

## Alternative under investigation: partial safetensors range reads

The 64-expert shard plan requires **103** official shards totaling
**552,128,300,840 bytes (514.21 GiB)**. This exceeds the workstation's
currently available 508.24 GiB and does not include the ~17.5 GB
calibration or destination checkpoint. **Do not bulk-download those
103 shards onto the workstation.**

Safetensors encodes named tensor byte offsets in its JSON header.
The Hugging Face/Xet protocols document byte-range requests. To verify
that this pinned BF16 donor serves safe partial requests, run:

```bash
python -m unittest discover -s tests \
  -p 'test_glm53_remote_range_probe.py' -v
python -m scripts.probe_glm53_remote_ranges \
  --plan "$HOME/glm53-official-shards-for-iq64.json" \
  --max-shards 1
```

The probe fetches only the first 8 bytes and the JSON safetensors
header of a single pinned official shard. It **refuses to download
anything** if the server returns HTTP 200 instead of 206, omits the
Content-Range header, or supplies invalid offsets. It validates file
size and tensor header bounds. It does **not** read any tensor payload,
prove all 103 shards support ranges, or implement tensor streaming.
The latter is a separate next step if the probe works.

No GPU job or weight download is started by these commands.

## Single command to install and audit GLM transfer dependencies

The transfer requirements now declare `requests>=2.32,<3` for the
remote HTTP range probe. Rather than relying on shell activation or
reinstalling the existing CPU PyTorch, run:

```bash
cd ~/IQ-transfer-review
bash scripts/setup_glm53_workflow.sh
```

The setup script uses
`~/IQ-integrate/.venv-iq-client/bin/python` **directly**, verifies
that it is an isolated venv, installs the declared dependencies,
runs `pip check`, imports every GLM transfer CLI/utility module, and
runs the relevant CPU unit tests. It neither upgrades/reinstalls
PyTorch nor downloads donor weights. `IQ_VENV=/path/to/venv` or
`IQ_PYTHON=/path/to/venv/bin/python` overrides the interpreter.

A clean preflight prints
`"status": "ready_for_glm53_metadata_workflow"`.
If a dependency is missing, the preflight prints its name and cause,
rather than deferring discovery until a long-running compile.
Additional, optional transformers runtime used for full GLM activation
capture is **not installed automatically**, because the current
metadata/weight-transfer commands do not require it.

Run the safe header-only range probe **without activating the venv**:

```bash
"$HOME/IQ-integrate/.venv-iq-client/bin/python" \
  -m scripts.probe_glm53_remote_ranges \
  --plan "$HOME/glm53-official-shards-for-iq64.json" \
  --max-shards 1
```

## New tools

- `scripts/plan_glm53_weight_extension.py`: fetches GLM and IQ
  metadata, checks the source-layer map, and produces a JSON coverage plan.
- `canonical_glm53_config(routed_experts=64)`: opt-in configuration.
  Valid counts are 32, 64, 128, or 256. All other geometry is frozen.
- `iq_transfer.cli glm53-bootstrap-calibration --routed-experts 64`:
  generate calibration compatible with the candidate configuration.
- `iq_transfer.cli glm53-plan-compile-shards --routed-experts 64`:
  calculate exactly which donor shards the candidate compile needs.
- `iq_transfer.cli glm53-compile --routed-experts 64`:
  build into an EMPTY, NEW output directory, after donors and
  matching calibration are available. The original checkpoint remains
  unchanged. New manifests include selected donor expert IDs.
- `scripts/verify_mamba_frozen.py`: compare hashes of all 32 Mamba
  checkpoint shards and the pretrained lexical/global shard before
  treating a candidate as preservation-safe. Identical weights do not
  guarantee identical activations in the hybrid model.

## Run the CPU-only metadata audit

```bash
python scripts/plan_glm53_weight_extension.py --routed-experts 64
python -m unittest discover -s tests -p 'test_glm53_weight_extension.py' -v
python -m unittest discover -s tests -p 'test_glm53_expert_capacity.py' -v
python -m unittest discover -s tests -p 'test_glm53_mamba_preservation.py' -v
```

The planning script uses Hugging Face metadata only; no GPU job.
To use local copies, specify `--donor-config`,
`--transplant-manifest` and `--iq-config`.

## Explicit boundary: unmapped GLM layers

Adding experts to the 24 existing GLM-derived stages does NOT transfer
the other 54 GLM layers. Mapping all 78 source layers would require
new physical context+MoE slots, and hence a different model topology.
It would also violate the current Mamba-dominance condition with only
32 Mamba physical slots. Such an expansion must be designed, validated,
and budgeted separately; do not silently rewrite the production
architecture or claim mathematically exact donor preservation.

Likewise, increasing expert capacity may greatly increase model size,
memory use, inference latency and compile work. No new model should
be promoted without tokenizer checks, Mamba hash parity, first-token
logit/activation diagnostics and held-out language evaluation.

No compute allocation, training job, model download or checkpoint
overwrite is performed by these tools.
