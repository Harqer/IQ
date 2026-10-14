# IQ NVIDIA Megatron Bridge + Multi-Storage Client

**Status:** executable donor conversion integration and multi-bucket MSC configuration
are committed. This **does not** mean donor weights have been converted into
IQ's custom 78-block fused model. Bucket contents are managed by the owner.

## Official references

- [Megatron Bridge checkpoint conversion](https://docs.nvidia.com/nemo/megatron-bridge/nightly/bridge-guide.html)
- [NVIDIA Nemotron 3 Super checkpoint support](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/nemotron/nemotron3-super.html)
- [NVIDIA GLM-5/5.2/5.3 shared GLM5Bridge](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/glm/glm5.html)
- [Megatron Core Multi-Storage Client](https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/msc_integration.html)
- [MSC cloud provider config](https://nvidia.github.io/multi-storage-client/references/configuration.html)
- [HF Storage Buckets access](https://huggingface.co/docs/hub/storage-buckets-access)
- [HF Storage Buckets S3 gateway](https://huggingface.co/docs/hub/storage-buckets-s3)

## 1. Connect the existing Mocha bucket

Hugging Face filesystem mount is the simplest path for **donor weights**:

```bash
hf-mount start bucket kofecheckmate/mocha /mnt/mocha
```

This maps `hf://buckets/kofecheckmate/mocha/` to an ordinary directory. The
individual donor roots must each contain the **actual** HF `config.json`
and all weight/index shards. The bucket root itself is not necessarily a valid
Hugging Face model repository. Do not assume particular donor subdirectory
names or prepopulate missing checkpoints.

The S3-compatible gateway alternative needs *separate HF-generated S3
credentials* (an existing `hf auth login` alone does not create them).
To generate an MSC configuration with the real Mocha profile:

```bash
python -m scripts.iq_megatron_bridge storage-config \
  --output /scratch/iq-msc.yaml \
  --cache-dir /scratch/msc-cache \
  --cache-size 500G
export MSC_CONFIG=/scratch/iq-msc.yaml
```

The generated `mocha` profile has:
- `type: s3`, `base_path: mocha`,
  `endpoint_url: https://s3.hf.co/kofecheckmate`;
- `region_name: us-east-1`, **path-style addressing**;
- request/response checksums `when_required`, per Hugging Face gateway docs;
- credentials referenced as `${HF_S3_ACCESS_KEY}` and
  `${HF_S3_SECRET_KEY}` environment variables only.

Export the **separately generated HF S3 keys** using those environment names
on the GPU worker. Never commit or send the values to the repository. MSC can
then address `msc://mocha/<object-key>` after your worker has been configured
and authenticated. For import from HF into Megatron Bridge, **use the mounted
local HF model directory**, not `msc://` as the `--hf-model` parameter:
the current conversion CLI does not document direct MSC input URLs.

## 2. Add other cloud buckets without changing the IQ code

MSC uses a YAML profile config to reference multiple cloud stores. IQ can
combine your Mocha profile with any additional provider in a local JSON spec
and write the resulting MSC-compatible YAML. Each `profiles[]` entry
has a `name`, `provider`, `bucket` (MSC `base_path`) and the required
provider-specific properties:

| Provider | `provider` | Required extra |
| --- | --- | --- |
| Amazon S3 or compatible object stores | `s3` | Optional `region_name`, `endpoint_url`; default AWS credential chain |
| Google Cloud Storage | `gcs` | `project_id`; workload identity recommended |
| Azure Blob Storage | `azure` | HTTPS `endpoint_url`; managed identity or SDK defaults |
| Oracle Object Storage | `oci` | `namespace` |
| NVIDIA AIStore | `ais` or `ais_s3` | Backend endpoint under provider options |
| SwiftStack | `s8k` | S3-compatible HTTPS endpoint |
| GCS via S3 API | `gcs_s3` | S3-compatible endpoint and credentials |

Use `extra_options` only for **documented provider constructor options**.
The CLI rejects literal access/secret keys as arguments: for S3-compatible
profiles use `access_key_env` and `secret_key_env`, which become MSC
environment expansions, or rely on the worker's normal role credentials.

```bash
python -m scripts.iq_megatron_bridge storage-config \
  --additional-profiles-json /scratch/iq-storage-profiles.json \
  --output /scratch/iq-multi-msc.yaml \
  --cache-dir /scratch/msc-cache
export MSC_CONFIG=/scratch/iq-multi-msc.yaml
```

No arbitrary cloud bucket is presumed to exist. Provide your own profile
configuration with actual bucket IDs, endpoint URLs, and permissions.

## 3. Run NVIDIA's genuine donor import (not IQ fusion)

Run from a worker with a pinned Megatron-Bridge checkout, enough CPU memory
and/or appropriately distributed NVIDIA GPUs, and a real donor model on
`/mnt/mocha`. Each model is imported **separately** through its supported
official Bridge.

**NVIDIA Nemotron Super:** Hub ID
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16`, pinned revision
`d51eab0d1f979ebc26b546e634a04f450d99158e`.

```bash
python -m scripts.iq_megatron_bridge import \
  --donor nemotron-super \
  --source nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16 \
  --output /scratch/megatron/nemotron-super \
  --bridge-root /opt/Megatron-Bridge \
  --device gpu --executor local --gpus-per-node 8 \
  --tp 1 --pp 1 --ep 8 --etp 1 \
  --manifest /scratch/iq-nemotron-bridge-plan.json
```

**GLM-5.3:** Hub ID `zai-org/GLM-5.3`, pinned revision
`aca966e4e02791568aa6a4ced368624b3d897f42`.
NVIDIA's shared `GLM5Bridge` handles GLM5 MLA/DSA/MoE mappings and
**FP8 blockwise dequantization to BF16**. Full-model GLM-5.3 verification
is pending; the published verified GLM-5 base model is not GLM-5.3.

```bash
python -m scripts.iq_megatron_bridge import \
  --donor glm53 \
  --source zai-org/GLM-5.3 \
  --output /scratch/megatron/glm53 \
  --bridge-root /opt/Megatron-Bridge \
  --device gpu --executor local --gpus-per-node 8 \
  --tp 1 --pp 1 --ep 8 --etp 1 \
  --manifest /scratch/iq-glm53-bridge-plan.json
```

The commands above **only generate** source/provenance and a concrete NVIDIA
command plan; nothing large is downloaded or converted until you append
`--execute`. To read the donor provider through the real NVIDIA Python
module, first install Bridge on the worker, then append `--probe-provider`.
For weights already staged in Mocha, set `--source` to the **absolute mounted
HF model directory**, with `config.json` inside it. Do not invent model
names beneath the mount.

Import execution creates a *donor Megatron checkpoint*, not an IQ checkpoint.
On a proper GPU cluster, scale `--nodes`/`--tp`/`--pp`/`--ep`/`--etp`
to the published donor's verified topology and actual available resources;
the CLI performs no CUDA capability assumption and does not claim 8 GPUs are
sufficient for GLM-5.3. It refuses to overwrite an existing target.

## 4. Multi-bucket checkpoint save/load with Megatron Core

Megatron Core's **Multi-Storage Client** works with the distributed
`torch_dist` checkpoint format, opt-in by `--enable-msc`.
For a Megatron Core training job that supports those flags:

```bash
export MSC_CONFIG=/scratch/iq-multi-msc.yaml
python pretrain_gpt.py --enable-msc \
  --save msc://mocha/iq-checkpoints/run-01 \
  --load msc://mocha/iq-checkpoints/run-01 \
  --save-interval 1000
```

This is NVIDIA's Megatron Core checkpoint interface, **not** a new
`megatron.bridge` conversion flag. Do not pass `--enable-msc` to
`convert.sh` unless the installed version documents that option. Bridge
donor import/export should use a fast local checkpoint directory, then use
your chosen MSC/CLI transport to archive completed distributed checkpoint
files to the destination bucket after validating them.

Use a local NVMe `cache.location` sized for the working set, not shared
temporary memory. HF Storage Buckets are mutable/non-versioned: keep
per-run directories, pinned source commits, index/checksums and immutable
provenance manifests so an accidental overwrite does not silently change
the training source.

## 5. IQ-specific custom bridge remains a separate task

NVIDIA's GLM5Bridge maps 78 donor blocks to **156 physical HybridModel
layers** (attention/mixing and MoE paired). IQ maps the same 78 donor stages
to **78 composite physical blocks**, each holding **native rank-4 Mamba-3
MIMO + CSA/HCA attention in parallel, plus MoE**. The layer depths matching
numerically does not mean the model parameter keys, residual coordinates or
execution semantics are identical.

NVIDIA Nemotron Super donor is 88-layer interleaved Mamba-2/attention/MoE,
not 78 parallel native Mamba-3 blocks. Consequently:

1. Import and validate the exact NVIDIA/GLM **donor** tensors using Bridge.
2. Map each donor-derived attention/expert projection with IQ's own
   WARM-style coordinate transport and manifest.
3. Load the separate Mamba-3 MIMO `24 → 78` native overlay without importing
   Nemotron's Mamba-2 tensors as Mamba-3.
4. Align GLM/Nemotron representations to the NVIDIA-owned vocabulary/residual
   frame; check Mamba-source drift and expert routing.
5. Run CUDA prefill/step parity and held-out capability validation; only then
   publish an IQ-compatible checkpoint.

Nothing in this PR claims steps 2-5 have executed on real pretrained weights.
