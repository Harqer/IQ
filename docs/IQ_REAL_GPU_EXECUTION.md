# IQ: real GPU execution, accounts, checkpoint and rollout gates

**Source-of-truth policy:** NVIDIA's published Megatron Bridge/Megatron Core
documentation overrides IQ code. Anything marked IQ-specific is not claimed
to be an upstream NVIDIA feature. No generated expert shards, manifests or
CPU unit tests establish a completed 78-block pretrained model.

## Accounts and billing

1. **GitHub:** repository access is already sufficient to check out IQ.
2. **NVIDIA Megatron Core/Bridge:** public-source Python packages require no
   NVIDIA developer account or license key by themselves. Public NGC
   containers may be pulled as a guest; protected/entitled NGC images require
   NGC login and an authorized API key. Neither provides paid GPUs automatically.
3. **GPU provider:** create a cloud-compute account with an authorized payment
   method or credits and request the actual GPU topology, node RAM, NVLink/
   NVSwitch and local/shared storage needed. Vendor pricing and GPU
   availability must be confirmed in the account before starting workloads.
   Example: https://www.runpod.io/pricing (not a promise of cluster capacity).
   Never enter credit-card numbers, provider secrets or API keys in GitHub.
4. **HF Mocha:** the owner already selected
   `hf://buckets/kofecheckmate/mocha`. Native `hf-mount` uses HF auth.
   The S3-compatible interface additionally requires a *separate scoped HF S3
   credential pair* derived from a Hub access token. Separate read-only donor
   permission from write permission for output/checkpoint prefixes. Bucket
   objects are mutable and not versioned; never overwrite donors.
   Quotas, storage fees and access permissions depend on the HF account.

## Actual capacity

NVIDIA's GLM-5/GLM-5.2 verified reference uses **16 GB200 GPUs** to import
a roughly 1.4 TB checkpoint with 4 nodes and **192 GB200 GPUs** for published
training/resume validation. Those validation cards are for the named GLM
variants, *not* proof that GLM-5.3 and IQ train on the same machines.
GLM-5.3's shared `GLM5Bridge` mappings exist but its complete conversion/
behavior qualification remains pending as of the October 2026 guide.

IQ's 75×256×(three 2048×2048 matrices) BF16 routed experts alone need
roughly **450 GiB total**, excluding native Mamba-3, DSA, embeddings, experts'
optimizer moments, pipeline activations, all other parameters, FP32 master
weights and communications. EP=8 means ~56.25 GiB of routed *weights* per EP
rank, **not** 56.25 GiB of complete training memory. Calculate total
optimizer+grad+activations before reserving hardware. IQ presently has no
verified tensor-/pipeline-parallel full-model implementation; this is an
explicit blocker for production training, irrespective of cloud payment.

## Stage A — Free/low-cost developer acceptance (no full IQ weights)

```bash
git clone https://github.com/Harqer/IQ.git
cd IQ
git checkout feat/iq-mcore-training-checkpoints
python -m pip install -r requirements-model.txt
python -m pip install "megatron-core==0.19.0"
python -m unittest discover -s tests -p "test_iq_mcore_checkpoint.py" -v
```

The **real** 2-process CPU/Gloo test writes Megatron Core `torch_dist`
shards, reloads model/AdamW moments/scheduler/rank-local RNG/dataloader
cursor, then executes another training step and rejects overwriting a
checkpoint iteration. CI has a dedicated `native-checkpoint` job.
A passing test proves only this miniature real EP-world layout, not the
full IQ 78-block GPU model.

## Stage B — Authenticate and inspect real donor data (no mutation)

Official sources:
- https://huggingface.co/docs/hub/storage-buckets-access
- https://huggingface.co/docs/hub/storage-buckets-s3

On a remote worker with HF authentication:

```bash
hf-mount start bucket kofecheckmate/mocha /mnt/mocha
```

Identify the actual model directories from the mount. The GLM BF16 directory
must contain its true `config.json` and indexed safetensors. Do not assume
one exists merely because the bucket exists. Validate exact publisher
revisions and the model's license/use terms.

```bash
python -m scripts.preflight_glm53_parallel \
  --donor-config "$GLM_BF16_DIR/config.json" \
  --output /scratch/iq/glm53-topology.json
```

The pinned Mamba-3 donor is **not** NVIDIA's Mamba-2 operator.
Do not load Mamba-2 recurrent tensors into Mamba-3 MIMO by name.

## Stage C — NVIDIA donor conversion (GPU cluster, not a prototype IQ model)

Use NVIDIA's *actual* `Megatron-Bridge/scripts/conversion/convert.sh import`
for supported Nemotron/GLM donors, with **existing** HF full-model paths or
proper pinned Hub IDs, exact source revision and adequate host/GPU memory.
IQ's CLI can render/review its arguments:

```bash
python -m scripts.iq_megatron_bridge import \
  --donor glm53 --source zai-org/GLM-5.3 \
  --output /scratch/iq/donors/glm53-mcore \
  --bridge-root /opt/Megatron-Bridge \
  --device gpu --executor local --gpus-per-node 8 --ep 8
```

This is **planning only** unless `--execute` is explicitly passed, and the
example layout is not a performance-qualified configuration for GLM-5.3.
Large GLM-5.3 conversion should use a topology checked against NVIDIA's
installed GLM5 recipe and your actual cluster. NVIDIA's published verified
GLM-5 results use much greater total node memory than a laptop.

The output here is a **donor Megatron checkpoint**. It is not an IQ native
78-block checkpoint and cannot be used as `checkpoint.load` on IQ without
IQ-specific operator/shape/coordinate conversions.

## Stage D — IQ weight transport and native EP checkpoint

The existing code provides independent native Mamba-3 24→78 overlay,
GLM expert EP-shard conversion and a metadata inventory. These require
real per-stage activation-refined calibration and deterministic learned
coordinate maps. They **do not yet** jointly instantiate every IQ attention,
Mamba and expert weight from GLM/NVIDIA or implement expert-tensor-to-full-
Megatron-optimizer checkpoint migration.

The new `iq_training.mcore_checkpoint` saves IQ-native
`torch_dist` *training state* from a model already initialized with real
weights; it does not convert/initialize donor weights. Explicit limits:
- EP group must equal the whole distributed world, with the same EP size
  on resume; TP/PP/DP repartition is blocked rather than faked.
- Exact AdamW, real scheduler state, caller-provided deterministic dataloader
  state, optimizer moments where initialized and rank-local RNG are saved.
- A unique `iter_N` directory is written, then an atomic
  `iq_latest.json` tracker after collective completion.
- Resume uses the IQ loader, **not** NVIDIA Bridge's `checkpoint.load`
  entry point. These are not interchangeable training-run directories.
- The owning training application must implement dataloader cursor
  restoration and synchronize replicated router/shared parameters across EP
  ranks during training; future NCCL/PP/TP tests are essential.

NVIDIA training recipe docs require `checkpoint.pretrained_checkpoint`
for **weight initialization** and `checkpoint.load` for a full
**native Megatron Bridge resume**. The latter expects model+optimizer,
scheduler, RNG, iteration and (when relevant) dataloader state and its
tracker. Do not pass an IQ partial expert shard directory there.

## Stage E — Production acceptance before paid full training

Require evidence from the *actual* cluster:
- Pinned source commits, tokenizer/license validation, donor tensor-key/shape
  coverage, reproducible BF16/FP8 dequantization and sharded weight hashes.
- GLM DSA MLA/IndexShare causal forward/backward parity, packed CP and
  recomputation-safe cache behavior; independent Mamba-3 CUDA CuTe steps.
- TP/PP/EP/DP expert/attention/recurrent partition strategy with measured
  peak memory and throughput.
- Full native checkpoint save/load (strict optimizer+RNG+dataloader resume),
  interrupted-job recovery, exact continuation of loss/gradients.
- Student/teacher capability retention and nondeterminism/NaN checks.

Only after every gate passes should the checkpoint be uploaded (in an
immutable run/version prefix) into Mocha. Keep GPU disks and cloud volumes
accounted for; terminating GPU compute does not necessarily delete volumes.

## Authoritative references

- NVIDIA Megatron Core install:
  https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html
- NVIDIA sharded tensor API:
  https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.dist_checkpointing.mapping.html
- NVIDIA distributed serialization:
  https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.dist_checkpointing.serialization.html
- NVIDIA Bridge checkpointing/resume:
  https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html
- NVIDIA GLM-5 and GLM-5.3 qualification differences:
  https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/glm/glm5.html
- NVIDIA MSC bucket support:
  https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/msc_integration.html
- Hugging Face S3 key issuance:
  https://huggingface.co/docs/hub/storage-buckets-s3
