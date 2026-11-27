# Experimental Colab TPU inference: full IQ235 source

The source bucket contains 539,388,468,802 bytes indexed across 20,283 shards.
A single TPU cannot hold that checkpoint. This entrypoint uses the existing
235-layer plain PyTorch reference and streams one layer and its selected
BF16 experts at a time, retaining recurrent state and attention history on host.
All 79 Mamba-3 rank-4, 78 MoE top-8, 21 CSA and 57 HCA layers run on the chosen
device; embedding, depth residual, final norm, reasoning and LM head run on CPU.
No training, quantization or bucket writes. Disk cache is bounded and evictable.

## Colab

Select a TPU runtime. Run in a fresh notebook:

```python
!git clone -b feat/iq235-colab-tpu-streaming https://github.com/Harqer/IQ.git
%cd IQ
%pip install -U torch 'torch_xla[tpu]' 'huggingface_hub==1.14.0' 'transformers==5.17.0' safetensors
```

Keep torch and torch_xla on matching release versions; restart the notebook
runtime if packages were replaced. Set HF_TOKEN from Colab Secrets without
printing it. The source tokenizer is gated: the token needs access to
meta-llama/Llama-3.1-8B as well as the weight bucket.

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
!python scripts/iq235_colab_tpu.py --prompt "The capital of France is" --max-new-tokens 2
```

If only tokenizer access is blocked, a real full-model token-ID diagnostic is
available: `--token-ids 128000 32 --max-new-tokens 2`. It does not certify text quality.
No trained GLM lexical overlay is present in the source checkpoint: its 128256
rows use the Mamba/Llama lexical foundation, not the 154880 GLM rows. Do not
substitute a GLM tokenizer for these embeddings. A GLM-tokenized deployment
requires matching trained embedding/head tensors.

Status: experimental, not TPU parity-tested. A single generation may take a long
time and exhaust free runtime/network quotas; XLA operator support and actual
HBM allocation must be established by real execution. The manifest identity
checks cover config/index, not independent payload provenance.


## Alternative: direct hf-mount access (no duplicate shard cache)

Install hf-mount using Homebrew if available, or install the Linux binaries
from https://github.com/huggingface/hf-mount/releases. Both the daemon and the
NFS backend binary must be available on PATH. NFS is the default; FUSE is
optional and requires a usable /dev/fuse and fuse3. Hosted Colab may deny
kernel mounts; installing the binary does not guarantee mount permission.

With HF_TOKEN already set in the environment:

```bash
mkdir -p /content/iq-weights
hf-mount start --read-only --cache-size 10000000000 bucket kofecheckmate/iq-glm53-all256 /content/iq-weights
hf-mount status
ls /content/iq-weights/IQ/runs/glm53-full78-local-20261025a/iq_config.json
python scripts/iq235_colab_tpu.py --mount-dir /content/iq-weights --max-new-tokens 2
```

The mounted provider reads the original safetensors on demand, without
downloading a separate copy into the application's cache. The hf-mount daemon
still transfers requested bytes over the network and may cache them locally.
The mount is not TPU memory. The runner streams each physical layer and its
routed experts to the TPU as before. If the hosted runtime denies mounting,
omit --mount-dir to use the existing bucket API transport instead.

When finished: `hf-mount stop /content/iq-weights`.

Do not mount the bucket read-write for inference. This path does not modify
checkpoint artifacts. Actual hosted mount and TPU execution remain unverified.
