# IQ: GLM-5.3 lexical foundation on native Mamba-3 MIMO

The student retains **all 24 pretrained rank-4 Mamba-3 MIMO mixers and gated MLPs**. It uses GLM-5.3's original 154,880-token tokenizer, 154880×6144 input embeddings and 154880×6144 untied output head, with separately trainable 6144→2048 and 2048→6144 adapters. The GLM embeddings and head are frozen. GLM attention and MoE are *not* directly transferred. Teacher distillation is needed to acquire their capabilities.

Sources: https://huggingface.co/zai-org/GLM-5.3-BF16 and https://huggingface.co/state-spaces/mamba3-mimo-1.5b

## Stage only GLM's lexical tensors

The official pinned GLM revision used by IQ is abf42e6c0e53bc29a875511301e7b7fcd220315b.

~~~bash
GLM_DIR=./artifacts/glm53-lexical
GLM_REV=abf42e6c0e53bc29a875511301e7b7fcd220315b
mkdir -p "$GLM_DIR"
hf download zai-org/GLM-5.3-BF16 \
  config.json tokenizer.json tokenizer_config.json model.safetensors.index.json \
  --revision "$GLM_REV" --local-dir "$GLM_DIR"

python - "$GLM_DIR" <<'PY'
import json, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1])
index=json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]
names=sorted({index[k] for k in ("model.embed_tokens.weight","lm_head.weight")})
assert names and all(Path(name).name == name and name.endswith(".safetensors") for name in names)
subprocess.run(["hf","download","zai-org/GLM-5.3-BF16",*names,
                "--revision","abf42e6c0e53bc29a875511301e7b7fcd220315b",
                "--local-dir",str(root)],check=True)
PY
~~~

The native upstream Mamba-3 CUDA package and kernel tail guard are covered in [the original native runner instructions](MAMBA3_PRETRAINED_RUNTIME.md). No Llama tokenizer is loaded by this GLM student.

## Preflight and generate

~~~bash
python -m scripts.iq_glm53_mamba3 --glm-root "$GLM_DIR" --preflight
python -m scripts.iq_glm53_mamba3 \
  --glm-root "$GLM_DIR" \
  --glm-tokenizer-sha256 "$PINNED_GLM_TOKENIZER_MAP_SHA256" \
  --prompt "def square(x): return x * x" --max-new-tokens 16
~~~

The tokenizer mapping SHA-256 is optional for initial exploration, but when provided it must be independently established from the pinned GLM source. A hash computed from arbitrary unverified local files does not independently authenticate them. Teacher-logit training requires a pinned fingerprint.

## GLM knowledge transfer

The code at iq_training/glm_mamba3_distillation.py performs one supervised optimizer step comparing student versus GLM teacher last-token logits on the **same GLM input IDs**, with weighted KL plus next-token cross-entropy. Every target is bound to input ID fingerprint, tokenizer mapping fingerprint and teacher revision. The newly initialized adapters train by default; setting train_mamba_blocks=True permits the native MIMO stack to be tuned. GLM lexical weights remain frozen.

**Current result:** architecture, I/O, loader and teacher-loss code, plus CPU contract tests. No claim of coherent generation or inherited GLM reasoning/coding capability is made before the student has actually been trained and evaluated.
