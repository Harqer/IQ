# GLM-5.3 tokenizer-only IQ variant

Use the **official GLM-5.3 tokenizer** with the existing IQ v2 hybrid. This does not replace `IQHybridForCausalLM`, its Mamba-3 MIMO mixers, Stable LatentMoE, CSA/HCA, Block AttnRes, or reasoning controllers.

Only the lexical vocabulary boundary changes:

- Input IDs: GLM-5.3's tokenizer (source: `zai-org/GLM-5.3-BF16`, revision `abf42e6c0e53bc29a875511301e7b7fcd220315b`).
- IQ input embedding: `[154880, 4096]` at the production IQ residual width.
- IQ output head: `[154880, 4096]`.
- GLM EOS IDs: `154820, 154827, 154829`; pad ID: `154820`.
- All physical IQ layers and Mamba recurrence remain unchanged.
- **No GLM neural-weight download** is needed. Only tokenizer files and metadata are fetched.

The existing IQ code already sizes its input embedding and LM head from `IQHybridConfig.model.vocab_size`. The new `with_glm_tokenizer_config(...)` helper retains all other IQ configuration values and scheduling.

## Run without an IQ config path

From the IQ repository root:

~~~bash
python -m scripts.check_iq_glm_tokenizer --prompt "def square(x): return x*x"
~~~

The command loads only the pinned GLM-5.3 tokenizer artifacts and model **config.json** metadata (not neural weights) and prints the actual token IDs, mapping fingerprint and official special IDs. It does not initialize or modify any model. No checkpoint, hybrid config path or inference engine is required.

To prohibit all network requests and use the previously cached tokenizer:

~~~bash
python -m scripts.check_iq_glm_tokenizer --offline --prompt "Hello, world"
~~~

Offline mode fails with a clear error if the pinned tokenizer is not present in the Hugging Face cache. If the files already exist in a local directory, `--tokenizer-dir` loads that directory directly without accessing the Hub.

The optional `--iq-config` parameter is used **only** when validating an existing IQ hybrid JSON. For an actual configuration that already exists on disk, pass `--iq-config` and `--write-config` to write a second copy with the GLM vocabulary. The tool refuses `--write-config` without a verified source config path; it does not invent a backbone schedule.

An independently recorded tokenizer mapping fingerprint can be supplied using `--expected-mapping-sha256`. Without it, the tool checks full token ID geometry and outputs the observed hash, but does not claim independent fingerprint authentication.

## Exact boundary of the change

This is a tokenizer-only architecture configuration, **not** a claim that Mamba or GLM language ability is preserved automatically.

Token ID 1234 in one tokenizer usually does **not** represent the same token as ID 1234 in another. Therefore:

- An IQ checkpoint trained against Llama/Mamba token IDs **cannot** be reused unchanged as a GLM-tokenized model simply by updating its config.
- A model using GLM token IDs needs its input embedding/output head aligned through training or a separately validated weight remapping. Those are IQ neural parameters, not GLM checkpoint weights.
- GLM tokenization alone does **not** import GLM's Transformer, MoE, coding or reasoning capabilities.
- The repository's default existing Mamba-vocabulary weight-transfer path remains a separate variant; do not combine the two silently.

**No inference engine or GPU-backed model run is needed for these contract checks.** This validates IDs, configuration, and geometry. It cannot validate semantic language quality or pretrained checkpoint compatibility by itself.
