# P0–P3 audit and remediation

Scope: GLM-5.3-first IQ with complete rank-4 Mamba-3 MIMO blocks. No model weights, checkpoints, remote buckets, Baseten endpoints or GPUs were modified.

## Verified source contracts

- GLM-5.3 BF16 revision abf42e6c0e53bc29a875511301e7b7fcd220315b. Vocab 154880; hidden 6144; 78 Transformer layers; first 3 dense and 75 MoE; 256 routed experts, one shared, top-8; SiLU, QK head width 256.
- Mamba-3 MIMO 1.5B revision bc6b5d0f7994fe4cb3478242e92da8daf9ee29ec. Vocab 128256; hidden 2048; 24 Mamba mixer + gated-MLP blocks; recurrent state 128; MIMO rank 4; MLP intermediate 3824. Llama 3.1 is the standalone Mamba model's tokenizer, NOT Llama neural weights.
- Existing all256 manifest reports Mamba pretrained lexical/recurrent foundation plus GLM context/MoE transforms. Its compile marker says inference_ready=false and end_to_end_knowledge_retention_proven=false, with 13 sampled shard headers and no packaged tokenizer.
- HF bucket contains one incomplete temporary .model-mamba-018.safetensors.mgbn0e4t.partial (about 214 MB). This does not establish that any referenced final shard is missing; shard inventory and hashes remain unchecked.
- IQ hybrid runtime's Mamba3ResidualLayer has Mamba recurrent mixer only, without the standalone donor's additional gated MLP and norm2. The GLM-derived MoE in adjacent layers is not automatically the same pretrained Mamba MLP.

## Audit -> remediation -> verification

1. GitHub run 43961188625: 3 failures in 214 unit tests. Two modules assumed that MappingTensorSource was iterable even though the protocol only has keys(), shape() and get(). Repaired iq_transfer/executor.py and iq_transfer/glm53.py to use keys(); rerun the full unit suite.
2. Added scripts/audit_glm_mamba_references.py to validate both pinned donor config contracts, GLM EOS IDs, all 24 MIMO block metadata and legacy checkpoint provenance. Tests include wrong vocab, EOS, SiSO and absent gated-MLP configurations.
3. Added scripts/audit_iq_hybrid_mamba_block.py to prevent certifying a mixer-only class as a complete Mamba MIMO block. Regression test confirms the current runtime is BLOCKED and a synthetic complete structure is only structurally complete, not functionally verified.
4. Earlier draft changes guard the Mamba-founded compiler from implicit invocation and validate intended lexical owner, weight mapping, graph connectivity and inference status.

## CPU-only tests

python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/audit_iq_hybrid_mamba_block.py --source iq_model/hybrid.py
python scripts/audit_glm_mamba_references.py --glm-config /local/glm/config.json --mamba-config /local/mamba/config.json

## Phase exit status

- P0: source identity and config provenance audited; full cryptographic checkpoint inventory still blocked.
- P1: GLM-first topology and four invariant categories documented; no silent dimensional loss authorized.
- P2: original GLM tokenizer/teacher config verified; real teacher forward/logit/cache parity NOT executed.
- P3: full Mamba block layout verified from official docs; existing IQ mixer-only discrepancy flagged. Donor tensor/kernel/MLP numerical parity NOT executed.

Next work: establish an actual GLM reference runtime, implement the complete Mamba block and adapters, validate zero-gated teacher parity. Paid resources require separate approval. Keep all256 read-only.
