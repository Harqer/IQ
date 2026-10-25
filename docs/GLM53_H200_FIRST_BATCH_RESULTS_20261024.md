# First full78 CUDA WARM recovery: verified H200 batch

**Completed:** 2026-10-25 02:44 UTC. **Account:** `kofecheckmate`. **Branch:** `audit/glm53-cuda-stack-preflight-20261024`; draft [PR #65](https://github.com/Harqer/IQ/pull/65).

## Result

The real NVIDIA H200 computed **four previously missing GLM-5.3 full78 source calibration stages** (`1, 2, 4, 5`). Each stage published its calibration artifacts into the existing public bucket with a receipt written last. No old checkpoint file was overwritten. An **independent read-only CPU Job** verified SHA-256 of every newly written artifact against its saved receipt.

| Source layer | Calibrated artifacts | Bytes independently SHA-256 verified |
| ---: | ---: | ---: |
| 1 | 12 | 906,560,911 |
| 2 | 12 | 906,560,911 |
| 4 | 11 | 705,236,261 |
| 5 | 11 | 705,236,261 |
| **Total** | **46** | **3,223,594,344** |

**Independent audit:** [Hugging Face Job 6add6cfa046eb82b1ece1ce8](https://huggingface.co/jobs/kofecheckmate/6add6cfa046eb82b1ece1ce8) — status `COMPLETED`, `FULL78_NEW_GPU_STAGE_AUDIT_PASS`, all 46 hashes matched; original 41 source-layer-3 routed expert shards still exist.

**Successful GPU worker:** [Hugging Face Job 6add6b2b046eb82b1ece1cc7](https://huggingface.co/jobs/kofecheckmate/6add6b2b046eb82b1ece1cc7) — `h200`, status `COMPLETED`, started `2026-10-25T02:36:27.799Z`, finished `2026-10-25T02:43:10.127Z`. Python/CUDA math smoke passed, original and public bucket mounts mounted successfully, pinned GLM donor revision `abf42e6c0e53bc29a875511301e7b7fcd220315b`.

**Artifact root:** `hf://buckets/kofecheckmate/iq-glm53-all256/calibration78/glm53-full78-20261024-c/recovery-stage-uploads/`; each `stage-XX/` contains artifacts plus `receipt.json` with expected target fingerprint `6c292b471e85dfd97d2210087af68b247bc46702e75e5277d08f3ee7c49e788d`.

## Resume inventory

- 28 previously persisted later-stage calibration maps (50–77) remain original candidates, with 9 stages previously SHA-verified against the protected 24-stage source.
- 15 of the 24 protected old calibration source stages cover source layers 0–49 and were previously SHA-verified. They are reused read-only; they were not republished here.
- **4 new stages** 1,2,4,5 are durably persisted and independently SHA-256 verified.
- **31** new source stages still need CUDA WARM computation before constructing the full 78-stage calibration.
- 41 saved routed expert shards from source layer 3 remain available but **producer provenance and numerical donor equivalence are not yet established**.
- No complete 866-file full78 calibration SHA256 receipt or `COMPILE_COMPLETE.json` has been published. No model-quality or inference claim is made.

Next batch should choose `7,8,9,11` through `scripts/select_glm53_missing_gpu_stages.py`, but only after confirming funds and user spending authorization. No other GPU Job should start automatically.

## Budget and errors resolved

The user's spend authorization for this operation was **at most $1.25 in existing prepaid HF credits**, with automatic recharge assumed disabled by user confirmation. Both previous failed H200 attempts ended before the third successful one:

- [Job 6add67f21e112550ce7e6769](https://huggingface.co/jobs/kofecheckmate/6add67f21e112550ce7e6769): Python import path issue, elapsed ~12 sec; fixed via image `PYTHONPATH=/workspace/IQ`.
- [Job 6add682a046eb82b1ece1c43](https://huggingface.co/jobs/kofecheckmate/6add682a046eb82b1ece1c43): independently verified 166 original plus 99 overlapping saved map files; then Hugging Face CLI Typer/Click `Sentinel` failure, elapsed ~5m43s. Fixed by using `huggingface_hub.hf_hub_download()`.
- [Job 6add6b2b046eb82b1ece1cc7](https://huggingface.co/jobs/kofecheckmate/6add6b2b046eb82b1ece1cc7): 4 real CUDA WARM stages completed, elapsed ~6m42s. It avoids repeated 16 GiB hashing on paid GPU and saves each stage immediately.

Estimated H200 billing, rounding each Job to an individual minute: **14 GPU-minutes × $5/hour ≈ $1.1667**. Independent CPU-basic verification ~34 sec, estimated **$0.00017**. Combined hardware estimate **~$1.17**. Exact charges/remaining prepaid balance and auto-recharge state are not exposed through the HF plugin; check the HF Billing page. All Jobs above were finished and no automatic future launches were scheduled.

Keep the per-stage receipts and continue only after independent storage/budget verification. **No additional paid H200 allocations are authorized by the $1.25 cap once this operation is complete.**
