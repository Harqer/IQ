# IQ / GLM-5.3 full78: CUDA batches on a **strict $0 compute budget**

Current HF output bucket: `kofecheckmate/iq-glm53-all256` is **public** (verified through the authenticated HF connector). Original private `mocha` calibration is preserved read-only.

## Immutable scope and resumability

- Donor: all **78** GLM-5.3 BF16 source layers; all **256 routed experts in each of the 75 sparse layers** (19,200 genuine routed experts), and three dense stages.
- IQ target: 78 context/MoE pairs plus its Mamba-3 foundation. The initial 32 pretrained Mamba layers do not magically train or initialize the additional 47 slots.
- **No isolated one-expert GPU jobs**. Run full-transfer stage queues, group routed experts into CUDA batches, prefetch the next bounded batch, and checkpoint each completed expert atomically. CUDA OOM halves only the current already-downloaded batch. Single writer per checkpoint namespace.
- Recover exactly **35 missing** WARM source stages; reuse 15 original SHA256-verified maps, 28 saved late-stage groups, and retain 41 existing expert shards as unverified provenance candidates. Verify the 866-file receipt independently before compiling.
- Completion = durable 78-stage + 19,200-expert inventory **and** numerical fidelity + tokenizer/evaluation; presence of safetensors alone is not a coherent model.

## Hardware profiles, automatically selected by actual VRAM

| Available worker VRAM | CUDA experts per batch requested | Max BF16 donor source payload per batch |
| --- | ---: | ---: |
| H200 ~141 GiB | 24 | 2048 MiB |
| RTX Pro 6000 Blackwell ~96 GiB | 16 | 1536 MiB |
| L40S ~48 GiB | 8 | 1024 MiB |
| L4/T4/RTX 2060 etc <40 GiB | 4 | 512 MiB |

`iq_transfer/glm53_gpu_batch.py` automatically selects based on the detected worker's GPU total VRAM. The input shape and payload budget may reduce the actual batch count. Override with `IQ_MOE_GPU_BATCH_EXPERTS=1..32` and `IQ_MOE_GPU_BATCH_SOURCE_MIB=1..2048` only if memory profiling justifies it. A worker using multiple GPUs still does **not** distribute experts across GPUs automatically—multi-GPU *device sharding* must be a separately reviewed implementation.

These capacity profiles do not guarantee throughput, zero GPU OOM or correct GLM reasoning. Inline full-run throughput/peak memory/fidelity checks remain mandatory, not repeated one-expert benchmarks.

## Where to get GPU time for $0

- **Hugging Face Jobs H200:** **$5/hour/GPU**, per minute billed, **not free**. No automatic H200 Job launches under a $0 limit. Source: https://huggingface.co/docs/hub/jobs-pricing
- **Hugging Face ZeroGPU:** Free eligible personal accounts can host up to two ZeroGPU Gradio Spaces. Currently shared **RTX Pro 6000 Blackwell**, not H200, with 48 GB standard or 96 GB xlarge (2× quota cost) and only **5 minutes daily per Free account**. This is not a suitable uninterrupted full78 job; it would require strictly short, resumable Gradio calls and persistent output writes. Source: https://huggingface.co/docs/hub/spaces-zerogpu
- **Lightning AI Free credits:** Advertises **5 free credits on signup without a card**, additional 25 credits only with a card, and H200 availability at advertised ~$4.50/hour (actual credit pricing/availability must be verified on the account). Five credits could provide approximately one hour H200, **not a guarantee of completing the full transfer**. Avoid adding a payment card or enabling pay-as-you-go under the $0 budget. Source: https://lightning.ai/pricing
- **Google Colab Free** and Kaggle free GPU notebook sessions can offer useful, intermittent NVIDIA GPUs, but availability, duration and VRAM are not guaranteed. Local RTX 2060 SUPER is a slower, unrestricted fallback. Colab official limitations: https://research.google.com/colaboratory/faq.html
- **HF community GPU grants / public research sponsorship:** Apply through the Settings tab of a public Space; acceptance is discretionary. Source: https://huggingface.co/docs/hub/spaces-overview

## Public output: explicit limited-assurance policy

`scripts/check_glm53_storage_policy.py` reads `huggingface_hub.bucket_info()` to verify *actual* bucket visibility. For private buckets, the existing provider-verified 600+ GiB attestation remains mandatory. For public buckets, set `IQ_ACKNOWLEDGE_PUBLIC_BEST_EFFORT=1` only after accepting HF's explicit public storage fair-use uncertainty; **there is no guaranteed 600 GB quota**. The storage script checks destinations only; it is **not authorization to spend money on GPU compute**.

Hugging Face notes that significant free public storage is best-effort and asks users to share useful model artifacts, not use public hosting as unrestricted temporary backing store: https://huggingface.co/docs/hub/storage-limits

**Do not** expose private tokens, the original private `mocha` bucket, licensing-restricted files, or user datasets when producing public IQ weights. Check third-party model licensing and reproducibility metadata before model release.

## Next executable gates

1. Choose/receive a GPU session with verified **$0 out-of-pocket** billing, ideally a donated/credit-funded H200/L40S/Blackwell. Do **not** call `hf jobs run --flavor h200` unless externally sponsored credits fully cover it.
2. Recover the 35 missing calibration stages inside that GPU's real, bounded execution; publish only nonduplicate files to the public bucket under the approved fair-use policy.
3. Verify all 866 calibration hashes and remaining late-stage provenance independently, with no new paid GPU Job.
4. Start one full checkpoint compile with 78 source stages and CUDA auto-batching; resume missing shards by atomic file presence **plus key/shape/dtype/provenance validation**, never restart entire layers or launch isolated single-expert benchmarks.
5. Verify model functionality separately before announcing successful knowledge transfer.

**Status:** GPU adaptive batching and public-bucket policy are implemented in PR #65; no paid GPU runtime or public weight writes have been started by this change.
