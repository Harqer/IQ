# GLM-5.3 → IQ full-78 partial recovery audit

**Audit date:** 2026-10-24. **Status:** original artifacts preserved; staged recovery logic implemented, but no full checkpoint or verified calibration receipt has been published. No full-transfer GPU Job started by this audit.

## Independently observed assets

Sources: read-only Hugging Face mounts of `kofecheckmate/iq-glm53-all256` and original `kofecheckmate/mocha/legacy/visionboxx-macha/calibration`.

| Asset | Observed | Status |
| --- | ---: | --- |
| Full78 calibration metadata | Declares source layers 0–77 | Metadata only; incomplete content |
| Complete physical calibration stages | 28 (50–77) | Existing candidates with missing original run-level receipt |
| Initial calibration stages absent | 50 (0–49) | Requires selective recovery |
| Originally calibrated source layers covering 0–49 | 15 | **166 files, 10.04 GiB**, SHA-verified from original protected calibration |
| Calibration source layers still requiring WARM | 35 | **387 files**, only these need fresh computation |
| Files already saved under full78 calibration prefix | 313 | 308 late-stage files + 4 global maps + 1 manifest |
| Expected final calibration file inventory | 866 | 313 + 166 + 387 |
| Existing checkpoint files | 50 | 4 contexts + 4 common + 1 shared + 41 routed-expert shards |
| Saved routed experts | 41 / 19,200 (layer 3 slots 0–40) | BF16 headers/shapes and persisted SHA256 checked; donor-value equivalence **not** proven |
| Full78 completion/provenance/calibration SHA receipt | Missing | **Full checkpoint resume must stay blocked** |

The original 24-stage calibration uses pinned GLM donor revision `abf42e6c0e53bc29a875511301e7b7fcd220315b`. The full-78 target configuration fingerprint declared in the existing partial run is `6c292b471e85dfd97d2210087af68b247bc46702e75e5277d08f3ee7c49e788d`.

The original 24-stage calibration source-layer map is:
`0,3,6,10,14,17,22,23,26,30,34,37,42,44,46,50,54,57,62,64,66,70,74,77`.

15 of these old layers fall in the missing 0–49 interval; the remaining nine overlap with saved later-stage artifacts. The **lexical-input**, **final-output** and final **stage-77 residual** map hashes match the original SHA-protected calibration byte-for-byte, including relevant JSON companions.

The saved expert SHA256 inventory lives in [`recovery/glm53-full78-20261024-c-candidate-sha256.json`](../recovery/glm53-full78-20261024-c-candidate-sha256.json). These hashes establish current persisted bytes, not original producer provenance or functional fidelity. They must never be used alone to adopt a partial model.

### Completed read-only checks

- [Source and output inventory CPU audit](https://huggingface.co/jobs/kofecheckmate/6add37cc1e112550ce7e4b59) — verified all 28 complete late-stage groups, layer-3 expert slots 0–40, no malformed expert headers.
- [Original WARM reuse SHA-256 CPU audit](https://huggingface.co/jobs/kofecheckmate/6add39ef1e112550ce7e4cf8) — checked **all 166 original candidate map files** against the original calibration's recorded SHA256; assembled a **temporary zero-copy read-only overlay** referencing 312 saved candidate files. This ephemeral overlay is not a persistent new bucket and was not published.
- [Existing expert SHA256 and original lexical/final boundary audit](https://huggingface.co/jobs/kofecheckmate/6add3a9b046eb82b1ece0850) — hashed all 41 candidate experts and confirmed original calibration/boundary hashes.
- Independent late-stage overlap verification is separately scripted; its successful result, if obtained, must be recorded before treating additional saved stages as original-equivalent.

## Recovery implementation

- `scripts/reconcile_glm53_partial.py`: offline inventory; validates canonical original manifest hash and donor revision; copies only SHA-verified missing old24 map files, or assembles a **temporary zero-copy local overlay** using original read-only files and existing stage-50..77 candidates. Refuses overwrite, unknown names, partial stages and wrong fingerprints.
- `iq_transfer/glm53_calibration.py`: now has a streaming sink to skip complete stages and emit missing stages without holding all 78 giant matrices.
- `iq_transfer/glm53_partial_repair.py`: stage-wise calibrated asset writer, mapping/fingerprint guards, atomic local files and no rewrite of completed stage groups.
- `iq_transfer/cli.py`: `glm53-bootstrap-calibration --repair-overlay` generates **only absent source layers** (expected 35).
- `scripts/repair_glm53_full78_calibration.sh`: recovery uses the above, then materializes symlinks only in scratch when sufficient verified storage exists, signs the final 866-file receipt and syncs without overwrite. `RUN_ID` remains the original run ID; previously persisted buckets stay intact until explicit sync.
- CI covers the expected 15/35 split, 166 verified old-file names, 312 later/global files, no-overwrite behavior, invalid donor receipt, incorrect target geometry and fail-closed partial stage writes.

## Cost and storage gate

As observed during this audit, `mocha` contains ~78.0 **GB**, `iq-glm53-all256` ~22.0 **GB** and other private buckets contain additional data. This is around 100 GB already saved overall. It is not evidence that a further 600 GiB of private free capacity is available. **Never allocate the long GPU transfer or publish large duplicate snapshots based only on manually asserted capacity.**

The repair script's preflight requires verified large-output storage. Source reuse avoids recomputing the 15 original calibrated stages and 28 saved late stages, and a zero-copy workspace avoids unnecessary extra copies before final publication. New WARM computation is needed for **35 stages**, not 78.

## Outstanding acceptance gates

1. Verify the nine later-stage calibration overlaps against the protected original where applicable; validate the 19 remaining saved late-stage source mappings before promotion.
2. Compute only the 35 absent stages, materialize a complete byte-for-byte consistent 866-file durable snapshot, then verify all SHA256 hashes from a separate read-only worker.
3. Verify **all 41** saved expert outputs against the pinned donor weights and canonical stage-3 transform in one in-run batched equivalence pass. Do not individually benchmark each expert. Only allow adopting them when exact provenance and numerical fidelity requirements pass.
4. Obtain an independently verified private storage-capacity attestation and enforce single-writer concurrency before full GPU compilation.
5. After full compilation, independently check 78-layer/19,200-expert coverage and meaningful inference; coverage is not GLM quality preservation.

**Never fabricate `run-provenance.json` or `COMPILE_COMPLETE.json` for the currently incomplete run.**
