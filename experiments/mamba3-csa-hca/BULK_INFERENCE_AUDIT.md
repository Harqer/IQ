# Bulk generated-answer audit (Phase 5e)

This is **actual greedy text generation**, not another logits-only proxy.
`scripts/benchmark_iq_mamba3_csa_hca.py` calls the **same**
`Mamba3CSAHCARunner.generate()` method used by the experimental HTTP
backend, decodes the response text, and compares **baseline vs hybrid** on
exactly the same input prompts. It reports literal responses, first-answer
word accuracy, paired source-fact switching, hallucination/misses, regressions,
runtime errors and generation time. Nothing in this benchmark trains or edits
the accepted checkpoint or the pretrained foundation.

## Run in one batch on your Linux machine

```bash
set -euo pipefail
cd ~/IQ
git pull --ff-only origin main
source .venv/bin/activate

python -m pytest -q \
  tests/iq_local/test_mamba3_csa_hca_inference.py \
  tests/iq_local/test_mamba3_bulk_audit.py

python -u scripts/benchmark_iq_mamba3_csa_hca.py \
  --attention-checkpoint \
    ~/IQ/checkpoints/mamba3-csa-hca/iq_phase5e_attention_candidate.pt \
  --fresh-pairs 8 \
  --regression-pairs 4 \
  --paraphrase-pairs 4 \
  --general-prompts 6 \
  --max-new-tokens 8 \
  --threads 2 \
  2>&1 | tee ~/iq-phase5e-bulk-terminal.log
```

The default-sized run performs **76 separate full greedy generations**:
8 new paired project facts (both original and swapped, baseline and hybrid);
4 previously inspected Phase 5e paired project facts (regression-only);
4 additional new paired facts with a different question template; and
6 simple/general prompts tested in both modes. Each generation emits up to
8 tokens with this command. The original model is loaded **once**, and only
the optional attention insertion changes between baseline and hybrid.

For substantially longer individual responses, use
`--max-new-tokens 12` or `--max-new-tokens 16`. Full-prefix CPU
recomputation is expensive; the command above emits progress after every
response, with incremental `responses.jsonl` saved along the way.

After interruption, rerun the same command with **`--resume`**. Do not
mix different test counts or new-token limits in a resumed output directory.

## Read results

- `~/iq-phase5e-bulk-audit/responses.jsonl` — complete prompts, decoded
  generated responses, token IDs, timings and errors, saved per generation.
- `~/iq-phase5e-bulk-audit/summary.json` — consolidated scores per suite,
  baseline-versus-hybrid wins/losses, original/swapped evidence switches, and
  failure classifications.
- `~/iq-phase5e-bulk-audit/manifest.json` — settings and run identity.
- `~/iq-phase5e-bulk-terminal.log` — progress and condensed summary.

A correct single-word fact response must start with the expected answer
word. Mentioning the answer later in a longer incorrect reply does not count
as correct. Generic story continuation is recorded but **not falsely graded
as factual accuracy**. Paired success requires both responses to be right,
not merely different after the evidence is swapped.

This benchmark **must not** be used to tune on the already-inspected Phase
5e locked cases or on new held-outs after observing their answers. If it finds
systematic deficits, fix model architecture/data/training on a separate
development set, then evaluate new truly unused held-outs. Do not change
weights automatically or weaken the acceptance gate. The result is research
evidence, not a certification of general retrieval or official CUDA parity.
