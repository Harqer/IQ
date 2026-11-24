"""Phase 2: actual pretrained Mamba-3 MIMO CPU text-generation probe.

This is deliberately a baseline/identity/wiring test; it does NOT train CSA/HCA.
The CPU MIMO recurrence is not yet numerically certified against CUDA TileLang.
The tokenizer is explicitly sourced from a Llama-3.1 repository; no remapping.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from iq_hybrid_cpu import load_cpu_checkpoint, default_insertions


def run(checkpoint: Path, tokenizer_repo: str, prompt: str, steps: int):
    torch.set_num_threads(2)
    tokenizer_file = hf_hub_download(repo_id=tokenizer_repo, filename="tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_file)
    vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    bos = tokenizer.token_to_id('<|begin_of_text|>')
    if vocab != 128256 or bos != 128000:
        raise RuntimeError(f"Tokenizer identity check failed: vocab={vocab}, BOS={bos}; required vocab=128256, BOS=128000")
    print(json.dumps({'event': 'tokenizer_ready', 'repo': tokenizer_repo, 'vocab_size': vocab,
                      'bos_id': bos, 'note': 'mirror token-ID mapping not proven byte-identical to gated original; use original repo when authorized'}), flush=True)
    ids = tokenizer.encode(prompt, add_special_tokens=True).ids
    if not ids or max(ids) >= 128256:
        raise RuntimeError(f"Tokenizer returned invalid prompt IDs: {ids}")
    print(json.dumps({'event':'prompt', 'prompt':prompt, 'token_ids':ids}), flush=True)
    t0 = time.monotonic()
    model, cfg = load_cpu_checkpoint(checkpoint)
    attn = default_insertions(cfg).eval()
    print(json.dumps({'event':'weights_loaded', 'model': 'state-spaces/mamba3-mimo-1.5b',
                      'pretrained_block_count': len(model.backbone.layers),
                      'pretrained_parameters_frozen': all(not p.requires_grad for p in model.parameters()),
                      'seconds': round(time.monotonic()-t0,2)}), flush=True)
    tokens = torch.tensor([ids], dtype=torch.long)
    assert tokens.shape[1] <= 2048
    with torch.inference_mode():
        a = model(tokens)
        b = model(tokens, insertions=attn)
        if not torch.equal(a, b):
            raise AssertionError('First-pass identity logits differ')
        # Nonzero change is a wiring demonstration only (random, NOT trained).
        for layer in attn.values():
            torch.nn.init.normal_(layer.out_proj.weight, std=1e-6)
        active = model(tokens, insertions=attn)
        max_diff = (active-a).abs().max().item()
        active_top_id = int(active[0,-1].argmax())
        for layer in attn.values():
            layer.out_proj.weight.zero_()
    print(json.dumps({'event':'active_untrained_wiring', 'max_abs_logit_delta':max_diff,
                      'active_next_id':active_top_id, 'note':'NOT evidence that CSA/HCA improves generation'}), flush=True)
    gen_ids = []
    for step in range(steps):
        if tokens.shape[1] >= 2048:
            break
        start = time.monotonic()
        with torch.inference_mode():
            logits = model(tokens)
            hybrid_identity_logits = model(tokens, insertions=attn)
            if not torch.equal(logits, hybrid_identity_logits):
                raise AssertionError('Identity drift at step {}'.format(step+1))
            next_id = int(logits[0,-1].argmax())
        gen_ids.append(next_id)
        tokens = torch.cat([tokens, torch.tensor([[next_id]],dtype=torch.long)],dim=1)
        print(json.dumps({'event':'generated_step','step':step+1,'token_id':next_id,
                          'token_text':tokenizer.decode([next_id],skip_special_tokens=False),
                          'new_text':tokenizer.decode(gen_ids,skip_special_tokens=True),
                          'elapsed_seconds':round(time.monotonic()-start,2),
                          'identity_logits_exact':True}), flush=True)
    print(json.dumps({'event':'completed','result':'PASS','same_pretrained_weights':True,
                      'identity_greedy_trajectory_exact':True,
                      'original_prompt':prompt,
                      'generated_token_ids':gen_ids,
                      'generated_text':tokenizer.decode(gen_ids,skip_special_tokens=True),
                      'whole_text':tokenizer.decode(tokens[0].tolist(),skip_special_tokens=True),
                      'seconds_total':round(time.monotonic()-t0,2),
                      'warning':'Exact identity only compares two paths in THIS CPU implementation. Accuracy vs official CUDA MIMO kernel is not established; attention remains untrained.'}, indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--tokenizer-repo',default='unsloth/Meta-Llama-3.1-8B')
    ap.add_argument('--prompt',default='The capital of France is')
    ap.add_argument('--steps',type=int,default=6)
    args=ap.parse_args()
    if args.steps<1 or args.steps>12:ap.error('--steps must be 1..12')
    run(args.checkpoint,args.tokenizer_repo,args.prompt,args.steps)
