"""IQ Phase 5 CPU retrieval experiment, using ORIGINAL pretrained Mamba-3 1.5B.

Continues from the accepted Phase 4d attention-only checkpoint. Trains only
CSA/HCA-inspired query/key projections on labeled compressed-memory locations.
Measures downstream answer-token likelihood and checks language preservation.

This is NOT DeepSeek's exact CSA/HCA; the source Mamba CPU recurrence is not
verified against official CUDA kernels. No full-backbone training is attempted.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from iq_hybrid_cpu import default_insertions, load_cpu_checkpoint
from iq_calibrate_cpu import (checksum, event, guarded_insertions,
                               hidden_taps)

MODEL_ID = 'state-spaces/mamba3-mimo-1.5b'
TOKENIZER_ID = 'unsloth/Meta-Llama-3.1-8B'


@dataclass
class Example:
    key: str
    answer: str
    target_id: int
    needle_at: int
    ids: torch.Tensor
    question_text: str
    source_text: str
    split: str


def _plain_ids(tok, s):
    return tok.encode(s, add_special_tokens=False).ids


def make_example(tok, *, index: int, split: str):
    """Construct answerable labeled long-context QA with distractors.

    The relevant evidence rotates between the first two complete 128-token
    compression pools. The prompt ends beyond token 260, so HCA can see both.
    No exact answer is repeated in the query, but the source contains it.
    """
    vocabulary = ['amber', 'cobalt', 'violet', 'silver', 'bronze', 'indigo',
                  'copper', 'orange', 'maple', 'cedar', 'quartz', 'green',
                  'scarlet', 'teal', 'yellow', 'blue']
    valid = [w for w in vocabulary if len(_plain_ids(tok, ' ' + w)) == 1]
    if len(valid) < 8:
        raise RuntimeError(f'Tokenizer unexpectedly lacks enough one-token targets: {valid}')
    answer = valid[(index * 3 + 1) % len(valid)]
    other = valid[(index * 3 + 4) % len(valid)]
    if answer == other:
        raise AssertionError('Wrong distractor')
    key = ['maple', 'delta', 'cedar', 'atlas', 'comet', 'harbor', 'ember', 'stone'][index % 8]
    distractor_key = ['river', 'orchid', 'planet', 'forest', 'valley', 'island', 'garden', 'canyon'][index % 8]
    # Distinct question wording and filler by dataset partition.
    needle_at = 24 if index % 2 == 0 else 168
    distractor_at = 168 if needle_at == 24 else 24
    filler = _plain_ids(tok, ' The archive records ordinary observations about the weather and the landscape.')
    if len(filler) == 0:
        raise RuntimeError('Empty filler')
    prefix = [128000] + (filler * (270 // len(filler) + 3))[:269]
    source_text = f' Project {key} has code {answer}.'
    decoy_text = f' Project {distractor_key} has code {other}.'
    for at, txt in ((needle_at, source_text), (distractor_at, decoy_text)):
        piece = _plain_ids(tok, txt)
        if at + len(piece) >= 256:
            raise AssertionError('Record does not fit into the first two completed HCA pools')
        prefix[at:at + len(piece)] = piece
    if len(prefix) != 270:
        raise AssertionError('Fixed-length record prefix changed')
    question_text = f' Question: What code belongs to Project {key}? Answer:'
    ids = prefix + _plain_ids(tok, question_text)
    if not 270 < len(ids) < 320:
        raise AssertionError('Unexpected context length')
    # Answer tokenization is checked exactly, no cross-vocabulary distillation.
    target_ids = _plain_ids(tok, ' ' + answer)
    assert len(target_ids) == 1
    return Example(key, answer, target_ids[0], needle_at,
                   torch.tensor([ids], dtype=torch.long), question_text, source_text, split)


def pooled_index_scores(module, x):
    """Supervised attention *selection* objective; no LM-head shortcut.

    Scores the completed compressed pools for the final question token.
    This trains Q and K on known evidence locations rather than reconstructing
    an arbitrary prefix activation. Does not change V/output projections.
    """
    z = module.norm(x)
    b, t, _ = z.shape
    h, dim, kh = module.heads, module.dim, module.kv_heads
    nfull = t // module.ratio
    if nfull <= 1:
        raise ValueError(f'{module.kind} needs >=2 completed memory pools, got {nfull}')
    q = module.q_proj(z[:, -1:, :]).reshape(b, 1, h, dim).transpose(1, 2)
    k = module.k_proj(z[:, :nfull * module.ratio]).reshape(b, nfull, module.ratio, kh, dim)
    # Mean within each compression pool, then broadcast grouped KV heads.
    k = k.mean(dim=2).permute(0, 2, 1, 3).repeat_interleave(h // kh, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)).squeeze(-2) / math.sqrt(dim)
    # [batch, heads, completed pools]
    return scores


def selector_ce(module, x, target_pos):
    scores = pooled_index_scores(module, x)
    pool = target_pos // module.ratio
    if pool >= scores.shape[-1]:
        raise ValueError('Evidence pool is not causally available')
    y = torch.full((scores.shape[0] * scores.shape[1],), pool,
                   dtype=torch.long, device=x.device)
    return F.cross_entropy(scores.reshape(-1, scores.shape[-1]), y)


@torch.no_grad()
def selector_stats(modules, examples_and_taps):
    hits = {m: 0 for m in modules}
    losses = {m: 0.0 for m in modules}
    for ex, taps in examples_and_taps:
        for pos, mod in modules.items():
            s = pooled_index_scores(mod, taps[pos])
            pool = ex.needle_at // mod.ratio
            if pool >= s.shape[-1]:
                raise RuntimeError('Expected pool unavailable')
            # Selection hit = evidence pool ranks first on average across heads.
            correct = int(s.mean(dim=1).argmax(-1).item() == pool)
            hits[pos] += correct
            losses[pos] += float(selector_ce(mod, taps[pos], ex.needle_at))
    n = max(1, len(examples_and_taps))
    return {modules[pos].kind: {
        'top1_pool_hits': hits[pos], 'count': n, 'top1_pool_accuracy': hits[pos] / n,
        'selection_cross_entropy': losses[pos] / n,
    } for pos in modules}


def train_selectors(modules, examples_and_taps, steps, learning_rate):
    if steps <= 0:
        return []
    # Freeze output and value paths from the accepted Phase 4d pilot.
    train_params = []
    for mod in modules.values():
        for p in mod.parameters():
            p.requires_grad_(False)
        for p in (mod.q_proj.weight, mod.k_proj.weight):
            p.requires_grad_(True)
            train_params.append(p)
    optimizer = torch.optim.AdamW(train_params, lr=learning_rate, weight_decay=0)
    report = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = [selector_ce(modules[pos], taps[pos], ex.needle_at)
                  for ex, taps in examples_and_taps for pos in modules]
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite selector training objective')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        if not bool(torch.isfinite(grad)):
            raise FloatingPointError('Non-finite selector gradient norm')
        optimizer.step()
        item = {'step': step + 1, 'selector_loss': float(loss.detach()),
                'grad_norm': float(grad)}
        report.append(item)
        event('retrieval_selector_train_step', **item)
    for p in modules.parameters():
        p.requires_grad_(False)
    return report


@torch.no_grad()
def answer_probe(model, modules, ex, gain):
    base_logits = model(ex.ids)[:, -1, :].float()
    hybrid_logits = model(ex.ids, guarded_insertions(modules, gain))[:, -1, :].float()
    base_logp = base_logits.log_softmax(-1)
    hybrid_logp = hybrid_logits.log_softmax(-1)
    y = ex.target_id
    kl = float((base_logp.exp() * (base_logp - hybrid_logp)).sum().clamp_min(0))
    return {
        'split': ex.split, 'key': ex.key, 'answer': ex.answer,
        'needle_at': ex.needle_at, 'sequence_tokens': ex.ids.shape[1],
        'target_token_id': y,
        'baseline_target_nll': -float(base_logp[0, y]),
        'hybrid_target_nll': -float(hybrid_logp[0, y]),
        'baseline_top1_correct': int(base_logits.argmax(-1).item() == y),
        'hybrid_top1_correct': int(hybrid_logits.argmax(-1).item() == y),
        'kl_nats': kl, 'active_logit_delta': float((hybrid_logits - base_logits).abs().max()),
    }


@torch.no_grad()
def plain_language_guard(model, modules, tokenizer, gain):
    prompts = ['The capital of France is',
               'The largest planet in the solar system is',
               'Two plus two equals']
    results = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=True).ids
        x = torch.tensor([ids], dtype=torch.long)
        baseline = model(x)[:, -1, :].float()
        hybrid = model(x, guarded_insertions(modules, gain))[:, -1, :].float()
        q = baseline.log_softmax(-1)
        p = hybrid.log_softmax(-1)
        results.append({'prompt': prompt,
                        'same_top1': int(baseline.argmax(-1)) == int(hybrid.argmax(-1)),
                        'kl_nats': float((q.exp() * (q - p)).sum().clamp_min(0)),
                        'max_abs_logit_delta': float((hybrid - baseline).abs().max())})
    return results


def toy_run():
    from iq_hybrid_cpu import _toy_config, MambaLMCPU
    torch.set_num_threads(2)
    torch.manual_seed(9)
    cfg = _toy_config()
    model = MambaLMCPU(cfg)
    for p in model.parameters(): p.requires_grad_(False)
    modules = default_insertions(cfg)
    for mod in modules.values():
        with torch.no_grad():
            nn.init.normal_(mod.out_proj.weight, std=.001)
    # Standalone selector supervision using generated hidden states.
    exs=[]
    for i in range(4):
        needle = 16 if i%2 == 0 else 138
        ids = torch.randint(0,cfg['vocab_size'],(1,264))
        taps = hidden_taps(model, ids, modules.keys())
        ex = Example('toy','toy',3,needle,ids,'','', 'toy')
        exs.append((ex, taps))
    initial=selector_stats(modules,exs)
    report=train_selectors(modules,exs,steps=8,learning_rate=2e-4)
    final=selector_stats(modules,exs)
    assert all(final[m]['selection_cross_entropy'] < initial[m]['selection_cross_entropy']
               for m in ('csa','hca')), (initial,final)
    assert report and all(math.isfinite(x['selector_loss']) for x in report)
    assert all(not p.requires_grad for p in model.parameters())
    event('toy_retrieval_selector_pass', initial=initial, final=final,
          note='Synthetic hidden states only, not trained pretrained-model retrieval')


def real_run(args):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    pilot_path = Path(args.pilot)
    if not pilot_path.is_file():
        raise FileNotFoundError(f'Phase 4d checkpoint missing: {pilot_path}')
    pilot_sha = __import__('hashlib').sha256(pilot_path.read_bytes()).hexdigest()
    if args.expected_pilot_sha256 and pilot_sha != args.expected_pilot_sha256:
        raise ValueError(f'Pilot SHA-256 mismatch: {pilot_sha} != {args.expected_pilot_sha256}')
    checkpoint = torch.load(pilot_path, map_location='cpu', weights_only=True)
    if checkpoint.get('source_model') != MODEL_ID:
        raise ValueError('Phase 4d attention source model mismatch')
    if checkpoint.get('placement_after_blocks') != [8,16]:
        raise ValueError('Phase 4d attention placement mismatch')
    if checkpoint.get('csa_ratio') != 4 or checkpoint.get('hca_ratio') != 128:
        raise ValueError('Phase 4d compression configuration mismatch')
    gain = float(checkpoint['max_relative_norm'])
    if not 0 < gain <= 0.1:
        raise ValueError('Unsafe Phase 4d residual bound')
    event('pilot_loaded', path=str(pilot_path), sha256=pilot_sha,
          selected_gain=gain, source_model=checkpoint['source_model'])
    model_path = Path(args.model_checkpoint) if args.model_checkpoint else Path(
        hf_hub_download(MODEL_ID, filename='pytorch_model.bin'))
    model,cfg = load_cpu_checkpoint(model_path)
    baseline_hash = checksum(model)
    if checkpoint.get('source_model_sha256') != baseline_hash:
        raise ValueError('Model weights do not match the Phase 4d reference hash')
    modules = default_insertions(cfg)
    modules.load_state_dict(checkpoint['attention_state_dict'], strict=True)
    modules.eval()
    tokenizer_path = hf_hub_download(TOKENIZER_ID, filename='tokenizer.json')
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.get_vocab_size(True) != 128256 or tokenizer.token_to_id('<|begin_of_text|>') != 128000:
        raise ValueError('Unexpected Llama tokenizer mapping')
    examples_train = [make_example(tokenizer,index=i,split='train') for i in range(args.train_cases)]
    examples_test = [make_example(tokenizer,index=i+args.train_cases,split='heldout') for i in range(args.heldout_cases)]
    event('qa_dataset', train_count=len(examples_train), heldout_count=len(examples_test),
          min_length=min(e.ids.shape[1] for e in examples_train+examples_test),
          max_length=max(e.ids.shape[1] for e in examples_train+examples_test),
          note='Controlled synthetic needle retrieval, distinct train/holdout records')

    # Save initial states in-memory; never write checkpoint until assessment.
    pilot_state=copy.deepcopy(modules.state_dict())
    # Obtain frozen representations once: CPU-friendly local selector training.
    train_taps=[]
    for i,ex in enumerate(examples_train):
        taps=hidden_taps(model,ex.ids,modules.keys())
        train_taps.append((ex,taps))
        event('training_tap_cached',case=i+1,seq_tokens=ex.ids.shape[1],needle_at=ex.needle_at)
    test_taps=[]
    for i,ex in enumerate(examples_test):
        taps=hidden_taps(model,ex.ids,modules.keys())
        test_taps.append((ex,taps))
        event('heldout_tap_cached',case=i+1,seq_tokens=ex.ids.shape[1],needle_at=ex.needle_at)
    initial_train=selector_stats(modules,train_taps)
    initial_test=selector_stats(modules,test_taps)
    event('selector_before_training', train=initial_train, heldout=initial_test)
    training=train_selectors(modules,train_taps,steps=args.steps,learning_rate=args.learning_rate)
    final_train=selector_stats(modules,train_taps)
    final_test=selector_stats(modules,test_taps)
    event('selector_after_training', train=final_train, heldout=final_test)
    # Whole-model answer likelihood with new attention vs original pretrained Mamba.
    before_results=[]
    new_results=[]
    for ex in examples_test:
        before_modules=default_insertions(cfg)
        before_modules.load_state_dict(pilot_state)
        before_modules.eval()
        before_results.append(answer_probe(model,before_modules,ex,gain))
        new_results.append(answer_probe(model,modules,ex,gain))
    guards=plain_language_guard(model,modules,tokenizer,gain)
    for i,(old,new) in enumerate(zip(before_results,new_results)):
        event('qa_heldout_comparison',case=i+1,pilot=old,trained=new)
    for result in guards:event('language_preservation_probe',**result)
    for p in modules.parameters():p.requires_grad_(False)
    final_hash=checksum(model)
    if final_hash != baseline_hash:
        raise AssertionError('ORIGINAL MAMBA PARAMETER HASH ALTERED')
    train_improved=all(final_train[m]['selection_cross_entropy'] < initial_train[m]['selection_cross_entropy']
                       for m in ('csa','hca'))
    heldout_selector_improved=all(final_test[m]['selection_cross_entropy'] < initial_test[m]['selection_cross_entropy']
                                 for m in ('csa','hca'))
    old_nll=sum(x['hybrid_target_nll'] for x in before_results)/len(before_results)
    new_nll=sum(x['hybrid_target_nll'] for x in new_results)/len(new_results)
    language_ok=all(x['same_top1'] and x['kl_nats'] <= args.kl_budget for x in guards)
    accepted=bool(train_improved and heldout_selector_improved and
                  new_nll < old_nll - args.nll_min_improvement and language_ok and
                  final_hash==baseline_hash)
    reasons=[]
    if not train_improved:reasons.append('selector_train_loss_not_improved')
    if not heldout_selector_improved:reasons.append('heldout_selector_loss_not_improved')
    if not new_nll < old_nll - args.nll_min_improvement:reasons.append('heldout_answer_likelihood_not_improved')
    if not language_ok:reasons.append('ordinary_language_regression')
    summary={'accepted':accepted,'reasons':reasons,
             'train_selector_before':initial_train,'train_selector_after':final_train,
             'heldout_selector_before':initial_test,'heldout_selector_after':final_test,
             'heldout_pilot_answer_nll':old_nll,'heldout_candidate_answer_nll':new_nll,
             'heldout_answer_nll_delta':new_nll-old_nll,
             'language_guards':guards,'foundation_hash_unchanged':final_hash==baseline_hash,
             'train_steps':args.steps,'seconds_total':round(time.monotonic()-started,2),
             'limitations':['CSA/HCA-inspired, not exact DeepSeek implementation',
                            'CPU MIMO reference is not cross-validated with official CUDA kernel',
                            'Synthetic QA and narrow language probes are not broad benchmark proof']}
    event('phase5_quality_gate',**summary)
    reportpath=Path(args.report)
    reportpath.parent.mkdir(parents=True,exist_ok=True)
    with reportpath.open('w') as f:
        json.dump({'summary':summary,'training':training,'cases':new_results},f,indent=2)
    event('phase5_report_saved',path=str(reportpath))
    if accepted:
        out=Path(args.output)
        out.parent.mkdir(parents=True,exist_ok=True)
        state={'attention_state_dict':modules.state_dict(),
               'source_model':MODEL_ID,'source_model_sha256':baseline_hash,
               'placement_after_blocks':[8,16],'csa_ratio':4,'hca_ratio':128,
               'max_relative_norm':gain,'phase5_kind':'supervised pool-selection pilot',
               'pilot_parent_sha256': __import__('hashlib').sha256(pilot_path.read_bytes()).hexdigest(),
               'phase5_summary':summary,'not_production_ready':True}
        torch.save(state,out)
        event('phase5_attention_checkpoint_saved',path=str(out),bytes=out.stat().st_size)
    else:
        event('phase5_candidate_rejected',reasons=reasons,
              note='Phase 4d pilot remains untouched; diagnostic metrics saved')
        raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--toy',action='store_true')
    p.add_argument('--pilot',default='/checkpoint/iq_attention_pilot.pt')
    p.add_argument('--expected-pilot-sha256', default='f7472337b7773c4c3fc71fe146b0d4519b5b88f9701e79967aa6e889364b6e53')
    p.add_argument('--model-checkpoint',default='')
    p.add_argument('--output',default='/checkpoint/iq_phase5_attention_candidate.pt')
    p.add_argument('--report',default='/checkpoint/iq_phase5_retrieval_report.json')
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--train-cases',type=int,default=4)
    p.add_argument('--heldout-cases',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--kl-budget',type=float,default=0.01)
    p.add_argument('--nll-min-improvement',type=float,default=0)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=47)
    args=p.parse_args()
    if args.toy:toy_run();return
    if not 2 <= args.train_cases <= 8:p.error('--train-cases must be 2..8')
    if not 1 <= args.heldout_cases <= 4:p.error('--heldout-cases must be 1..4')
    if args.train_cases + args.heldout_cases > 8: p.error('train + heldout cases must be <=8 to keep unique project keys')
    if not 0 <= args.steps <= 100:p.error('--steps must be 0..100')
    if not 0 < args.learning_rate <= 1e-3:p.error('learning rate must be (0,1e-3]')
    real_run(args)

if __name__=='__main__':main()
