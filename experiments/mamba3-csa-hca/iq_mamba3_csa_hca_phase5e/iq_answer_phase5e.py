"""IQ Phase 5e: paired-context, top-1-aware frozen-HCA-readout experiment.

Starts from the SHA-pinned accepted Phase 4d pilot. Holds all original 24 Mamba-3
layers, embeddings, CSA, and HCA selection/compressor weights fixed. Trains
only HCA value and output projections with *both* a factual input and an
otherwise-identical, single-token fact substitution for the same question.

Optimizes real frozen-Mamba-tail next-token NLL plus a hard-negative ranking
term; selects on development pairs, evaluates NEW key-disjoint locked pairs,
checks ordinary-language KL and a source intervention. Results are a *tiny
synthetic experiment*, not evidence of robust retrieval/general LM quality.
The CPU MIMO recurrence is not yet certified versus official CUDA kernels.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import iq_retrieval_phase5b as records
from iq_hybrid_cpu import MambaLMCPU, _toy_config, default_insertions, load_cpu_checkpoint
from iq_calibrate_cpu import checksum, event
from iq_retrieval_phase5 import plain_language_guard, Example
from iq_answer_phase5d import (cache_prefix, tail_logits, get_positions,
                                swap_context_answer, counterfactual_test)

# Completely disjoint from the first 24 keys used or revealed by phases 5–5d.
# IMPORTANT: Do not retune the run using these locked keys after seeing results.
FRESH_LOCKED_KEYS = [
    'lighthouse', 'juniper', 'waypoint', 'horizon', 'meerkat', 'wildflower',
    'northstar', 'bluebird', 'riverbend', 'pinecone', 'seabreeze', 'sunstone',
]

def ensure_fresh_keys():
    if len(records.PROJECTS) == 24:
        records.PROJECTS.extend(FRESH_LOCKED_KEYS)
    elif records.PROJECTS[24:24 + len(FRESH_LOCKED_KEYS)] != FRESH_LOCKED_KEYS:
        raise RuntimeError('Project registry mismatch')
    assert len(records.PROJECTS) >= 24 + len(FRESH_LOCKED_KEYS)


def rank_loss(logits, token_id, margin=0.5, temperature=1.0):
    """Differentiable penalty when any incorrect vocabulary token outranks target."""
    x = logits.float()
    target = x[0, token_id]
    incorrect = torch.cat((x[0, :token_id], x[0, token_id+1:])).max()
    return F.softplus((incorrect - target + margin) / temperature) * temperature


def answer_terms(logits, token_id, rank_weight, margin):
    y = torch.tensor([token_id], dtype=torch.long, device=logits.device)
    nll = F.cross_entropy(logits.float(), y)
    ranking = rank_loss(logits, token_id, margin=margin)
    return nll + rank_weight * ranking, nll.detach(), ranking.detach()


@torch.no_grad()
def pair_metrics(model, hca, pairs, gain, hca_pos):
    """Evaluate actual answer probabilities for BOTH evidence variants."""
    mean_nll = 0.0
    top1_orig = 0
    top1_swap = 0
    source_shift = 0.0
    rows = []
    for orig, p_orig, changed, p_changed in pairs:
        lo = tail_logits(model, hca, p_orig, gain, hca_pos)
        ls = tail_logits(model, hca, p_changed, gain, hca_pos)
        po = lo.log_softmax(-1)[0]
        ps = ls.log_softmax(-1)[0]
        nll_o = -float(po[orig.target_id]); nll_s = -float(ps[changed.target_id])
        m_o = float(po[orig.target_id] - po[changed.target_id])
        m_s = float(ps[orig.target_id] - ps[changed.target_id])
        hit_o = int(int(lo.argmax(-1)) == orig.target_id)
        hit_s = int(int(ls.argmax(-1)) == changed.target_id)
        source_shift += m_o - m_s
        top1_orig += hit_o
        top1_swap += hit_s
        mean_nll += nll_o + nll_s
        rows.append(dict(key=orig.key, original_answer=orig.answer,
                         swapped_answer=changed.answer, original_nll=nll_o,
                         swapped_nll=nll_s, original_top1=hit_o, swapped_top1=hit_s,
                         original_pair_margin=m_o, swapped_pair_margin=m_s,
                         evidence_shift=m_o - m_s))
    n = max(1,len(pairs))
    return dict(n=len(pairs), paired_answer_nll=mean_nll/(2*n),
                original_top1=top1_orig, swapped_top1=top1_swap,
                total_top1=top1_orig+top1_swap,
                mean_source_shift=source_shift/n, rows=rows)


def brief(metrics):
    return {k:v for k,v in metrics.items() if k!='rows'}


def freeze_and_train(model, mods, train, dev, gain, steps, lr, weight, margin,
                     checkpoint_tail=True, patience=3):
    for p in model.parameters():p.requires_grad_(False)
    for p in mods.parameters():p.requires_grad_(False)
    csa_pos, hca_pos = get_positions(mods)
    hca=mods[str(hca_pos)]
    all_frozen = {k:v.detach().clone() for k,v in hca.state_dict().items()
                  if k not in ('v_proj.weight','out_proj.weight')}
    params=[]
    for name in ('v_proj','out_proj'):
        for p in getattr(hca,name).parameters():
            p.requires_grad_(True)
            params.append(p)
    assert len(params)==2
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=0.)
    dev_before=pair_metrics(model,hca,dev,gain,hca_pos)
    best=copy.deepcopy(hca.state_dict())
    best_key=(-dev_before['total_top1'],dev_before['paired_answer_nll'])
    best_step=0;stale=0;hist=[]
    event('phase5e_dev_before',**brief(dev_before))
    for step in range(steps):
        orig, p_orig, changed, p_changed = train[step%len(train)]
        opt.zero_grad(set_to_none=True)
        train_nll=[];train_rank=[]
        # Separate backward calls limit activation memory on the CPU node.
        for ex,prefix in ((orig,p_orig),(changed,p_changed)):
            logits=tail_logits(model,hca,prefix,gain,hca_pos,checkpoint_tail)
            loss,nll,ranking=answer_terms(logits,ex.target_id,weight,margin)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite paired training loss')
            (loss*0.5).backward()
            train_nll.append(float(nll));train_rank.append(float(ranking))
            del logits,loss
        g=torch.nn.utils.clip_grad_norm_(params,1.)
        if not bool(torch.isfinite(g)):
            raise FloatingPointError('Nonfinite paired training gradient')
        opt.step()
        row=dict(step=step+1,key=orig.key,train_pair_nll=sum(train_nll)/2,
                 train_hard_negative_loss=sum(train_rank)/2,grad_norm=float(g))
        if (step+1)%2==0 or (step+1)==steps:
            metrics=pair_metrics(model,hca,dev,gain,hca_pos)
            row['dev']=brief(metrics)
            key=(-metrics['total_top1'],metrics['paired_answer_nll'])
            if key < best_key and metrics['paired_answer_nll'] < dev_before['paired_answer_nll'] + 0.1:
                best_key=key;best=copy.deepcopy(hca.state_dict())
                best_step=step+1;stale=0
            else:
                stale+=1
        hist.append(row)
        event('phase5e_train_step',**row)
        if stale>=patience:
            event('phase5e_early_stop',step=step+1,patience=patience)
            break
    hca.load_state_dict(best,strict=True)
    for k,v in all_frozen.items():
        if not torch.equal(hca.state_dict()[k],v):
            raise RuntimeError(f'Frozen HCA selector/compressor changed: {k}')
    for p in mods.parameters():p.requires_grad_(False)
    event('phase5e_selector_frozen_verified',tensors=sorted(all_frozen.keys()),
          trainable=['hca.v_proj','hca.out_proj'])
    dev_after=pair_metrics(model,hca,dev,gain,hca_pos)
    event('phase5e_dev_selected',best_step=best_step,**brief(dev_after))
    return hist,dev_before,dev_after,best_step


def toy_run():
    """No Hugging Face access needed; test paired gradients, frozen selector, key selection."""
    torch.manual_seed(171);torch.set_num_threads(2)
    model=MambaLMCPU(_toy_config()).eval()
    for p in model.parameters():p.requires_grad_(False)
    mods=default_insertions(model.cfg).eval()
    for m in mods.values():
        torch.nn.init.normal_(m.out_proj.weight,std=0.001)
    # Use same last-query frozen-tail path as real training.
    hca_pos=get_positions(mods)[1]
    original=torch.randint(0,model.cfg['vocab_size'],(1,260))
    changed=original.clone();changed[0,30]=(int(changed[0,30])+1)%model.cfg['vocab_size']
    o=Example('toy','one',int(original[0,-1]),24,original,'Q','S','train')
    s=Example('toy','two',int(changed[0,30]),24,changed,'Q','S','train')
    pair=(o,cache_prefix(model,mods,o.ids,0.02),s,cache_prefix(model,mods,s.ids,0.02))
    hist,d0,d1,step=freeze_and_train(model,mods,[pair],[pair],0.02,
                                    steps=2,lr=1e-4,weight=.5,margin=.5,
                                    checkpoint_tail=True)
    assert hist and all(math.isfinite(x['train_pair_nll']) for x in hist)
    assert all(not p.requires_grad for p in model.parameters())
    assert all(not p.requires_grad for p in mods.parameters())
    assert len(pair_metrics(model,mods[str(hca_pos)],[pair],.02,hca_pos)['rows'])==1
    event('phase5e_toy_pass',frozen_foundation=True,frozen_selector=True,
          paired_training=True,dev_selected_step=step)


def real_run(args):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    start=time.monotonic();torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    pilot_path=Path(args.pilot)
    pilot_sha=hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    if pilot_sha!=args.pilot_sha:raise ValueError('Phase4d SHA mismatch')
    pilot=torch.load(pilot_path,map_location='cpu',weights_only=True)
    model_path=hf_hub_download(records.MODEL_ID,filename='pytorch_model.bin')
    model,cfg=load_cpu_checkpoint(model_path)
    base_hash=checksum(model)
    if base_hash!=pilot['source_model_sha256']:raise ValueError('Pretrained foundation hash mismatch')
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    mods=records.load_pilot_modules(cfg,pilot).eval()
    gain=float(pilot['max_relative_norm'])
    csa_pos,hca_pos=get_positions(mods)
    tok=Tokenizer.from_file(hf_hub_download(records.TOKENIZER_ID,filename='tokenizer.json'))
    if tok.get_vocab_size(True)!=128256 or tok.token_to_id('<|begin_of_text|>')!=128000:
        raise ValueError('Tokenizer metadata mismatch')
    ensure_fresh_keys()
    train_cases=[records.make_record(tok,i,'train') for i in range(args.train_cases)]
    dev_cases=[records.make_record(tok,i+args.train_cases,'dev') for i in range(args.dev_cases)]
    locked_cases=[records.make_record(tok,i+args.locked_offset,'locked') for i in range(args.test_cases)]
    if ({x.key for x in locked_cases}&({x.key for x in train_cases}|{x.key for x in dev_cases})):
        raise ValueError('Locked keys overlap training or development')
    def build(xs,split):
        pairs=[]
        for i,ex in enumerate(xs):
            alternate=swap_context_answer(ex,tok)
            assert int((ex.ids!=alternate.ids).sum())==1
            pairs.append((ex,cache_prefix(model,mods,ex.ids,gain),
                          alternate,cache_prefix(model,mods,alternate.ids,gain)))
            event('phase5e_cached_pair',split=split,index=i+1,tokens=int(ex.ids.shape[1]),
                  key=ex.key)
        return pairs
    train=build(train_cases,'train');dev=build(dev_cases,'dev');locked=build(locked_cases,'locked')
    event('phase5e_setup',train_pairs=len(train),dev_pairs=len(dev),locked_pairs=len(locked),
          trainable=['hca.v_proj','hca.out_proj'],objective='paired_fact_conditioned_true_answer_NLL_plus_top1_ranking',
          fresh_locked_keys=[x.key for x in locked_cases],gain=gain,source_hash=base_hash)
    before_locked=pair_metrics(model,mods[str(hca_pos)],locked,gain,hca_pos)
    event('phase5e_locked_baseline',**brief(before_locked))
    history,dev_before,dev_after,best_step=freeze_and_train(
        model,mods,train,dev,gain,args.steps,args.lr,args.rank_weight,args.margin,
        checkpoint_tail=True,patience=args.patience)
    after_locked=pair_metrics(model,mods[str(hca_pos)],locked,gain,hca_pos)
    ordinary=plain_language_guard(model,mods,tok,gain)
    preserved=(checksum(model)==base_hash)
    # The shift is computed within matched same-question contexts; if it
    # improves it is evidence of context dependence, but not yet general QA.
    shift_gain=after_locked['mean_source_shift']-before_locked['mean_source_shift']
    checks={
        'development_paired_answer_nll_improved':dev_after['paired_answer_nll'] < dev_before['paired_answer_nll']-args.min_dev_gain,
        'locked_original_answer_accuracy_improved':after_locked['original_top1']>before_locked['original_top1'],
        'locked_swapped_answer_accuracy_improved':after_locked['swapped_top1']>before_locked['swapped_top1'],
        'locked_paired_nll_improved':after_locked['paired_answer_nll']<before_locked['paired_answer_nll']-args.min_locked_gain,
        'source_conditioning_improved':shift_gain>=args.min_shift_gain,
        'ordinary_language_preserved':all(x['same_top1'] and x['kl_nats']<=args.kl_budget for x in ordinary),
        'foundation_hash_unchanged':preserved,
    }
    report=dict(accepted=all(checks.values()),failed_checks=[k for k,v in checks.items() if not v],
                before_locked=before_locked,after_locked=after_locked,
                before_dev=brief(dev_before),after_dev=brief(dev_after),
                best_step=best_step,source_shift_gain=shift_gain,
                language_guards=ordinary,checks=checks,foundation_hash_unchanged=preserved,
                elapsed_seconds=round(time.monotonic()-start,2),
                limitations=['Synthetic project/code records, not broad QA',
                             'Only eight previously unused project keys in locked evaluation',
                             'Mamba CPU reference not numerically certified against official CUDA',
                             'CSA/HCA-inspired approximation, not DeepSeek reproduction'])
    dest=Path(args.report);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(dict(report=report,history=history),indent=2))
    event('phase5e_quality_gate',**{k:v for k,v in report.items() if k not in ('before_locked','after_locked')},
          locked_before=brief(before_locked),locked_after=brief(after_locked))
    event('phase5e_report_saved',path=str(dest))
    if not report['accepted']:
        event('phase5e_candidate_rejected',failed_checks=report['failed_checks'],pilot_unchanged=True)
        raise SystemExit(2)
    # Only accepted candidates get new checkpoint names; pilot is never edited.
    output=Path(args.output)
    torch.save(dict(attention_state_dict=mods.state_dict(),source_model=records.MODEL_ID,
                    source_model_sha256=base_hash,placement_after_blocks=[csa_pos,hca_pos],
                    max_relative_norm=gain,pilot_parent_sha256=pilot_sha,
                    phase5e_report=report,not_production_ready=True),output)
    event('phase5e_checkpoint_saved',path=str(output),bytes=output.stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--toy',action='store_true')
    p.add_argument('--pilot',default='/checkpoint/iq_attention_pilot.pt')
    p.add_argument('--pilot-sha',default='f7472337b7773c4c3fc71fe146b0d4519b5b88f9701e79967aa6e889364b6e53')
    p.add_argument('--report',default='/checkpoint/iq_phase5e_retrieval_report.json')
    p.add_argument('--output',default='/checkpoint/iq_phase5e_attention_candidate.pt')
    p.add_argument('--train-cases',type=int,default=8)
    p.add_argument('--dev-cases',type=int,default=4)
    p.add_argument('--test-cases',type=int,default=8)
    p.add_argument('--locked-offset',type=int,default=24)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--lr',type=float,default=0.00005)
    p.add_argument('--rank-weight',type=float,default=0.5)
    p.add_argument('--margin',type=float,default=0.5)
    p.add_argument('--min-dev-gain',type=float,default=0.005)
    p.add_argument('--min-locked-gain',type=float,default=0.01)
    p.add_argument('--min-shift-gain',type=float,default=0.02)
    p.add_argument('--kl-budget',type=float,default=0.01)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=47)
    args=p.parse_args()
    if args.toy:toy_run();return
    ensure_fresh_keys()
    if args.train_cases<4 or args.dev_cases<2 or args.test_cases<2:
        p.error('Need >=4 train, >=2 development, >=2 locked pairs')
    if args.locked_offset<24 or args.locked_offset+args.test_cases>len(records.PROJECTS):
        p.error('Locked keys must be fresh indices 24..35')
    if args.train_cases+args.dev_cases>24 or not (1<=args.steps<=30):
        p.error('Training/development keys must use first 24; steps 1..30')
    real_run(args)

if __name__=='__main__':main()
