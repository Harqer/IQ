"""IQ Phase 5d: frozen HCA selector, train answer-bearing value/output path.

Experimental CPU-only pilot. Original Mamba-3 backbone and the Phase 4d
checkpoint remain read-only. We freeze CSA and the first 16 Mamba blocks,
cache their exact outputs with the frozen pilot CSA insertion, and backpropagate
answer-token NLL through the *real frozen final eight Mamba blocks* to HCA.
Only the existing HCA V/O projections at block 16 are trained; its Q/K/pool selector
and the entire pretrained foundation stay frozen (no new transformer layer).

This CPU Mamba reference has NOT been numerically verified against the official
CUDA TileLang implementation. Passing this 4-example QA gate is not proof of
real-world retrieval/general-language quality or of genuine DeepSeek HCA.
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
from torch import nn
from torch.utils.checkpoint import checkpoint

from iq_hybrid_cpu import MambaLMCPU, _toy_config, default_insertions, load_cpu_checkpoint
from iq_calibrate_cpu import BoundedInsertion, checksum, event, guarded_insertions
from iq_retrieval_phase5 import answer_probe, plain_language_guard, _plain_ids, Example
from iq_retrieval_phase5b import (MODEL_ID, TOKENIZER_ID, WORDS, make_record,
                                 load_pilot_modules, pool_ce,
                                 last_query_pool_scores)


def get_positions(modules):
    positions=sorted(int(x) for x in modules.keys())
    if len(positions)!=2:
        raise ValueError('Expected exactly two CSA/HCA placements')
    a,b=positions
    if modules[str(a)].kind!='csa' or modules[str(b)].kind!='hca':
        raise ValueError('Placements must be CSA then HCA')
    return a,b


@torch.no_grad()
def cache_prefix(model, mods, ids, gain):
    """Compute frozen first blocks including CSA, then cache the HCA boundary.

    The returned x,residual are exactly the pair the next Mamba layer receives
    before the HCA insertion (not a standalone tapped hidden state).
    """
    csa_pos,hca_pos=get_positions(mods)
    if csa_pos>=hca_pos or hca_pos>=len(model.backbone.layers):
        raise ValueError('Invalid HCA insertion position')
    x=model.backbone.embedding(ids)
    residual=None
    csa=BoundedInsertion(mods[str(csa_pos)],gain)
    for i in range(1,hca_pos+1):
        x,residual=model.backbone.layers[i-1](x,residual)
        if i==csa_pos:
            x=x+csa(x+residual)
    return x.detach().clone(), residual.detach().clone()


def tail_logits(model, hca_mod, prefix, gain, hca_pos, checkpoint_tail=False):
    """Exact continuation after the HCA insertion, including frozen layers.

    Trainable HCA receives gradient through these unchanged Mamba blocks.
    Does not approximate target logits with embedding-direction alignment.
    """
    x,residual=prefix
    x=x+BoundedInsertion(hca_mod,gain)(x+residual)
    for block in model.backbone.layers[hca_pos:]:
        if checkpoint_tail and torch.is_grad_enabled():
            # Checkpoint the frozen block's activations, but still propagate
            # gradients to HCA through its frozen computations.
            x,residual=checkpoint(lambda xx,rr,blk=block:blk(xx,rr),
                                  x,residual,use_reentrant=False)
        else:
            x,residual=block(x,residual)
    final=model.backbone.norm_f((x+residual)[:,-1:,:])[:,-1,:]
    return model.lm_head(final).float()


def case_loss(model,hca,prefix,ex,gain,hca_pos,selector_weight,checkpoint_tail):
    logits=tail_logits(model,hca,prefix,gain,hca_pos,checkpoint_tail)
    label=torch.tensor([ex.target_id],dtype=torch.long,device=logits.device)
    ce=F.cross_entropy(logits,label)
    # Selector is completely frozen. Report its CE without optimizing it.
    with torch.no_grad():
        sel=pool_ce(hca,prefix[0]+prefix[1],ex.needle_at)
    return ce, ce, sel


@torch.no_grad()
def cached_answer_eval(model,hca,data,gain,hca_pos):
    total=0.;hits=0;pool_hit=0;pool_ce_total=0.
    for ex,prefix in data:
        logits=tail_logits(model,hca,prefix,gain,hca_pos)
        logp=logits.log_softmax(-1)
        total+=-float(logp[0,ex.target_id])
        hits+=int(int(logits.argmax(-1))==ex.target_id)
        with torch.no_grad():
            scores=last_query_pool_scores(hca,prefix[0]+prefix[1])
            pool_hit+=int(scores.mean(dim=1).argmax(-1).item()==ex.needle_at//hca.ratio)
            pool_ce_total+=float(pool_ce(hca,prefix[0]+prefix[1],ex.needle_at))
    n=max(1,len(data))
    return dict(n=len(data),answer_nll=total/n,answer_top1=hits,
                hca_pool_hits=pool_hit,hca_pool_ce=pool_ce_total/n)


def train_hca(model,mods,train,dev,gain,steps,lr,selector_weight,
              checkpoint_tail=False,patience=2):
    hca_pos=get_positions(mods)[1]
    for m in mods.values():
        m.eval()
        for p in m.parameters():p.requires_grad_(False)
    hca=mods[str(hca_pos)]
    # Only existing HCA weights are updated. Foundation + CSA stay frozen.
    for name in ('v_proj','out_proj'):
        for p in getattr(hca,name).parameters():p.requires_grad_(True)
    params=[p for p in hca.parameters() if p.requires_grad]
    assert len(params)==2, 'V/O-only training contract violated'
    frozen_selector={k:v.detach().clone() for k,v in hca.state_dict().items()
                     if k not in ('v_proj.weight','out_proj.weight')}
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=0.)
    before_dev=cached_answer_eval(model,hca,dev,gain,hca_pos)
    best=copy.deepcopy(hca.state_dict());best_dev=before_dev['answer_nll']
    best_step=0;stale=0;history=[]
    event('phase5d_dev_before',**before_dev)
    for step in range(steps):
        ex,prefix=train[step%len(train)]
        opt.zero_grad(set_to_none=True)
        loss,answer_ce,selector_ce=case_loss(model,hca,prefix,ex,gain,hca_pos,
                                              selector_weight,checkpoint_tail)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite Phase 5d loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(params,max_norm=1.)
        if not bool(torch.isfinite(grad)):raise FloatingPointError('Nonfinite Phase 5d gradient')
        opt.step()
        item=dict(step=step+1,example=ex.key,train_answer_nll=float(answer_ce.detach()),
                  train_selector_ce=float(selector_ce.detach()),grad_norm=float(grad))
        # Model selection uses only development REAL answer NLL; never locked QA.
        if (step+1)%2==0 or step+1==steps:
            dev_now=cached_answer_eval(model,hca,dev,gain,hca_pos)
            item['dev']=dev_now
            if dev_now['answer_nll']<best_dev-1e-5:
                best_dev=dev_now['answer_nll'];best=copy.deepcopy(hca.state_dict())
                best_step=step+1;stale=0
            else:
                stale+=1
        history.append(item)
        event('phase5d_train_step',**item)
        if stale>=patience:
            event('phase5d_early_stop',at_step=step+1,patience=patience)
            break
    hca.load_state_dict(best,strict=True)
    for k,before in frozen_selector.items():
        if not torch.equal(hca.state_dict()[k],before):
            raise RuntimeError(f'Frozen HCA selector mutated: {k}')
    for p in hca.parameters():p.requires_grad_(False)
    event('phase5d_selector_frozen_verified',checked_tensors=list(frozen_selector),
          trainable_projections=['v_proj','out_proj'])
    after_dev=cached_answer_eval(model,hca,dev,gain,hca_pos)
    event('phase5d_dev_selected',best_step=best_step,**after_dev)
    return history,before_dev,after_dev,best_step



def swap_context_answer(ex, tok):
    """Construct a causally comparable question with ONLY the evidence answer token changed.

    The question/key, needle placement, distractor and every other token stay identical.
    Fail closed if the tokenizer cannot represent a one-token replacement.
    """
    original=_plain_ids(tok,ex.source_text)
    at=int(ex.needle_at)
    actual=ex.ids[0,at:at+len(original)].tolist()
    if actual!=original:
        raise AssertionError('Needle evidence tokenization mismatch')
    orig_answer=_plain_ids(tok,' '+ex.answer)
    if len(orig_answer)!=1:
        raise ValueError('Original target is not single token')
    hit=[j for j,v in enumerate(original) if v==orig_answer[0]]
    if len(hit)!=1:
        raise ValueError('Answer token is ambiguous within source')
    # Rotate the answer in the fixed vocabulary, rather than always choosing
    # the same distractor token for every test example.
    start=WORDS.index(ex.answer) if ex.answer in WORDS else 0
    for offset in range(1,len(WORDS)):
        word=WORDS[(start+7+offset-1)%len(WORDS)]
        if word==ex.answer:continue
        replacement=_plain_ids(tok,' '+word)
        if len(replacement)!=1 or replacement[0]==orig_answer[0]:continue
        other=word;other_id=replacement[0];break
    else:raise ValueError('No distinct one-token counterfactual answer')
    new_ids=ex.ids.clone()
    new_ids[0,at+hit[0]]=other_id
    assert (new_ids!=ex.ids).sum().item()==1
    modified=Example(ex.key,other,other_id,at,new_ids,ex.question_text,
                     ex.source_text.replace(' '+ex.answer+'.',' '+other+'.'),ex.split)
    return modified


@torch.no_grad()
def counterfactual_test(model,pilot_modules,trained_modules,locked,tok,gain):
    """Measure how changing ONLY the source answer changes paired answer preferences.

    Reports baseline vs pilot vs trained. This is a diagnostic causal intervention,
    not proof of semantic retrieval or a substitute for a larger held-out benchmark.
    """
    results=[]
    for ex in locked:
        changed=swap_context_answer(ex,tok)
        # Full-model forward is deliberately used to test actual readout behavior.
        def margin(modules,case):
            logits=model(case.ids,guarded_insertions(modules,gain))[:,-1,:].float()
            logp=logits.log_softmax(-1)
            return float(logp[0,ex.target_id]-logp[0,changed.target_id])
        orig_pilot=margin(pilot_modules,ex)
        swap_pilot=margin(pilot_modules,changed)
        orig_trained=margin(trained_modules,ex)
        swap_trained=margin(trained_modules,changed)
        pilot_shift=orig_pilot-swap_pilot
        trained_shift=orig_trained-swap_trained
        results.append({'key':ex.key,'original_answer':ex.answer,
                        'swapped_answer':changed.answer,
                        'pilot_context_shift':pilot_shift,
                        'candidate_context_shift':trained_shift,
                        'context_shift_gain':trained_shift-pilot_shift,
                        'candidate_original_margin':orig_trained,
                        'candidate_swapped_margin_for_original':swap_trained})
    delta=sum(x['context_shift_gain'] for x in results)/len(results)
    positive=sum(x['context_shift_gain']>0 for x in results)
    return {'cases':results,'mean_context_shift_gain':delta,
            'positive_case_count':positive,'count':len(results)}

def toy_run():
    torch.manual_seed(71);torch.set_num_threads(2)
    model=MambaLMCPU(_toy_config()).eval()
    for p in model.parameters():p.requires_grad_(False)
    mods=default_insertions(model.cfg).eval()
    for m in mods.values():
        nn.init.normal_(m.out_proj.weight,std=1e-3)
    gain=.02
    tokens=torch.randint(0,model.cfg['vocab_size'],(1,260))
    prefix=cache_prefix(model,mods,tokens,gain)
    hca_pos=get_positions(mods)[1]
    with torch.no_grad():
        full=model(tokens,guarded_insertions(mods,gain))[:,-1].float()
        tail=tail_logits(model,mods[str(hca_pos)],prefix,gain,hca_pos)
    err=float((full-tail).abs().max())
    assert err<1e-4, ('Frozen tail mismatch with actual model forward',err)
    exs=[]
    from iq_retrieval_phase5 import Example
    for i in range(4):
        ids=torch.randint(0,model.cfg['vocab_size'],(1,260))
        ex=Example('key'+str(i),'word'+str(i),int(ids[0,-1]),24 if i%2==0 else 168,
                   ids,'question','source','train' if i<3 else 'dev')
        exs.append((ex,cache_prefix(model,mods,ids,gain)))
    # True end-to-end frozen-tail answer NLL must backpropagate into HCA.
    hist,prev,post,best=train_hca(model,mods,exs[:3],exs[3:],gain,steps=2,
                                  lr=1e-4,selector_weight=.01,checkpoint_tail=True)
    assert hist and math.isfinite(prev['answer_nll']) and math.isfinite(post['answer_nll'])
    assert all(not p.requires_grad for p in model.parameters())
    # Check actual differentiability of the answer loss w.r.t. V and output.
    for p in mods[str(hca_pos)].parameters():p.requires_grad_(False)
    v=mods[str(hca_pos)].v_proj.weight
    o=mods[str(hca_pos)].out_proj.weight
    v.requires_grad_(True);o.requires_grad_(True)
    l,_,_=case_loss(model,mods[str(hca_pos)],exs[0][1],exs[0][0],gain,hca_pos,0,False)
    gv,go=torch.autograd.grad(l,(v,o),allow_unused=False)
    assert bool(torch.isfinite(gv).all()) and bool(torch.isfinite(go).all())
    assert float(gv.abs().sum())>0 and float(go.abs().sum())>0
    v.requires_grad_(False);o.requires_grad_(False)
    assert all(not p.requires_grad for name,p in mods[str(hca_pos)].named_parameters()
               if name.startswith(('q_proj.','k_proj.','pool_score_proj.'))), 'Selector unexpectedly trainable'
    event('phase5d_toy_pass',full_tail_max_logit_error=err,
          frozen_backbone=True,real_answer_loss_gradients=True,
          v_and_o_gradients_nonzero=True,
          dev_selected_step=best,steps_run=len(hist))


def real_run(args):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    started=time.monotonic()
    torch.manual_seed(args.seed);torch.set_num_threads(args.threads)
    pilot_path=Path(args.pilot)
    pilotsha=hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    if pilotsha!=args.pilot_sha:raise ValueError('Pilot SHA mismatch')
    pilot=torch.load(pilot_path,map_location='cpu',weights_only=True)
    model_path=hf_hub_download(MODEL_ID,filename='pytorch_model.bin')
    model,cfg=load_cpu_checkpoint(model_path)
    foundation_hash=checksum(model)
    if foundation_hash!=pilot['source_model_sha256']:
        raise ValueError('Pretrained Mamba-3 source checkpoint mismatch')
    modules=load_pilot_modules(cfg,pilot).eval()
    gain=float(pilot['max_relative_norm'])
    csa_pos,hca_pos=get_positions(modules)
    tokenizer=Tokenizer.from_file(hf_hub_download(TOKENIZER_ID,filename='tokenizer.json'))
    if tokenizer.get_vocab_size(True)!=128256 or tokenizer.token_to_id('<|begin_of_text|>')!=128000:
        raise ValueError('Tokenizer ID assumptions violated')
    train=[make_record(tokenizer,i,'train') for i in range(args.train_cases)]
    dev=[make_record(tokenizer,i+args.train_cases,'dev') for i in range(args.dev_cases)]
    locked=[make_record(tokenizer,i+args.locked_offset,'locked') for i in range(args.test_cases)]
    if args.locked_offset<args.train_cases+args.dev_cases:
        raise ValueError('Locked test indices must not overlap train or dev')
    if max(args.locked_offset+args.test_cases,args.train_cases+args.dev_cases)>24:
        raise ValueError('Projects exhausted')
    event('phase5d_setup',train=len(train),dev=len(dev),locked=len(locked),
          pilot_sha256=pilotsha,hca_pos=hca_pos,csa_pos=csa_pos,gain=gain,
          training_objective='actual_frozen_Mamba_tail_answer_NLL',
          trainable_paths=['hca.v_proj','hca.out_proj'],
          locked_dataset_offset=args.locked_offset,
          locked_dataset_note='Fresh indices 16-23; prior phases exposed indices 12-15')
    def cache(xs,split):
        result=[]
        for i,ex in enumerate(xs):
            result.append((ex,cache_prefix(model,modules,ex.ids,gain)))
            event('phase5d_cached',split=split,case=i+1,tokens=ex.ids.shape[1])
        return result
    train_cache=cache(train,'train')
    dev_cache=cache(dev,'dev')
    locked_cache=cache(locked,'locked')
    before_locked=cached_answer_eval(model,modules[str(hca_pos)],locked_cache,gain,hca_pos)
    event('phase5d_locked_baseline_measured',**before_locked)
    history,dev_before,dev_after,step=train_hca(model,modules,train_cache,dev_cache,
          gain,args.steps,args.lr,args.selector_weight,checkpoint_tail=True,patience=args.patience)
    # Independent locked QA probe: report actual full-model answer metrics.
    after_locked=cached_answer_eval(model,modules[str(hca_pos)],locked_cache,gain,hca_pos)
    pilot_modules=load_pilot_modules(cfg,pilot).eval()
    pilot_locked=[answer_probe(model,pilot_modules,ex,gain) for ex in locked]
    candidate_locked=[answer_probe(model,modules,ex,gain) for ex in locked]
    ordinary=plain_language_guard(model,modules,tokenizer,gain)
    counterfactual=counterfactual_test(model,pilot_modules,modules,locked,tokenizer,gain)
    event('phase5d_counterfactual',**counterfactual)
    if checksum(model)!=foundation_hash:raise RuntimeError('Frozen foundation changed')
    old_nll=sum(r['hybrid_target_nll'] for r in pilot_locked)/len(pilot_locked)
    new_nll=sum(r['hybrid_target_nll'] for r in candidate_locked)/len(candidate_locked)
    old_hits=sum(r['hybrid_top1_correct'] for r in pilot_locked)
    new_hits=sum(r['hybrid_top1_correct'] for r in candidate_locked)
    # Do not accept purely improved surrogate QA; require full-model predictions.
    heldout_answer_gain=(new_hits>old_hits and new_nll<old_nll-args.nll_margin)
    selector_nondegrade=(after_locked['hca_pool_hits']==before_locked['hca_pool_hits']
         and abs(after_locked['hca_pool_ce']-before_locked['hca_pool_ce'])<1e-6)
    # Freeze is verified by torch.equal in train_hca; this checks the metric too.
    context_gain=(counterfactual['mean_context_shift_gain']>args.min_context_gain
                  and counterfactual['positive_case_count']>=args.min_positive_cases)
    language_ok=all(x['same_top1'] and x['kl_nats']<=args.kl_budget for x in ordinary)
    dev_gain=dev_after['answer_nll']<dev_before['answer_nll']-1e-5
    checks={
        'dev_actual_answer_nll_improved':dev_gain,
        'locked_actual_answer_accuracy_and_nll_improved':heldout_answer_gain,
        'locked_hca_selector_unchanged':selector_nondegrade,
        'context_intervention_sensitivity_improved':context_gain,
        'ordinary_language_preserved':language_ok,
        'foundation_hash_unchanged':True}
    failed=[name for name,ok in checks.items() if not ok]
    accepted=not failed
    report=dict(accepted=accepted,failed_checks=failed,
                before_locked=before_locked,after_locked=after_locked,
                before_pilot_answers=pilot_locked,after_candidate_answers=candidate_locked,
                answer_nll_before=old_nll,answer_nll_after=new_nll,
                answer_top1_before=old_hits,answer_top1_after=new_hits,
                dev_before=dev_before,dev_after=dev_after,
                best_step=step,foundation_hash_unchanged=True,
                language_guards=ordinary,counterfactual=counterfactual,checks=checks,
                seconds_total=round(time.monotonic()-started,2),
                warning='Experimental frozen-tail answer learning; not full DeepSeek CSA/HCA, not kernel-validated or production certified')
    out=Path(args.report);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(report=report,history=history),indent=2))
    event('phase5d_quality_gate',**report)
    event('phase5d_report_saved',path=str(out))
    if accepted:
        state=dict(attention_state_dict=modules.state_dict(),source_model=MODEL_ID,
                   source_model_sha256=foundation_hash,
                   placement_after_blocks=[csa_pos,hca_pos],
                   max_relative_norm=gain,pilot_parent_sha256=pilotsha,
                   phase5d_report=report,not_production_ready=True)
        dest=Path(args.output);torch.save(state,dest)
        event('phase5d_checkpoint_saved',path=str(dest),bytes=dest.stat().st_size)
    else:
        event('phase5d_checkpoint_rejected',failed_checks=failed,pilot_unchanged=True)
        raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--toy',action='store_true')
    p.add_argument('--pilot',default='/checkpoint/iq_attention_pilot.pt')
    p.add_argument('--pilot-sha',default='f7472337b7773c4c3fc71fe146b0d4519b5b88f9701e79967aa6e889364b6e53')
    p.add_argument('--output',default='/checkpoint/iq_phase5d_attention_candidate.pt')
    p.add_argument('--report',default='/checkpoint/iq_phase5d_retrieval_report.json')
    p.add_argument('--train-cases',type=int,default=8)
    p.add_argument('--dev-cases',type=int,default=4)
    p.add_argument('--test-cases',type=int,default=8)
    p.add_argument('--locked-offset',type=int,default=16)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--lr',type=float,default=0.00005)
    p.add_argument('--selector-weight',type=float,default=0.0)
    p.add_argument('--min-context-gain',type=float,default=0.02)
    p.add_argument('--min-positive-cases',type=int,default=2)
    p.add_argument('--nll-margin',type=float,default=0.005)
    p.add_argument('--kl-budget',type=float,default=0.01)
    p.add_argument('--patience',type=int,default=2)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=47)
    args=p.parse_args()
    if args.toy:toy_run();return
    if args.train_cases<4 or args.dev_cases<2 or args.test_cases<2:
        p.error('Need >=4 train / 2 dev / 2 locked cases')
    if args.locked_offset < args.train_cases+args.dev_cases or args.locked_offset+args.test_cases>24:
        p.error('Locked evaluation must be disjoint from training and development, within 24 keys')
    if args.steps<1 or args.steps>30:p.error('steps must be 1..30')
    if args.selector_weight!=0:p.error('Frozen-selector phase requires --selector-weight 0')
    real_run(args)

if __name__=='__main__':main()
