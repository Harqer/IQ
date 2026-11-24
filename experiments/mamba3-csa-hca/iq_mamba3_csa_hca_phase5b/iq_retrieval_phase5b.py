"""IQ Phase 5b: content-aware HCA compression + answer-bearing readout pilot.

CPU-only, preserves 24 pretrained Mamba-3 MIMO blocks and Phase-4 pilot.
This is NOT DeepSeek CSA/HCA nor a validated long-context architecture.
Acceptance requires actual held-out language-model answer improvement, not proxies.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from iq_hybrid_cpu import default_insertions, load_cpu_checkpoint, MambaLMCPU, _toy_config
from iq_calibrate_cpu import checksum, event, guarded_insertions, hidden_taps
from iq_retrieval_phase5 import Example, _plain_ids, answer_probe, plain_language_guard

MODEL_ID='state-spaces/mamba3-mimo-1.5b'
TOKENIZER_ID='unsloth/Meta-Llama-3.1-8B'
WORDS=['amber','cobalt','violet','silver','bronze','indigo','copper','orange',
       'maple','cedar','quartz','green','scarlet','teal','yellow','blue']
PROJECTS=['maple','delta','cedar','atlas','comet','harbor','ember','stone',
          'falcon','willow','meadow','pioneer','aurora','marble','ocean','summit',
          'prairie','saturn','timber','canyon','acorn','violet','quartz','pebble']
DECOYS=['river','orchid','planet','forest','valley','island','garden','canyon']


def make_record(tok,index,split,seed=93):
    valid=[w for w in WORDS if len(_plain_ids(tok,' '+w))==1]
    if len(valid)<10:raise ValueError('Insufficient tokenizer word targets')
    key=PROJECTS[index%len(PROJECTS)]
    answer=valid[(index*5+3)%len(valid)]
    decoy=valid[(index*5+9)%len(valid)]
    needle_at=24 if index%2==0 else 168
    distractor_at=168 if needle_at==24 else 24
    fillers=[_plain_ids(tok,x) for x in (
        ' The log contains a routine observation of the landscape.',
        ' A technician recorded the weather earlier in the day.',
        ' The archive includes unrelated field notes and reports.',
        ' No further changes were entered into the ledger.')]
    if any(not x for x in fillers):raise ValueError('Missing filler tokens')
    rng=random.Random(seed+index*1543)
    prefix=[128000]
    while len(prefix)<270:
        prefix.extend(rng.choice(fillers))
    prefix=prefix[:270]
    needle=f' Project {key} has code {answer}.'
    distractor=f' Project {DECOYS[index%len(DECOYS)]} has code {decoy}.'
    for pos,txt in ((needle_at,needle),(distractor_at,distractor)):
        a=_plain_ids(tok,txt)
        if pos+len(a)>256:raise ValueError('Evidence span outside first 2 complete 128-token pools')
        prefix[pos:pos+len(a)]=a
    question=f' Question: What code belongs to Project {key}? Answer:'
    ids=torch.tensor([prefix+_plain_ids(tok,question)],dtype=torch.long)
    return Example(key,answer,_plain_ids(tok,' '+answer)[0],needle_at,ids,question,needle,split)


def final_query_attention(mod, x):
    """Exact last-token forward operation of updated CompressedAttention.

    Computing just the last query avoids costly all-token gradients during CPU
    adapter training. No future tokens exist at this query by construction.
    """
    b,t,d=x.shape
    z=mod.norm(x)
    h,hd,kh=mod.heads,mod.dim,mod.kv_heads
    q=mod.q_proj(z[:,-1:]).reshape(b,1,h,hd).transpose(1,2)
    k=mod.k_proj(z).reshape(b,t,kh,hd).transpose(1,2).repeat_interleave(h//kh,dim=1)
    v=mod.v_proj(z).reshape(b,t,kh,hd).transpose(1,2).repeat_interleave(h//kh,dim=1)
    lo=max(0,t-mod.window)
    logits_local=(q@k[:,:,lo:].transpose(-1,-2))/math.sqrt(hd)
    nfull=t//mod.ratio
    if nfull:
        kr=k[:,:,:nfull*mod.ratio].reshape(b,h,nfull,mod.ratio,hd)
        vr=v[:,:,:nfull*mod.ratio].reshape(b,h,nfull,mod.ratio,hd)
        if mod.kind=='hca':
            assert hasattr(mod,'pool_score_proj')
            w=mod.pool_score_proj(z[:,:nfull*mod.ratio]).reshape(b,1,nfull,mod.ratio).softmax(-1)
            pk=(kr*w.unsqueeze(-1)).sum(3)
            pv=(vr*w.unsqueeze(-1)).sum(3)
        else:
            pk=kr.mean(3)
            pv=vr.mean(3)
        logits_pool=(q@pk.transpose(-1,-2))/math.sqrt(hd)
        if mod.kind=='csa' and nfull>mod.topk:
            ids=logits_pool.topk(mod.topk,dim=-1).indices
            selected=torch.zeros_like(logits_pool,dtype=torch.bool).scatter_(-1,ids,True)
            logits_pool=logits_pool.masked_fill(~selected,torch.finfo(logits_pool.dtype).min)
        weights=torch.softmax(torch.cat((logits_local,logits_pool),dim=-1),dim=-1)
        wl,wp=weights.split((t-lo,nfull),dim=-1)
        out=wl@v[:,:,lo:]+wp@pv
    else:
        out=torch.softmax(logits_local,dim=-1)@v[:,:,lo:]
    return mod.out_proj(out.transpose(1,2).reshape(b,1,d))


def last_query_pool_scores(mod,x):
    z=mod.norm(x)
    b,t,_=z.shape
    nfull=t//mod.ratio
    if nfull<2:raise ValueError('Need >=2 compression pools to train retrieval selector')
    h,hd,kh=mod.heads,mod.dim,mod.kv_heads
    q=mod.q_proj(z[:,-1:]).reshape(b,1,h,hd).transpose(1,2)
    k=mod.k_proj(z[:,:nfull*mod.ratio]).reshape(b,nfull,mod.ratio,kh,hd)
    if mod.kind=='hca':
        w=mod.pool_score_proj(z[:,:nfull*mod.ratio]).reshape(b,nfull,mod.ratio,1).softmax(2)
        k=(k*w.unsqueeze(-1)).sum(2)
    else:
        k=k.mean(2)
    k=k.permute(0,2,1,3).repeat_interleave(h//kh,dim=1)
    return (q@k.transpose(-1,-2)).squeeze(-2)/math.sqrt(hd)


def pool_ce(mod,x,needle_at):
    s=last_query_pool_scores(mod,x)
    target=needle_at//mod.ratio
    y=torch.full((s.shape[0]*s.shape[1],),target,dtype=torch.long)
    return F.cross_entropy(s.reshape(-1,s.shape[-1]),y)


def selector_eval(modules,data):
    result={}
    with torch.no_grad():
        for pos,mod in modules.items():
            hits=0; losses=[]
            for ex,taps in data:
                x=taps[pos]
                s=last_query_pool_scores(mod,x)
                hits+=int(s.mean(1).argmax(-1).item()==ex.needle_at//mod.ratio)
                losses.append(float(pool_ce(mod,x,ex.needle_at)))
            result[mod.kind]=dict(hits=hits,total=len(data),accuracy=hits/max(1,len(data)),
                                  mean_cross_entropy=sum(losses)/max(1,len(losses)))
    return result


def semantic_ce(mod,x,target_id,target_ids,embeddings,temperature=12.):
    # Auxiliary token-embedding alignment objective, NOT the true LM loss.
    delta=final_query_attention(mod,x)[:,-1].float()
    w=embeddings.float()
    scores=temperature*(F.normalize(delta,dim=-1,eps=1e-5)@
                        F.normalize(w,dim=-1,eps=1e-5).T)
    label=target_ids.index(target_id)
    return F.cross_entropy(scores,torch.tensor([label],dtype=torch.long))


def train_attention(modules,train,dev,model_emb,targets,steps,learning_rate,semantic_weight):
    params=[]
    for mod in modules.values():
        for p in mod.parameters():p.requires_grad_(False)
        for name in ('q_proj','k_proj','v_proj','out_proj'):
            for p in getattr(mod,name).parameters():p.requires_grad_(True);params.append(p)
        if mod.kind=='hca':
            for p in mod.pool_score_proj.parameters():p.requires_grad_(True);params.append(p)
    embedding=model_emb[targets].detach()
    opt=torch.optim.AdamW(params,lr=learning_rate,weight_decay=0.)
    best=None;best_score=float('inf');history=[]
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        losses=[]
        for ex,taps in train:
            for pos,mod in modules.items():
                x=taps[pos]
                losses.append(pool_ce(mod,x,ex.needle_at)+semantic_weight*semantic_ce(
                    mod,x,ex.target_id,targets,embedding))
        loss=torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Non-finite training loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(params,max_norm=1.)
        if not bool(torch.isfinite(grad)):raise FloatingPointError('Non-finite gradients')
        opt.step()
        if step==0 or (step+1)%2==0 or step+1==steps:
            dev_scores=selector_eval(modules,dev)
            # Validation selected only for HCA+CSA selection, locked QA test untouched.
            score=sum(dev_scores[k]['mean_cross_entropy'] for k in ('csa','hca'))
            if score < best_score:
                best_score=score
                best=copy.deepcopy(modules.state_dict())
            log={'step':step+1,'loss':float(loss.detach()),'dev_selection_score':score,
                 'dev_selection':dev_scores,'grad_norm':float(grad)}
            history.append(log);event('phase5b_train_step',**log)
    if best is not None:modules.load_state_dict(best,strict=True)
    for p in modules.parameters():p.requires_grad_(False)
    return history


def load_pilot_modules(cfg,pilot):
    modules=default_insertions(cfg)
    # The only new weight is the optional HCA content-aware compressor.
    info=modules.load_state_dict(pilot['attention_state_dict'],strict=False)
    expected_missing={f'{pos}.pool_score_proj.weight' for pos,m in modules.items() if m.kind=='hca'}
    assert set(info.missing_keys)==expected_missing,info
    assert not info.unexpected_keys,info
    return modules


def toy_run():
    torch.manual_seed(42);torch.set_num_threads(2)
    cfg=_toy_config();model=MambaLMCPU(cfg)
    for p in model.parameters():p.requires_grad_(False)
    mods=default_insertions(cfg)
    x=torch.randn(1,270,cfg['d_model'])
    for pos,mod in mods.items():
        nn.init.normal_(mod.out_proj.weight, std=1e-3)
        mod.eval()
        with torch.no_grad():
            # full and truncated must agree; 128:1 compressor is initialized uniform.
            whole=mod(x)[:,-1]
            small=final_query_attention(mod,x)[:,-1]
            assert torch.allclose(whole,small,atol=2e-5,rtol=2e-4),(mod.kind,(whole-small).abs().max())
        mod.train()
        out=final_query_attention(mod,x)
        (out.square().mean()+pool_ce(mod,x,24)).backward()
        assert mod.q_proj.weight.grad is not None
        if mod.kind=='hca':
            assert mod.pool_score_proj.weight.grad is not None
            assert mod.pool_score_proj.weight.grad.abs().max()>0, 'HCA weighted pool did not learn'
    xx=torch.randint(cfg['vocab_size'],(1,10))
    with torch.no_grad():
        base=model(xx)
        identity=default_insertions(cfg)
        assert torch.equal(base,model(xx,identity))
    # Test the actual two-objective train loop at toy width.
    rows=[]
    for i in range(4):
        ex=Example('key'+str(i),'word'+str(i),2+i,24 if i%2==0 else 168,
                   xx,'question','source','train' if i<3 else 'dev')
        taps={pos:torch.randn(1,270,cfg['d_model']) for pos in mods}
        rows.append((ex,taps))
    history=train_attention(mods,rows[:3],rows[3:],model.lm_head.weight,[2,3,4,5],
                            steps=2,learning_rate=1e-4,semantic_weight=.3)
    assert history and all(math.isfinite(i['loss']) for i in history)
    assert all(not p.requires_grad for p in model.parameters())
    event('phase5b_toy_pass',full_vs_last_query_parity=True,
          hca_compressor_gradients=True,pretrained_params_frozen=True,
          multitask_optimizer_passed=True)


def real_run(args):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    started=time.monotonic()
    pilot_path=Path(args.pilot)
    pilotsha=hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    if pilotsha != args.pilot_sha:raise ValueError('Pilot checkpoint hash mismatch')
    pilot=torch.load(pilot_path,map_location='cpu',weights_only=True)
    if pilot['source_model']!=MODEL_ID or pilot['placement_after_blocks']!=[8,16]:
        raise ValueError('Pilot model or attention positions do not match')
    gain=float(pilot['max_relative_norm'])
    model_path=Path(hf_hub_download(MODEL_ID,filename='pytorch_model.bin'))
    model,cfg=load_cpu_checkpoint(model_path)
    original_hash=checksum(model)
    if pilot['source_model_sha256']!=original_hash:raise ValueError('Original Mamba-3 hash mismatch')
    modules=load_pilot_modules(cfg,pilot).eval()
    tok=Tokenizer.from_file(hf_hub_download(TOKENIZER_ID,filename='tokenizer.json'))
    if tok.get_vocab_size(True)!=128256 or tok.token_to_id('<|begin_of_text|>')!=128000:
        raise ValueError('Tokenizer IDs mismatch')
    train=[make_record(tok,i,'train') for i in range(args.train_cases)]
    dev=[make_record(tok,i+args.train_cases,'dev') for i in range(args.dev_cases)]
    locked=[make_record(tok,i+args.train_cases+args.dev_cases,'locked') for i in range(args.test_cases)]
    event('phase5b_setup',train=len(train),dev=len(dev),locked=len(locked),
          hca_compression='learned_content_weighted_128_to_1',csa_ratio=4,
          checkpoint_sha256=pilotsha,base_param_hash=original_hash)
    def cache(examples, split):
        rows=[]
        for i,ex in enumerate(examples):
            rows.append((ex,hidden_taps(model,ex.ids,modules.keys())))
            event('phase5b_tap_cached',split=split,index=i+1,length=int(ex.ids.shape[1]),needle=ex.needle_at)
        return rows
    train_taps=cache(train,'train');dev_taps=cache(dev,'dev');test_taps=cache(locked,'locked')
    before_train=selector_eval(modules,train_taps)
    before_dev=selector_eval(modules,dev_taps)
    before_test=selector_eval(modules,test_taps)
    pilot_mod=load_pilot_modules(cfg,pilot).eval()
    before_answers=[answer_probe(model,pilot_mod,ex,gain) for ex in locked]
    # Fixed, predetermined candidate vocabulary; never derive training labels
    # or target classes from the locked evaluation examples.
    targets=sorted({_plain_ids(tok,' '+w)[0] for w in WORDS
                    if len(_plain_ids(tok,' '+w))==1})
    history=train_attention(modules,train_taps,dev_taps,model.lm_head.weight,targets,
                            args.steps,args.learning_rate,args.semantic_weight)
    after_train=selector_eval(modules,train_taps)
    after_dev=selector_eval(modules,dev_taps)
    after_test=selector_eval(modules,test_taps)
    after_answers=[answer_probe(model,modules,ex,gain) for ex in locked]
    ordinary=plain_language_guard(model,modules,tok,gain)
    assert checksum(model)==original_hash,'Pretrained Mamba weights changed'
    old_nll=sum(x['hybrid_target_nll'] for x in before_answers)/len(before_answers)
    new_nll=sum(x['hybrid_target_nll'] for x in after_answers)/len(after_answers)
    old_top1=sum(x['hybrid_top1_correct'] for x in before_answers)
    new_top1=sum(x['hybrid_top1_correct'] for x in after_answers)
    improved_hca=(after_test['hca']['mean_cross_entropy']<before_test['hca']['mean_cross_entropy'])
    improved_csa=(after_test['csa']['mean_cross_entropy']<before_test['csa']['mean_cross_entropy'])
    text_guard=all(x['same_top1'] and x['kl_nats']<=args.kl_budget for x in ordinary)
    # A tiny NLL improvement alone does NOT establish functional retrieval.
    qa_gain=(new_top1>old_top1 and new_nll<old_nll-args.nll_margin)
    accepted=bool(improved_hca and improved_csa and text_guard and qa_gain)
    failed=[]
    if not improved_hca:failed.append('hca_locked_selector_not_improved')
    if not improved_csa:failed.append('csa_locked_selector_not_improved')
    if not text_guard:failed.append('ordinary_language_regression')
    if not qa_gain:failed.append('actual_answer_accuracy_or_nll_not_improved')
    report={'accepted':accepted,'failed_checks':failed,
            'before_selector_train':before_train,'after_selector_train':after_train,
            'before_selector_dev':before_dev,'after_selector_dev':after_dev,
            'before_selector_locked':before_test,'after_selector_locked':after_test,
            'before_answers':before_answers,'after_answers':after_answers,
            'answer_nll_before':old_nll,'answer_nll_after':new_nll,
            'answer_top1_before':old_top1,'answer_top1_after':new_top1,
            'language_guards':ordinary,'original_weight_hash_unchanged':True,
            'limitation':'Experimental attention, proxy training and 128:1 learned compressor; not DeepSeek HCA or proof of general retrieval',
            'elapsed_seconds':round(time.monotonic()-started,2)}
    out=Path(args.report);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'report':report,'history':history},indent=2))
    event('phase5b_quality_gate',**report)
    event('phase5b_report_saved',path=str(out))
    if accepted:
        candidate={'attention_state_dict':modules.state_dict(),'source_model':MODEL_ID,
                   'source_model_sha256':original_hash,'placement_after_blocks':[8,16],
                   'csa_ratio':4,'hca_ratio':128,'max_relative_norm':gain,
                   'pilot_parent_sha256':pilotsha,'phase5b_report':report,
                   'new_hca_content_weighted_compression':True,'not_production_ready':True}
        p=Path(args.output);torch.save(candidate,p)
        event('phase5b_checkpoint_saved',path=str(p),size=p.stat().st_size)
    else:
        event('phase5b_checkpoint_rejected',failed_checks=failed,
              note='Phase4d pilot intact. No candidate written.')
        raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--toy',action='store_true')
    p.add_argument('--pilot',default='/checkpoint/iq_attention_pilot.pt')
    p.add_argument('--pilot-sha',default='f7472337b7773c4c3fc71fe146b0d4519b5b88f9701e79967aa6e889364b6e53')
    p.add_argument('--output',default='/checkpoint/iq_phase5b_attention_candidate.pt')
    p.add_argument('--report',default='/checkpoint/iq_phase5b_retrieval_report.json')
    p.add_argument('--train-cases',type=int,default=8)
    p.add_argument('--dev-cases',type=int,default=4)
    p.add_argument('--test-cases',type=int,default=4)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--semantic-weight',type=float,default=0.3)
    p.add_argument('--nll-margin',type=float,default=0.005)
    p.add_argument('--kl-budget',type=float,default=0.01)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=47)
    a=p.parse_args()
    if a.toy:toy_run();return
    if a.train_cases<4 or a.dev_cases<2 or a.test_cases<2:p.error('Need >=4 training, >=2 development, >=2 locked test cases')
    if a.train_cases+a.dev_cases+a.test_cases>len(PROJECTS):p.error('Unique project keys exhausted')
    if not 0<a.steps<=50:p.error('steps must be 1-50')
    real_run(a)

if __name__=='__main__':main()
