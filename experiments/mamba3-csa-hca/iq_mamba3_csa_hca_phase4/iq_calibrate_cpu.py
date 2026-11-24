"""Phase 4 CPU-only attention calibration for intact pretrained Mamba-3 MIMO.

This is a targeted pilot: train newly inserted CSA/HCA-inspired modules,
never the original pretrained parameters. The memory proxy is *not* proof of
retrieval capabilities, and the CPU MIMO recurrence lacks CUDA-kernel parity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from iq_hybrid_cpu import MambaLMCPU, default_insertions, load_cpu_checkpoint, _toy_config


def event(name, **details):
    print(json.dumps({'event':name, **details}, default=str, ensure_ascii=False), flush=True)


def checksum(model):
    """Digest original model parameters (CPU tensors); no conversion/repacking."""
    digest = hashlib.sha256()
    for name, t in model.state_dict().items():
        digest.update(name.encode())
        digest.update(t.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def hidden_taps(model, tokens, positions):
    """Collect exact original (mixer+MLP+residual) outputs at insertion boundaries."""
    with torch.no_grad():
        x = model.backbone.embedding(tokens)
        residual = None
        cache = {}
        for i, block in enumerate(model.backbone.layers, start=1):
            x, residual = block(x, residual)
            if str(i) in positions:
                cache[str(i)] = (x + residual).detach().clone()
    if set(cache) != set(positions):
        raise RuntimeError(f'Missing expected insertion positions: {set(positions)-set(cache)}')
    return cache


def memory_proxy_warmstart(modules, hidden, iterations=4, lr=3e-4):
    """Small bounded memory reconstruction objective on real pretrained activations.

    Targets deliberately use a known completed history pool. This tests that
    compressed memory paths can be trained, NOT semantic retrieval or general LM.
    """
    report={}
    for name, mod in modules.items():
        x = hidden[name]
        ratio = mod.ratio
        if x.shape[1] <= ratio + 1:
            raise ValueError(f'{name}: need length > {ratio+1}, got {x.shape[1]}')
        # Train only on queries after the first complete compressed pool.
        target = x[:, :ratio].mean(dim=1, keepdim=True).detach()
        target_scale = x[:, ratio:].detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt().mean()
        target = F.normalize(target.float(), dim=-1) * (target_scale * 0.03)
        target = target.expand(-1, x.shape[1]-ratio, -1)
        mod.train()
        opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=0.0)
        initial = None
        for i in range(iterations):
            opt.zero_grad(set_to_none=True)
            out = mod(x)
            # Early queries should remain near zero. Later queries learn a
            # small, deterministic memory reconstruction objective.
            loss = F.mse_loss(out[:, ratio:].float(), target) + \
                0.05 * out[:, :ratio].float().square().mean()
            loss.backward()
            if not torch.isfinite(loss): raise FloatingPointError('Warmstart loss became nonfinite')
            nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            opt.step()
            if initial is None: initial=float(loss.detach())
            event('memory_warmstart_step', module=mod.kind, step=i+1, loss=float(loss.detach()))
        q_grad = mod.q_proj.weight.grad
        q_grad_max = float(q_grad.detach().abs().max()) if q_grad is not None else 0.0
        report[mod.kind] = {'initial_loss':initial,'final_loss':float(loss.detach()),
                            'nonzero_output':bool(mod(x).detach().abs().max().item()>0),
                            'q_projection_received_gradient':q_grad_max > 0,
                            'ratio':ratio}
        mod.eval()
    return report


class BoundedInsertion(nn.Module):
    """Limit added hidden delta norm per token, preserving original dimensions."""
    def __init__(self, module, max_relative_norm=0.02):
        super().__init__()
        self.module=module
        self.max_relative_norm=max_relative_norm

    def forward(self,x):
        a=self.module(x)
        xn=x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        an=a.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        c=(self.max_relative_norm * xn / an).clamp(max=1)
        return a * c.to(a.dtype)


def guarded_insertions(modules, max_relative_norm):
    return nn.ModuleDict({name:BoundedInsertion(mod, max_relative_norm) for name,mod in modules.items()})


def calibration_pass(model, modules, tokens, lr=1e-5, steps=1, max_relative_norm=.02):
    """Short-context original-model distillation through frozen downstream blocks.

    A separate original forward computes teacher logits. The training forward
    retains autograd through frozen backbone operations to reach the insertions.
    """
    wrapped=guarded_insertions(modules,max_relative_norm)
    opt=torch.optim.AdamW(modules.parameters(), lr=lr, weight_decay=0.0)
    with torch.no_grad():
        baseline=model(tokens)
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        logits=model(tokens, wrapped)
        teacher=baseline[:, :-1].float().detach()
        student=logits[:, :-1].float()
        tok_targets=tokens[:,1:]
        ce=F.cross_entropy(student.reshape(-1,student.shape[-1]),tok_targets.reshape(-1))
        kd=F.kl_div(F.log_softmax(student,dim=-1), F.softmax(teacher,dim=-1), reduction='batchmean')/student.shape[1]
        loss=ce+2*kd
        if not torch.isfinite(loss): raise FloatingPointError('Language calibration produced nonfinite loss')
        loss.backward()
        grad=nn.utils.clip_grad_norm_(modules.parameters(),0.5)
        opt.step()
        event('language_calibration_step',step=i+1,loss=float(loss.detach()),
              cross_entropy=float(ce.detach()), teacher_kl=float(kd.detach()),
              grad_norm=float(grad.detach()))
    return wrapped


@torch.no_grad()
def quality_probe(model, modules, tokens, max_relative_norm=.02):
    base=model(tokens)[:,-1,:].float()
    identity=default_insertions(model.cfg)
    assert torch.equal(base,model(tokens,identity)[:,-1,:].float()), 'Identity preservation failed'
    active=model(tokens,guarded_insertions(modules,max_relative_norm))[:,-1,:].float()
    base_logp=base.log_softmax(-1)
    active_logp=active.log_softmax(-1)
    kl=(base_logp.exp()*(base_logp-active_logp)).sum(-1).item()
    return {'baseline_token_id':int(base.argmax(-1).item()),
            'active_token_id':int(active.argmax(-1).item()),
            'top1_unchanged':bool(torch.equal(base.argmax(-1),active.argmax(-1))),
            'kl_nats':max(0.,float(kl)),
            'max_abs_logit_delta':float((base-active).abs().max().item()),
            'active_delta_nonzero':bool(torch.any(base!=active).item()),
            'identity_logits_exact':True}


@torch.no_grad()
def generation_probe(model, modules, tokenizer, prompt, steps, max_relative_norm):
    """Compare held-out active autoregressive text, never just a one-step logit."""
    ids=tokenizer.encode(prompt,add_special_tokens=True).ids
    if not ids or max(ids)>=model.cfg['vocab_size']:
        raise ValueError('Invalid held-out tokenizer ids')
    baseline=torch.tensor([ids],dtype=torch.long)
    active=baseline.clone()
    max_kl=0.0
    equal=True
    bounded=guarded_insertions(modules,max_relative_norm)
    for step in range(steps):
        teacher=model(baseline)[:,-1,:].float()
        student=model(active,bounded)[:,-1,:].float()
        tk=int(teacher.argmax(-1).item())
        sk=int(student.argmax(-1).item())
        equal=equal and tk==sk
        if torch.equal(baseline,active):
            q=teacher.log_softmax(-1)
            p=student.log_softmax(-1)
            kld=float((q.exp()*(q-p)).sum().item())
            max_kl=max(max_kl,max(0.,kld))
        baseline=torch.cat((baseline,torch.tensor([[tk]])),dim=1)
        active=torch.cat((active,torch.tensor([[sk]])),dim=1)
    return {'prompt':prompt,'steps':steps,'top1_identical':equal,
            'maximum_teacher_kl':max_kl,
            'baseline_continuation':tokenizer.decode(baseline[0,len(ids):].tolist(),skip_special_tokens=True),
            'hybrid_continuation':tokenizer.decode(active[0,len(ids):].tolist(),skip_special_tokens=True)}


def build_tokens(tokenizer, prompt, length):
    # Strong public mirror consistency checks; not a byte-level proof.
    if tokenizer.get_vocab_size(True)!=128256 or tokenizer.token_to_id('<|begin_of_text|>')!=128000:
        raise ValueError('Tokenizer mapping does not match expected Mamba vocabulary')
    ids=tokenizer.encode(prompt,add_special_tokens=True).ids
    if len(ids)<4 or max(ids)>=128256:raise ValueError('Invalid prompt for original tokenizer')
    if length is not None:
        # Repeat real text with several unique tokens so a 128-token HCA pool exists.
        passage=('Remember this sequence for later: amber cedar delta quartz. '
                 'The compressed memory path should retain earlier information. ')
        extra=tokenizer.encode(passage,add_special_tokens=False).ids
        ids=(ids+extra*((length-len(ids)+len(extra)-1)//len(extra)))[:length]
    return torch.tensor([ids],dtype=torch.long)


def real_run(args):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    torch.set_num_threads(args.threads)
    torch.manual_seed(58)
    model_path=Path(args.checkpoint) if args.checkpoint else Path(hf_hub_download(
        'state-spaces/mamba3-mimo-1.5b',filename='pytorch_model.bin'))
    tokenizer_path=hf_hub_download(args.tokenizer_repo,filename='tokenizer.json')
    tokenizer=Tokenizer.from_file(tokenizer_path)
    model,cfg=load_cpu_checkpoint(model_path)
    # Parameter identity protected by frozen trainable flags and hash.
    before=checksum(model)
    modules=default_insertions(cfg)
    language=build_tokens(tokenizer,'The capital of France is Paris. It is',None)
    long_tokens=build_tokens(tokenizer,'The capital of France is',args.context_tokens)
    event('setup',checkpoint=str(model_path),mamba_layers=len(model.backbone.layers),
          hidden=cfg['d_model'],trainable_base=sum(p.requires_grad for p in model.parameters()),
          trainable_attention=sum(p.numel() for p in modules.parameters()),
          long_context_tokens=long_tokens.shape[1])
    started=time.monotonic()
    initial=quality_probe(model,modules,language,args.max_relative_norm)
    event('before_training',**initial)
    taps=hidden_taps(model,long_tokens,modules.keys())
    report=memory_proxy_warmstart(modules,taps,args.memory_steps,args.memory_lr)
    event('memory_warmstart_complete',modules=report,seconds=round(time.monotonic()-started,2))
    pre_calibration=quality_probe(model,modules,language,args.max_relative_norm)
    event('after_memory_warmstart',**pre_calibration)
    if args.language_steps:
        calibration_pass(model,modules,language,args.language_lr,args.language_steps,args.max_relative_norm)
    after=quality_probe(model,modules,language,args.max_relative_norm)
    after_hash=checksum(model)
    if before != after_hash:raise AssertionError('Original pretrained parameter bytes CHANGED')
    # Holdout generation checks *actual active trajectories* on independent prompts.
    probes=[]
    for prompt in ['The capital of France is', 'The largest planet in the solar system is']:
        probe=generation_probe(model,modules,tokenizer,prompt,3,args.max_relative_norm)
        probes.append(probe)
        event('heldout_generation',**probe)
    # This is still a narrow acceptance gate, not proof of general-language parity.
    memory_learned = all(r['final_loss'] < r['initial_loss']
                         and r['nonzero_output'] and r['q_projection_received_gradient']
                         for r in report.values())
    accepted=(memory_learned and after['top1_unchanged'] and after['kl_nats'] <= args.kl_budget
              and all(p['top1_identical'] and p['maximum_teacher_kl'] <= args.kl_budget for p in probes))
    event('quality_gate',accepted=accepted,memory_proxy_improved=memory_learned,**after,heldout_prompts=len(probes),kl_budget=args.kl_budget,
          foundation_hash_unchanged=True,seconds_total=round(time.monotonic()-started,2))
    if accepted:
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'attention_state_dict':modules.state_dict(),
                    'source_model':'state-spaces/mamba3-mimo-1.5b',
                    'source_model_sha256':before,'placement_after_blocks':[8,16],
                    'csa_ratio':4,'hca_ratio':128,
                    'max_relative_norm':args.max_relative_norm,
                    'pilot_kind':'memory-proxy warmstart + short-language calibration',
                    'trained_not_proven_semantically':True,'acceptance':after}, output)
        event('attention_checkpoint_saved',path=str(output),bytes=output.stat().st_size,
              warning='Pilot only. Does NOT prove general-language parity or retrieval gains.')
    else:
        event('attention_checkpoint_rejected',reason='Top-1 or KL budget failed; original model remains untouched')


def toy_run():
    torch.manual_seed(4)
    torch.set_num_threads(2)
    cfg=_toy_config()
    model=MambaLMCPU(cfg)
    for p in model.parameters():p.requires_grad_(False)
    modules=default_insertions(cfg)
    original_hash=checksum(model)
    tokens=torch.randint(0,cfg['vocab_size'],(1,144))
    taps=hidden_taps(model,tokens,modules.keys())
    warm=memory_proxy_warmstart(modules,taps,iterations=3,lr=.001)
    assert all(x['nonzero_output'] and x['q_projection_received_gradient']
               and x['final_loss'] < x['initial_loss'] for x in warm.values())
    lang=torch.randint(0,cfg['vocab_size'],(1,9))
    check_before=quality_probe(model,default_insertions(cfg),lang)
    assert check_before['identity_logits_exact']
    calibration_pass(model,modules,lang,lr=1e-4,steps=1)
    check_after=quality_probe(model,modules,lang)
    assert check_after['active_delta_nonzero']
    assert all(p.grad is None for p in model.parameters())
    assert checksum(model)==original_hash, 'Frozen foundation altered after training'
    event('toy_pass',memory_paths_trained=True,attention_active=True,
          original_params_frozen=True,identity_baseline_exact=True,
          active_next_token=check_after['active_token_id'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toy',action='store_true')
    parser.add_argument('--checkpoint',default='')
    parser.add_argument('--tokenizer-repo',default='unsloth/Meta-Llama-3.1-8B')
    parser.add_argument('--context-tokens',type=int,default=136)
    parser.add_argument('--memory-steps',type=int,default=3)
    parser.add_argument('--language-steps',type=int,default=1)
    parser.add_argument('--memory-lr',type=float,default=3e-4)
    parser.add_argument('--language-lr',type=float,default=1e-5)
    parser.add_argument('--max-relative-norm',type=float,default=.02)
    parser.add_argument('--kl-budget',type=float,default=.01)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output',default='/scratch/iq_attention_pilot.pt')
    args=parser.parse_args()
    if not 130<=args.context_tokens<=192:parser.error('context-tokens must be 130..192 for HCA 128-token pool')
    if not 0<=args.memory_steps<=25:parser.error('memory-steps must be 0..25')
    if not 0<=args.language_steps<=3:parser.error('language-steps must be 0..3')
    if not 0<args.max_relative_norm<=.1:parser.error('max-relative-norm must be (0,.1]')
    if args.toy:toy_run()
    else:real_run(args)

if __name__=='__main__':main()
