"""Phase 4 CPU-only attention calibration for intact pretrained Mamba-3 MIMO.

This is a targeted pilot: train newly inserted CSA/HCA-inspired modules,
never the original pretrained parameters. The memory proxy is *not* proof of
retrieval capabilities, and the CPU MIMO recurrence lacks CUDA-kernel parity.
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


def memory_objective(mod, x, target, ratio, target_rms):
    """Scale-free proxy loss; zero output corresponds to roughly unit loss.

    This is an activation-memory reconstruction proxy, NOT an actual
    long-context QA benchmark. Loss is evaluated *after* each update.
    """
    out = mod(x).float()
    signal = F.mse_loss(out[:, ratio:], target)
    quiet = 0.05 * out[:, :ratio].square().mean()
    return (signal + quiet) / target_rms.square().clamp_min(1e-12)


def proxy_inputs(mod, x):
    """The same limited prefix-reconstruction diagnostic at every stage."""
    ratio = mod.ratio
    target = x[:, :ratio].mean(dim=1, keepdim=True).detach()
    magnitude = x[:, ratio:].float().square().mean(dim=-1).sqrt().mean()
    target = F.normalize(target.float(), dim=-1) * (magnitude * 0.03)
    target = target.expand(-1, x.shape[1] - ratio, -1).contiguous()
    target_rms = target.square().mean().sqrt().detach().clamp_min(1e-5)
    return target, target_rms


def memory_proxy_recheck(modules, hidden, max_relative_norm=None):
    """Measure the SAME proxy, optionally through the *deployed* norm limiter."""
    report = {}
    with torch.no_grad():
        for name, mod in modules.items():
            x = hidden[name]
            target, scale = proxy_inputs(mod, x)
            active = (BoundedInsertion(mod, max_relative_norm) if max_relative_norm is not None else mod)
            report[mod.kind] = float(memory_objective(active, x, target, mod.ratio, scale))
    return report


def _exact_output_direction_step(mod, x, target, target_rms):
    """Line-search the new zero-output projection (quadratic in its weights).

    AdamW takes a parameter-scale step even when the target is extremely small.
    Here the step size is computed from the ACTUAL loss along the steepest
    output-projection direction. No pretrained parameter is affected.
    """
    feature = {}
    def save_feature(module, inputs):
        feature['x'] = inputs[0].detach()
    hook = mod.out_proj.register_forward_pre_hook(save_feature)
    mod.zero_grad(set_to_none=True)
    try:
        objective = memory_objective(mod, x, target, mod.ratio, target_rms)
        objective.backward()
    finally:
        hook.remove()
    gradient = mod.out_proj.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or float(gradient.norm()) <= 0:
        return False, 'out_projection_zero_or_nonfinite_gradient'
    with torch.no_grad():
        direction = -gradient.detach()
        current = F.linear(feature['x'], mod.out_proj.weight).float()
        delta = F.linear(feature['x'], direction).float()
        r = mod.ratio
        residual = current[:, r:] - target
        numerator = (residual * delta[:, r:]).mean() + 0.05 * (current[:, :r] * delta[:, :r]).mean()
        denominator = delta[:, r:].square().mean() + 0.05 * delta[:, :r].square().mean()
        if not torch.isfinite(denominator) or denominator <= 0:
            return False, 'out_projection_flat_direction'
        alpha = float((-numerator / denominator).clamp(min=0, max=1e12))
        norm = float(direction.norm()) * alpha
        if norm > 2.0:
            alpha *= 2.0 / norm  # conservative trust region on the new projection
        baseline = float(memory_objective(mod, x, target, target_rms=target_rms, ratio=r))
        mod.out_proj.weight.add_(direction, alpha=alpha)
        candidate = float(memory_objective(mod, x, target, target_rms=target_rms, ratio=r))
        if math.isfinite(candidate) and candidate < baseline - 1e-8:
            return True, candidate
        mod.out_proj.weight.add_(direction, alpha=-alpha)
        return False, 'output_line_search_no_improvement'


def memory_proxy_warmstart(modules, hidden, iterations=8, lr=3e-4):
    """Projection line search followed by monotonic, norm-limited SGD.

    Never accepts a memory-proxy regression. A failed update restores the
    exact pre-step parameters. This remains an activation proxy, not semantic
    retrieval training or evidence of generalized long-context ability.
    """
    report = {}
    for name, mod in modules.items():
        x = hidden[name].detach()
        ratio = mod.ratio
        if x.shape[1] <= ratio + 1:
            raise ValueError(f'{name}: need length > {ratio+1}, got {x.shape[1]}')
        target, target_rms = proxy_inputs(mod, x)
        mod.train()
        with torch.no_grad():
            initial = float(memory_objective(mod, x, target, ratio, target_rms))
        accepted_steps, rejected_trials = 0, 0
        q_grad_observed = False
        ok, value = _exact_output_direction_step(mod, x, target, target_rms)
        best = float(value) if ok else initial
        if ok:
            accepted_steps += 1
            event('memory_projection_line_search', module=mod.kind,
                  initial_loss=initial, final_loss=best)
        else:
            event('memory_projection_line_search_failed', module=mod.kind, reason=value)
        # With the output projection no longer zero, gradients can reach Q/K/V.
        # SGD directions are normalized and accepted only on a true loss reduction.
        for i in range(max(0, iterations - 1)):
            mod.zero_grad(set_to_none=True)
            loss = memory_objective(mod, x, target, ratio, target_rms)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{mod.kind}: nonfinite proxy loss')
            loss.backward()
            q_grad = mod.q_proj.weight.grad
            if q_grad is not None and torch.isfinite(q_grad).all() and bool(q_grad.abs().max() > 0):
                q_grad_observed = True
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                   for p in mod.parameters()):
                raise FloatingPointError(f'{mod.kind}: nonfinite optimizer gradients')
            grads = [(p, p.grad.detach().clone()) for p in mod.parameters()
                     if p.grad is not None]
            norm = math.sqrt(sum(float(g.square().sum()) for _,g in grads))
            if not math.isfinite(norm) or norm < 1e-20:
                event('memory_warmstart_stopped', module=mod.kind, reason='flat_gradients')
                break
            trial = min(0.02, max(1e-5, lr * 50))
            success = False
            for attempt in range(14):
                with torch.no_grad():
                    for p,g in grads: p.add_(g, alpha=-trial/norm)
                    candidate = float(memory_objective(mod, x, target, ratio, target_rms))
                if math.isfinite(candidate) and candidate < best - 1e-9:
                    best = candidate
                    accepted_steps += 1
                    event('memory_warmstart_step', module=mod.kind, step=i+2,
                          loss=candidate, step_norm=trial,
                          rejected_trials=rejected_trials)
                    success = True
                    break
                with torch.no_grad():
                    for p,g in grads: p.add_(g, alpha=trial/norm)
                trial *= 0.5
                rejected_trials += 1
            if not success:
                event('memory_warmstart_stopped', module=mod.kind,
                      reason='no_improving_trust_region_step', loss=best)
                break
        with torch.no_grad():
            final = float(memory_objective(mod, x, target, ratio, target_rms))
            nonzero = bool(mod(x).abs().max() > 0)
        if not math.isfinite(final):
            raise FloatingPointError(f'{mod.kind}: nonfinite final memory proxy')
        report[mod.kind] = dict(initial_loss=initial, final_loss=final,
            improvement_pct=100*(initial-final)/max(initial,1e-12),
            accepted_steps=accepted_steps,rejected_trials=rejected_trials,
            nonzero_output=nonzero, q_projection_received_gradient=q_grad_observed,
            ratio=ratio,note='Diagnostic proxy only; not retrieval QA evidence')
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


@torch.no_grad()
def select_residual_scale(model, modules, tokenizer, training_tokens, budget, maximum):
    """Calibrate ONLY the bounded residual gain, never rewrite trained memory weights.

    Freeze new attention weights after the proxy warmstart. Choose the
    largest nonzero contribution satisfying the small heldout KL/top-1 gates.
    The selected scale must ALSO preserve proxy improvement once clipped.
    """
    prompt_tokens = [training_tokens]
    for prompt in ('The capital of France is',
                   'The largest planet in the solar system is'):
        prompt_tokens.append(build_tokens(tokenizer, prompt, None))
    base = [model(tokens)[:,-1,:].float().detach() for tokens in prompt_tokens]
    candidates = [maximum * f for f in (1., .75, .5, .375, .25, .125, .0625)]
    chosen = None
    for scale in candidates:
        worst_kl=0.0
        all_top1=True
        any_delta=False
        for tokens,teacher in zip(prompt_tokens,base):
            student=model(tokens,guarded_insertions(modules,scale))[:,-1,:].float()
            logp=teacher.log_softmax(-1)
            kl=max(0.,float((logp.exp()*(logp-student.log_softmax(-1))).sum()))
            worst_kl=max(worst_kl,kl)
            all_top1=all_top1 and bool(torch.equal(teacher.argmax(-1),student.argmax(-1)))
            any_delta=any_delta or bool(torch.any(teacher!=student))
        event('gain_calibration_trial',max_relative_norm=scale,
              worst_teacher_kl=worst_kl,top1_identical=all_top1,
              attention_nonzero=any_delta)
        if worst_kl<=budget and all_top1 and any_delta:
            chosen=scale
            break
    return chosen


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
    # Crucial distinction: do NOT take an unconstrained language optimizer step
    # through the same attention weights that were just trained for memory.
    # Select only the bounded inference contribution, leaving proxy weights intact.
    selected_scale=select_residual_scale(model,modules,tokenizer,language,
                                         args.kl_budget,args.max_relative_norm)
    gain_found = selected_scale is not None
    if selected_scale is None:
        event('gain_calibration_failed', reason='no_nonzero_scale_satisfies_KL_and_top1')
        selected_scale=args.max_relative_norm/64.
    else:
        event('gain_calibration_selected',max_relative_norm=selected_scale,
              attention_weights_modified=False)
    after=quality_probe(model,modules,language,selected_scale)
    retained_memory_losses=memory_proxy_recheck(modules,taps)
    effective_memory_losses=memory_proxy_recheck(modules,taps,selected_scale)
    event('post_calibration_memory_check', raw_losses=retained_memory_losses,
          deployed_bounded_losses=effective_memory_losses)
    after_hash=checksum(model)
    if before != after_hash:raise AssertionError('Original pretrained parameter bytes CHANGED')
    # Holdout generation checks *actual active trajectories* on independent prompts.
    probes=[]
    for prompt in ['The capital of France is', 'The largest planet in the solar system is']:
        probe=generation_probe(model,modules,tokenizer,prompt,3,selected_scale)
        probes.append(probe)
        event('heldout_generation',**probe)
    # This is still a narrow acceptance gate, not proof of general-language parity.
    memory_learned = all(r['improvement_pct'] >= 1.0
                         and retained_memory_losses[k] < 0.99 * r['initial_loss']
                         and effective_memory_losses[k] < 0.99 * r['initial_loss']
                         and r['nonzero_output'] and r['q_projection_received_gradient']
                         for k, r in report.items())
    failed_checks = []
    if not memory_learned: failed_checks.append('memory_proxy_not_improved_or_not_retained')
    if not gain_found: failed_checks.append('no_safe_nonzero_gain')
    if not after['top1_unchanged']: failed_checks.append('calibration_top1_changed')
    if after['kl_nats'] > args.kl_budget: failed_checks.append('calibration_kl_exceeded')
    if any(not p['top1_identical'] for p in probes): failed_checks.append('heldout_top1_changed')
    if any(p['maximum_teacher_kl'] > args.kl_budget for p in probes): failed_checks.append('heldout_kl_exceeded')
    accepted = not failed_checks
    event('quality_gate',accepted=accepted,failed_checks=failed_checks,memory_proxy_improved=memory_learned,**after,heldout_prompts=len(probes),selected_max_relative_norm=selected_scale,kl_budget=args.kl_budget,
          foundation_hash_unchanged=True,seconds_total=round(time.monotonic()-started,2))
    if accepted:
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'attention_state_dict':modules.state_dict(),
                    'post_calibration_memory_proxy':retained_memory_losses,
                    'deployed_bounded_memory_proxy':effective_memory_losses,
                    'source_model':'state-spaces/mamba3-mimo-1.5b',
                    'source_model_sha256':before,'placement_after_blocks':[8,16],
                    'csa_ratio':4,'hca_ratio':128,
                    'max_relative_norm':selected_scale,
                    'pilot_kind':'memory-proxy line-search warmstart + bounded gain selection',
                    'trained_not_proven_semantically':True,'acceptance':after}, output)
        event('attention_checkpoint_saved',path=str(output),bytes=output.stat().st_size,
              warning='Pilot only. Does NOT prove general-language parity or retrieval gains.')
    else:
        event('attention_checkpoint_rejected',failed_checks=failed_checks, reason='See failed_checks; original model remains untouched')
        raise SystemExit(2)


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
    warm=memory_proxy_warmstart(modules,taps,iterations=5,lr=.0003)
    assert all(x['nonzero_output'] and x['q_projection_received_gradient']
               and x['improvement_pct'] >= 1.0 for x in warm.values()), warm
    lang=torch.randint(0,cfg['vocab_size'],(1,9))
    check_before=quality_probe(model,default_insertions(cfg),lang)
    assert check_before['identity_logits_exact']
    # Toy check: a small bounded active branch changes the output without
    # training any frozen foundation parameter.
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
    parser.add_argument('--memory-steps',type=int,default=8)
    parser.add_argument('--language-steps',type=int,default=0,
                        help='Legacy CLI flag; phase 4c calibrates residual gain without language optimizer steps')
    parser.add_argument('--memory-lr',type=float,default=3e-4)
    parser.add_argument('--language-lr',type=float,default=1e-5)
    parser.add_argument('--max-relative-norm',type=float,default=.02)
    parser.add_argument('--kl-budget',type=float,default=.01)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output',default='/scratch/iq_attention_pilot.pt')
    args=parser.parse_args()
    if not 130<=args.context_tokens<=192:parser.error('context-tokens must be 130..192 for HCA 128-token pool')
    if not 0<=args.memory_steps<=25:parser.error('memory-steps must be 0..25')
    if args.language_steps:parser.error('Phase 4c disables language optimizer steps; use --language-steps 0')
    if not 0<args.max_relative_norm<=.1:parser.error('max-relative-norm must be (0,.1]')
    if args.toy:toy_run()
    else:real_run(args)

if __name__=='__main__':main()
