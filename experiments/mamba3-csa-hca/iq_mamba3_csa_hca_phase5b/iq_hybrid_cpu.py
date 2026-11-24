"""CPU research prototype for ORIGINAL pretrained Mamba-3 MIMO + two CSA/HCA-inspired residuals.

Not a drop-in reproduction of DeepSeek-V4 CSA/HCA, nor a proven drop-in
replacement for the official TileLang MIMO kernel. Uses the official MIMO
reference recurrent equations and native checkpoint parameter layout.

The base checkpoint is never modified or saved by this script.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def rms(x: torch.Tensor, weight: torch.Tensor, eps=1e-5) -> torch.Tensor:
    y = x.float()
    return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


class RMSNormCPU(nn.Module):
    def __init__(self, dim: int, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return rms(x, self.weight, self.eps)


class GatedMLPCPU(nn.Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        hidden = (intermediate + 127) // 128 * 128
        self.fc1 = nn.Linear(dim, 2*hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        a, gate = self.fc1(x).chunk(2, -1)
        return self.fc2(a * F.silu(gate))


def rotate_half_mimo(u, angle):
    """Split-half rotation in official MIMO fused/step kernels (NOT pairwise RoPE)."""
    half = u.shape[-1] // 2
    n_angles = angle.shape[-1]
    cs, sn = angle.cos(), angle.sin()
    cs = F.pad(cs, (0, half - n_angles), value=1.0).unsqueeze(2)
    sn = F.pad(sn, (0, half - n_angles), value=0.0).unsqueeze(2)
    left, right = u[..., :half], u[..., half:]
    return torch.cat([left * cs - right * sn, left * sn + right * cs], dim=-1)


class Mamba3MIMOCPU(nn.Module):
    """Faithful-in-shape implementation of official Mamba-3 MIMO recurrent equations.

    Always computes in float32 on CPU. Heavy and intended for short baseline
    probes, not speed or production. Cross-kernel parity is NOT yet established.
    """
    def __init__(self, d_model, d_state=128, expand=2, headdim=64,
                 ngroups=1, mimo_rank=4, rope_fraction=.5, **kwargs):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = int(expand * d_model)
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups
        self.mimo_rank = mimo_rank
        self.num_rope_angles = int(d_state * rope_fraction) // 2
        assert self.d_inner % headdim == 0
        assert ngroups == 1, 'CPU reference prototype currently supports one BC group'
        assert d_state % 2 == 0
        in_features = (2*self.d_inner + 2*d_state*ngroups*mimo_rank
                       + 3*self.nheads + self.num_rope_angles)
        self.in_proj = nn.Linear(d_model, in_features, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, mimo_rank, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, mimo_rank, d_state))
        self.B_norm = RMSNormCPU(d_state)
        self.C_norm = RMSNormCPU(d_state)
        self.mimo_x = nn.Parameter(torch.full((self.nheads,mimo_rank,headdim),1/mimo_rank))
        self.mimo_z = nn.Parameter(torch.ones(self.nheads,mimo_rank,headdim))
        self.mimo_o = nn.Parameter(torch.full((self.nheads,mimo_rank,headdim),1/mimo_rank))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        b, t, _ = u.shape
        h, r, p, n = self.nheads, self.mimo_rank, self.headdim, self.d_state
        vals = self.in_proj(u)
        z, x, kb, qc, dt_raw, a_raw, trap_raw, raw_angles = torch.split(
            vals, [self.d_inner, self.d_inner,
                   n*self.num_bc_heads*r, n*self.num_bc_heads*r,
                   h, h, h, self.num_rope_angles], dim=-1)
        z = z.reshape(b,t,h,p)
        x = x.reshape(b,t,h,p)
        kb = self.B_norm(kb.reshape(b,t,r,self.num_bc_heads,n))
        qc = self.C_norm(qc.reshape(b,t,r,self.num_bc_heads,n))
        # Official heavy-tail state-decay activation; A is strictly negative.
        neg, pos = a_raw.clamp_max(0), a_raw.clamp_min(0)
        a = -(pos + 1/(1-neg)).clamp_min(1e-4)
        dt = F.softplus(dt_raw + self.dt_bias)
        alpha = torch.exp(a * dt)
        trap = torch.sigmoid(trap_raw)
        state = u.new_zeros(b,h,p,n)
        prev_k = u.new_zeros(b,h,r,n)
        prev_v = u.new_zeros(b,h,r,p)
        angle_state = u.new_zeros(b,h,self.num_rope_angles)
        outputs=[]
        for i in range(t):
            kt = kb[:,i,:,0,:].unsqueeze(1).expand(-1,h,-1,-1) + self.B_bias.unsqueeze(0)
            qt = qc[:,i,:,0,:].unsqueeze(1).expand(-1,h,-1,-1) + self.C_bias.unsqueeze(0)
            v = x[:,i].unsqueeze(2) * self.mimo_x.unsqueeze(0)
            zg = z[:,i].unsqueeze(2) * self.mimo_z.unsqueeze(0)
            angle_state = angle_state + torch.tanh(raw_angles[:,i].unsqueeze(1))*dt[:,i].unsqueeze(-1)*math.pi
            qrot = rotate_half_mimo(qt, angle_state)
            krot = rotate_half_mimo(kt, angle_state)
            av = alpha[:,i].unsqueeze(-1).unsqueeze(-1)
            beta = ((1-trap[:,i])*dt[:,i]*alpha[:,i]).unsqueeze(-1).unsqueeze(-1)
            gamma = (trap[:,i]*dt[:,i]).unsqueeze(-1).unsqueeze(-1)
            state = (av*state + beta*torch.einsum('bhrn,bhrp->bhpn',prev_k,prev_v)
                     + gamma*torch.einsum('bhrn,bhrp->bhpn',krot,v))
            y = torch.einsum('bhpn,bhrn->bhrp',state,qrot)
            y = y + self.D[None,:,None,None]*v
            y = y * F.silu(zg)
            y = (y*self.mimo_o.unsqueeze(0)).sum(dim=2)
            outputs.append(y.reshape(b,self.d_inner))
            prev_k,prev_v = krot,v
        y = torch.stack(outputs,1)
        return self.out_proj(y)


class MambaBlockCPU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d=cfg['d_model']
        self.norm=RMSNormCPU(d)
        self.mixer=Mamba3MIMOCPU(d, **{k:v for k,v in cfg['ssm_cfg'].items()
                                     if k in {'d_state','expand','headdim','ngroups','mimo_rank','rope_fraction'}})
        self.norm2=RMSNormCPU(d)
        self.mlp=GatedMLPCPU(d,cfg['d_intermediate'])

    def forward(self, x, residual=None):
        residual = x + residual if residual is not None else x
        x=self.mixer(self.norm(residual))
        residual=x+residual
        x=self.mlp(self.norm2(residual))
        return x,residual


class MambaBackboneCPU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d=cfg['d_model']
        self.embedding=nn.Embedding(cfg['vocab_size'],d)
        self.layers=nn.ModuleList(MambaBlockCPU(cfg) for _ in range(cfg['n_layer']))
        self.norm_f=RMSNormCPU(d)


class MambaLMCPU(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.backbone=MambaBackboneCPU(cfg)
        self.lm_head=nn.Linear(cfg['d_model'],cfg['vocab_size'],bias=False)
        if cfg.get('tie_embeddings',True):
            self.lm_head.weight=self.backbone.embedding.weight

    def forward(self, tokens, insertions=None, apply_insertions=True):
        x=self.backbone.embedding(tokens)
        residual=None
        for i,block in enumerate(self.backbone.layers):
            x,residual=block(x,residual)
            if apply_insertions and insertions is not None and str(i+1) in insertions:
                a=insertions[str(i+1)](x+residual)
                # Mamba Block returns unsummed (x, residual), so injecting into
                # the returned x branch gives the exact new residual at next block.
                x=x+a
        x=self.backbone.norm_f(x+residual)
        return self.lm_head(x)


class CompressedAttention(nn.Module):
    """CSA/HCA-inspired demonstrator: compressed-memory attention plus causal local K/V.

    CSA: pool=4, query top-k among causally complete pools.
    HCA: pool=128, attend to all completed pools. Both also have local window.
    This is not claimed to be exact DeepSeek V4 attention implementation.
    """
    def __init__(self, d_model, kind, heads=8, kv_heads=2, head_dim=None,
                 local_window=32, topk=16):
        super().__init__()
        assert kind in ('csa','hca')
        assert heads%kv_heads==0
        self.kind=kind
        self.ratio=4 if kind=='csa' else 128
        self.window=local_window
        self.topk=topk
        self.heads,self.kv_heads=heads,kv_heads
        self.dim=head_dim if head_dim is not None else d_model//heads
        assert heads*self.dim==d_model
        self.norm=RMSNormCPU(d_model)
        self.q_proj=nn.Linear(d_model,heads*self.dim,bias=False)
        self.k_proj=nn.Linear(d_model,kv_heads*self.dim,bias=False)
        self.v_proj=nn.Linear(d_model,kv_heads*self.dim,bias=False)
        self.out_proj=nn.Linear(heads*self.dim,d_model,bias=False)
        # Learned HCA compression: initialized to uniform weights (= old mean).
        # This is an EXPERIMENTAL 128:1 content-aware compression, not DeepSeek HCA.
        if kind == 'hca':
            self.pool_score_proj = nn.Linear(d_model, 1, bias=False)
            nn.init.zeros_(self.pool_score_proj.weight)
        # Identity-preserving initial checkpoint, but trainable output projection.
        nn.init.zeros_(self.out_proj.weight)

    def forward(self,x):
        b,t,d=x.shape
        h=self.heads
        z=self.norm(x)
        q=self.q_proj(z).view(b,t,h,self.dim).transpose(1,2)
        k=self.k_proj(z).view(b,t,self.kv_heads,self.dim).transpose(1,2)
        v=self.v_proj(z).view(b,t,self.kv_heads,self.dim).transpose(1,2)
        k=k.repeat_interleave(h//self.kv_heads,dim=1)
        v=v.repeat_interleave(h//self.kv_heads,dim=1)
        time=torch.arange(t,device=x.device)
        local_allowed=(time[None,:] <= time[:,None]) & ((time[:,None]-time[None,:])<self.window)
        local_scores=(q@k.transpose(-1,-2))/math.sqrt(self.dim)
        local_scores=local_scores.masked_fill(~local_allowed[None,None,:,:],torch.finfo(local_scores.dtype).min)
        nfull=t//self.ratio
        if nfull:
            pk_raw=k[:,:,:nfull*self.ratio].reshape(b,h,nfull,self.ratio,self.dim)
            pv_raw=v[:,:,:nfull*self.ratio].reshape(b,h,nfull,self.ratio,self.dim)
            if self.kind == 'hca':
                pool_logit=self.pool_score_proj(z[:,:nfull*self.ratio]).reshape(b,1,nfull,self.ratio)
                pool_weight=pool_logit.softmax(-1).unsqueeze(-1)
                pk=(pk_raw*pool_weight).sum(3)
                pv=(pv_raw*pool_weight).sum(3)
            else:
                pk=pk_raw.mean(3)
                pv=pv_raw.mean(3)
            scores=(q@pk.transpose(-1,-2))/math.sqrt(self.dim)
            end=(torch.arange(nfull,device=x.device)+1)*self.ratio-1
            allowed=end[None,:]<=time[:,None]
            scores=scores.masked_fill(~allowed[None,None],torch.finfo(scores.dtype).min)
            if self.kind=='csa' and nfull>self.topk:
                # sparse top-k over valid compressed pools for each query
                ids=scores.topk(self.topk,dim=-1).indices
                selected=torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,ids,True)
                scores=scores.masked_fill(~selected,torch.finfo(scores.dtype).min)
            all_scores=torch.cat((local_scores,scores),dim=-1)
            weights=F.softmax(all_scores,dim=-1)
            wl,wm=weights.split((t,nfull),dim=-1)
            out=wl@v+wm@pv
        else:
            out=F.softmax(local_scores,dim=-1)@v
        return self.out_proj(out.transpose(1,2).reshape(b,t,d))


class Insertions(nn.ModuleDict):
    pass


def default_insertions(cfg):
    d=cfg['d_model']
    # Post-block positions preserve the 24 pretrained positions unmodified.
    return Insertions({str(cfg['n_layer']//3):CompressedAttention(d,'csa',heads=8,kv_heads=2),
                       str(2*cfg['n_layer']//3):CompressedAttention(d,'hca',heads=8,kv_heads=2)})


def _toy_config():
    return dict(d_model=64,d_intermediate=128,n_layer=3,vocab_size=128,
                tie_embeddings=True,ssm_cfg=dict(d_state=16,expand=2,headdim=16,
                ngroups=1,rope_fraction=.5,mimo_rank=2))


def smoke_test():
    torch.manual_seed(15)
    torch.set_num_threads(2)
    cfg=_toy_config()
    base=MambaLMCPU(cfg).eval()
    ext=default_insertions(cfg)
    x=torch.randint(0,cfg['vocab_size'],(1,12))
    with torch.no_grad():
        original=base(x)
        identity=base(x,ext)
    assert torch.equal(original,identity), (original-identity).abs().max().item()
    # Deliberately activate attention for causal/gradient checks.
    for layer in ext.values():
        torch.nn.init.normal_(layer.out_proj.weight,mean=0,std=0.001)
    with torch.no_grad():
        active=base(x,ext)
        diff=(active-original).abs().max().item()
        assert diff>0, 'active attention did not affect logits'
        # The prefix representation must not depend on future input IDs.
        variant=x.clone();variant[:,8:]=torch.randint(0,cfg['vocab_size'],(1,4))
        vout=base(variant,ext)
        causal_error=(vout[:,:8]-active[:,:8]).abs().max().item()
        assert causal_error<2e-4, causal_error
    # Frozen base, gradient flows through inserted modules into their weights.
    for p in base.parameters(): p.requires_grad_(False)
    ext.zero_grad()
    obj=F.cross_entropy(base(x,ext)[:,:-1].reshape(-1,cfg['vocab_size']),x[:,1:].reshape(-1))
    obj.backward()
    grads=[p.grad for p in ext.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # Must be possible to start with zero output but get output-proj gradients.
    inactive=default_insertions(cfg)
    obj2=base(x,inactive).square().mean()
    obj2.backward()
    assert any(z.out_proj.weight.grad is not None and z.out_proj.weight.grad.abs().sum()>0 for z in inactive.values())
    print(json.dumps(dict(result='PASS',checkpoint='toy/random',baseline_identity=True,
                          max_nonzero_attention_logit_delta=diff,
                          max_causal_prefix_error=causal_error,
                          finite_attention_gradients=True,
                          initial_identity_attention_trainable=True),indent=2))


def original_config():
    # Official state-spaces/mamba3-mimo-1.5b/config.json
    return dict(d_model=2048,d_intermediate=3824,n_layer=24,vocab_size=128256,
                tie_embeddings=True,ssm_cfg=dict(d_state=128,expand=2,headdim=64,
                ngroups=1,rope_fraction=.5,mimo_rank=4))


def load_cpu_checkpoint(path):
    cfg=original_config()
    model=MambaLMCPU(cfg)
    sd=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    if isinstance(sd,dict) and 'state_dict' in sd: sd=sd['state_dict']
    if not isinstance(sd,dict): raise TypeError('Expected checkpoint state_dict')
    try:
        model.load_state_dict(sd,strict=True)
    except RuntimeError as e:
        raise RuntimeError('Pretrained checkpoint keys/shapes must match EXACTLY. No widening/remapping performed. '+str(e)) from e
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,cfg


def real_test(args):
    path=Path(args.checkpoint)
    if not path.is_file():raise FileNotFoundError(path)
    model,cfg=load_cpu_checkpoint(path)
    attn=default_insertions(cfg).eval()
    # Accept raw token IDs to avoid mistakenly using a different tokenizer.
    tokens=torch.tensor([args.token_ids],dtype=torch.long)
    if not bool(((tokens>=0)&(tokens<cfg['vocab_size'])).all()):
        raise ValueError('Token IDs outside original Mamba-3 vocabulary')
    with torch.inference_mode():
        baseline=model(tokens)
        unchanged=model(tokens,attn)
        assert torch.equal(baseline,unchanged),'Identity bridge failed'
        base_next=baseline[0,-1].argmax().item()
        # Artificial nonzero contribution: interface test only, NOT useful attention.
        for layer in attn.values():nn.init.normal_(layer.out_proj.weight,std=1e-6)
        active=model(tokens,attn)
        active_next=active[0,-1].argmax().item()
    # Short greedy decoding. Both paths recompute the entire prefix; no CUDA
    # kernels or incremental state cache are assumed on CPU.
    generated = tokens.clone()
    identical_at_each_step = True
    predicted = []
    with torch.inference_mode():
        for _ in range(args.greedy_steps):
            base_logits = model(generated)
            # Restore exact identity for the autoregressive comparison;
            # the deliberately randomized attention above was only a wiring test.
            for layer in attn.values():
                layer.out_proj.weight.zero_()
            identity_logits = model(generated, attn)
            identical_at_each_step = identical_at_each_step and torch.equal(base_logits, identity_logits)
            next_id = int(base_logits[0, -1].argmax())
            predicted.append(next_id)
            generated = torch.cat((generated, torch.tensor([[next_id]], dtype=torch.long)), dim=1)
    if not identical_at_each_step:
        raise AssertionError('Original and hybrid identity trajectories diverged')
    decoded_text = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        if len(tok) != cfg['vocab_size']:
            raise RuntimeError(f'Tokenizer size mismatch: {len(tok)} vs {cfg["vocab_size"]}')
        decoded_text = tok.decode(generated[0].tolist(), skip_special_tokens=True)
    print(json.dumps(dict(result='PASS',run='original_checkpoint_cpu_reference',
        same_original_weights=True,identity_logits_exact=True,
        identity_greedy_trajectory_exact=identical_at_each_step,
        active_attention_delta=float((active-baseline).abs().max()),
        baseline_next_id=base_next,active_next_id=active_next,
        greedy_generated_token_ids=predicted,decoded_text=decoded_text,
        warning='CPU reference not yet numerically validated against official CUDA MIMO kernel; active attention untrained'),indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--checkpoint',help='Path to ORIGINAL 1.5B pytorch_model.bin')
    ap.add_argument('--greedy-steps',type=int,default=2,help='Full-prefix greedy decoding steps (slow on CPU)')
    ap.add_argument('--tokenizer',default=None,help='Optional exact Llama 3.1 tokenizer repo/path to decode generated text')
    ap.add_argument('--token-ids',nargs='+',type=int,default=[128000,791,1401,312,279,1665],help='Original Llama 3.1 token IDs; avoid tokenizer mismatch')
    args=ap.parse_args()
    if args.greedy_steps < 0 or args.greedy_steps > 16: ap.error('--greedy-steps must be 0..16')
    if args.smoke:smoke_test()
    elif args.checkpoint:real_test(args)
    else:ap.error('Specify --smoke or --checkpoint /path/to/pytorch_model.bin')


if __name__=='__main__':main()
