"""CPU math invariants for actual rank-4 Mamba-3 MIMO.

Closed-form state expansion is independent of the streaming scan; no CUDA,
pretrained weights, tokenizer download or SISO shortcut is used.
"""
from __future__ import annotations

import math
import unittest

import torch
from torch.nn import functional as F

from iq_model.state.mamba3_mimo_cpu_reference import (
    MIMOReferenceState,
    Mamba3MIMOTinyCPU,
    _expand_groups,
    _rotate_mimo,
    mamba3_mimo_cpu_reference,
)


def fixture(*, rank=4, batch=2, steps=9, heads=4, groups=2,
            seed=15, fused_norm=False, dstate=16, headdim=8):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float64)
    dt = F.softplus(-3 + rand(batch,heads,steps))
    negative_a = -F.softplus(rand(batch,heads,steps))
    kw = dict(
        q=rand(batch,steps,rank,groups,dstate),
        k=rand(batch,steps,rank,groups,dstate),
        v=rand(batch,steps,heads,headdim),
        adt=negative_a*dt,
        dt=dt,
        trap=rand(batch,heads,steps),
        q_bias=rand(heads,rank,dstate),
        k_bias=rand(heads,rank,dstate),
        angles=rand(batch,steps,heads,dstate//4),
        mimo_v=rand(heads,rank,headdim)/rank,
        mimo_o=rand(heads,rank,headdim)/rank,
        d=rand(heads),
        z=rand(batch,steps,heads,headdim),
        mimo_z=rand(heads,rank,headdim),
        fused_norm=fused_norm,
    )
    if fused_norm:
        kw["outproj_norm_weight"] = rand(heads,headdim)
    return kw


def closed_form(**kw):
    """Analytic finite-history sum of CURRENT and DELAYED rank-R K/V taps.

    The history sum never updates an SSM recurrent state, so it can expose
    errors in the independent CPU streaming implementation.
    """
    q,k,v = kw["q"],kw["k"],kw["v"]
    batch,t,r,groups,n = q.shape
    heads,p = v.shape[2:]
    dt,adt = kw["dt"],kw["adt"]
    phases = (torch.tanh(kw["angles"])
              * dt.transpose(1,2).unsqueeze(-1)*math.pi).cumsum(dim=1)
    qrot = _rotate_mimo(
        _expand_groups(q.reshape(batch*t,r,groups,n),heads)
            .reshape(batch,t,heads,r,n) + kw["q_bias"][None,None],
        phases.unsqueeze(3),rotary_dim_divisor=4)
    krot = _rotate_mimo(
        _expand_groups(k.reshape(batch*t,r,groups,n),heads)
            .reshape(batch,t,heads,r,n) + kw["k_bias"][None,None],
        phases.unsqueeze(3),rotary_dim_divisor=4)
    projected_v = v.unsqueeze(3)*kw["mimo_v"][None,None]
    outer = torch.einsum("bthrn,bthrp->bthpn",krot,projected_v)
    gamma = dt*torch.sigmoid(kw["trap"])
    beta = dt*(1-torch.sigmoid(kw["trap"]))*adt.exp()
    outputs = []
    final_state = None
    for target in range(t):
        state = torch.zeros_like(outer[:,0])
        for source in range(target+1):
            immediate = gamma[:,:,source] * adt[:,:,source+1:target+1].sum(-1).exp()
            state = state + immediate[:,:,None,None]*outer[:,source]
            if source < target:
                delayed = beta[:,:,source+1] * adt[:,:,source+2:target+1].sum(-1).exp()
                state = state + delayed[:,:,None,None]*outer[:,source]
        final_state = state
        out = torch.einsum("bhpn,bhrn->bhrp",state,qrot[:,target])
        out = out + kw["d"][None,:,None,None]*projected_v[:,target]
        gate = kw["z"][:,target].unsqueeze(2)*kw["mimo_z"][None]
        if kw["fused_norm"]:
            out = out*torch.rsqrt(out.square().mean(-1,keepdim=True)+1e-5)
            out = out*kw["outproj_norm_weight"][None,:,None,:]
        out = out*F.silu(gate)
        outputs.append((out*kw["mimo_o"][None]).sum(2))
    return torch.stack(outputs,dim=1),final_state


class RealMIMOCPUTests(unittest.TestCase):
    def test_rank4_matches_independent_closed_form(self):
        kw = fixture()
        scan,last = mamba3_mimo_cpu_reference(**kw)
        expected,exact_state = closed_form(**kw)
        torch.testing.assert_close(scan,expected,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(last.ssm.double(),exact_state,rtol=2e-6,atol=2e-6)
        self.assertEqual(scan.shape,(2,9,4,8))
        self.assertEqual(last.previous_key.shape,(2,4,4,16))

    def test_rank4_fused_norm_matches_independent_closed_form(self):
        kw = fixture(fused_norm=True,seed=33)
        scan,last = mamba3_mimo_cpu_reference(**kw)
        expected,exact_state = closed_form(**kw)
        torch.testing.assert_close(scan,expected,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(last.ssm.double(),exact_state,rtol=2e-6,atol=2e-6)

    def test_two_step_analytic_all_rank_contributions(self):
        q = torch.zeros((1,2,4,1,4),dtype=torch.float64)
        k = q.clone()
        q[:,:,:,:,0] = 1
        k[:,:,:,:,0] = 1
        v = torch.zeros((1,2,1,4),dtype=torch.float64)
        v[:,:,:,0] = 1
        proj = torch.zeros((1,4,4),dtype=torch.float64)
        proj[:,:,0] = 1
        out,st = mamba3_mimo_cpu_reference(
            q,k,v,
            torch.full((1,1,2),math.log(.5),dtype=torch.float64),
            torch.ones((1,1,2),dtype=torch.float64),
            torch.zeros((1,1,2),dtype=torch.float64),
            torch.zeros((1,4,4),dtype=torch.float64),
            torch.zeros((1,4,4),dtype=torch.float64),
            torch.zeros((1,2,1,1),dtype=torch.float64),
            proj,proj/4,d=torch.zeros((1,),dtype=torch.float64),
        )
        torch.testing.assert_close(
            out[0,:,0,0],torch.tensor([2.,4.],dtype=torch.float64))
        self.assertAlmostEqual(st.ssm[0,0,0,0].item(),4.)

    def test_state_continuation_matches_single_stream(self):
        kw = fixture(steps=13,seed=8)
        whole,final = mamba3_mimo_cpu_reference(**kw)

        def segment(a,b,state=None):
            args = dict(kw)
            for key in ("q","k","v","angles","z"):
                args[key] = kw[key][:,a:b]
            for key in ("adt","dt","trap"):
                args[key] = kw[key][:,:,a:b]
            if state is not None:
                args["state"] = state
            return mamba3_mimo_cpu_reference(**args)

        a,s0 = segment(0,1)
        b,s1 = segment(1,6,s0)
        c,s2 = segment(6,13,s1)
        torch.testing.assert_close(torch.cat((a,b,c),dim=1),whole,rtol=2e-6,atol=2e-6)
        for key in ("angle","ssm","previous_key","previous_value"):
            torch.testing.assert_close(getattr(s2,key),getattr(final,key),rtol=2e-6,atol=2e-6)

    def test_document_boundary_reset_is_not_state_leakage(self):
        kw = fixture(steps=7,batch=1,seed=29)
        whole,_ = mamba3_mimo_cpu_reference(**kw)

        def segment(a,b):
            args = dict(kw)
            for key in ("q","k","v","angles","z"):
                args[key] = kw[key][:,a:b]
            for key in ("adt","dt","trap"):
                args[key] = kw[key][:,:,a:b]
            return mamba3_mimo_cpu_reference(**args)[0]

        restarted = torch.cat((segment(0,3),segment(3,7)),dim=1)
        self.assertGreater((whole-restarted).abs().max().item(),1e-6)
        torch.testing.assert_close(restarted[:,:3],whole[:,:3])

    def test_quarter_rotary_known_pi_over_2(self):
        v = torch.arange(1.,9.,dtype=torch.float64).reshape(1,1,1,8)
        phase = torch.full((1,1,1,2),math.pi/2,dtype=torch.float64)
        actual = _rotate_mimo(v,phase,rotary_dim_divisor=4)
        expected = torch.tensor([-5.,-6.,3.,4.,1.,2.,7.,8.],
                                dtype=torch.float64).reshape(1,1,1,8)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=0)
        torch.testing.assert_close(actual.norm(),v.norm(),atol=1e-12,rtol=0)

    def test_zero_rotary_angle_is_identity(self):
        x = torch.randn(1,4,4,16,dtype=torch.float64)
        phase = torch.zeros(1,4,1,4,dtype=torch.float64)
        torch.testing.assert_close(_rotate_mimo(x,phase,rotary_dim_divisor=4),x)

    def test_mimo_rank4_gradients_finite_and_nonzero(self):
        kw = fixture(steps=4,seed=7)
        for field in ("q","k","mimo_v","mimo_o"):
            kw[field] = kw[field].clone().requires_grad_()
        out,_ = mamba3_mimo_cpu_reference(**kw)
        out.square().mean().backward()
        for field in ("q","k","mimo_v","mimo_o"):
            grad = kw[field].grad
            self.assertIsNotNone(grad)
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertGreater(grad.abs().sum().item(),1e-12)

    def test_fourth_rank_actually_changes_output(self):
        kw = fixture(seed=19)
        base,_ = mamba3_mimo_cpu_reference(**kw)
        kw["mimo_v"] = kw["mimo_v"].clone()
        kw["mimo_v"][:,3] = 0
        reduced,_ = mamba3_mimo_cpu_reference(**kw)
        self.assertGreater((base-reduced).abs().max().item(),1e-6)

    def test_random_tiny_mimo_mixer_continues_across_splits(self):
        torch.manual_seed(23)
        model = Mamba3MIMOTinyCPU(d_model=16,d_state=16,headdim=8,mimo_rank=4,ngroups=2)
        x = torch.randn(2,10,16)
        entire,full_state = model(x)
        prefix,cache = model(x[:,:4])
        suffix,new_state = model(x[:,4:],state=cache)
        torch.testing.assert_close(
            torch.cat((prefix,suffix),dim=1),entire,rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(full_state.ssm,new_state.ssm,rtol=2e-6,atol=2e-6)
        self.assertTrue(bool(torch.isfinite(entire).all()))
        self.assertEqual(entire.shape,x.shape)

    def test_small_synthetic_token_pipeline_forward_backward(self):
        torch.manual_seed(11)
        embed = torch.nn.Embedding(128,16)
        model = Mamba3MIMOTinyCPU(d_model=16,d_state=16,headdim=8,mimo_rank=4)
        head = torch.nn.Linear(16,128,bias=False)
        tokens = torch.randint(0,128,(2,6))
        hidden,_ = model(embed(tokens))
        logits = head(hidden)
        self.assertEqual(logits.shape,(2,6,128))
        loss = F.cross_entropy(logits[:,:-1].reshape(-1,128),tokens[:,1:].reshape(-1))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        params = dict(model.named_parameters())
        for key in ("in_proj.weight","mimo_x","mimo_z","mimo_o","out_proj.weight"):
            grad = params[key].grad
            self.assertIsNotNone(grad,key)
            self.assertTrue(bool(torch.isfinite(grad).all()),key)

    def test_fused_headwise_norm_tiny_mixer(self):
        torch.manual_seed(10)
        model = Mamba3MIMOTinyCPU(
            d_model=16,d_state=16,headdim=8,mimo_rank=4,is_outproj_norm=True)
        data = torch.randn(1,8,16)
        output,state = model(data)
        self.assertEqual(output.shape,data.shape)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool(torch.isfinite(state.ssm).all()))

    def test_projection_layout_matches_upstream_mimo(self):
        m = Mamba3MIMOTinyCPU(d_model=16,d_state=16,headdim=8,mimo_rank=4)
        expected = 2*32+2*16*4+3*4+4
        self.assertEqual(m.in_proj.weight.shape,(expected,16))
        self.assertEqual(m.mimo_x.shape,(4,4,8))
        self.assertEqual(m.C_bias.shape,(4,4,16))
        self.assertEqual(m.out_proj.weight.shape,(16,32))

    def test_reject_rank_one_and_invalid_dt(self):
        with self.assertRaisesRegex(ValueError,"rank"):
            mamba3_mimo_cpu_reference(**fixture(rank=1))
        kw = fixture()
        kw["adt"] = kw["adt"].abs()
        with self.assertRaisesRegex(ValueError,"nonpositive"):
            mamba3_mimo_cpu_reference(**kw)

    def test_invalid_recurrent_cache_is_rejected(self):
        kw = fixture()
        _,state = mamba3_mimo_cpu_reference(**kw)
        kw["state"] = MIMOReferenceState(
            state.angle,state.ssm[:,:1],state.previous_key,state.previous_value)
        with self.assertRaisesRegex(ValueError,"cache shape"):
            mamba3_mimo_cpu_reference(**kw)


if __name__ == "__main__":
    unittest.main()
