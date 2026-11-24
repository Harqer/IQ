"""Synthetic standalone attention tests that force CSA 4x and HCA 128x pools to exist.
These tests do NOT prove language improvement or DeepSeek implementation parity.
"""
import json
import torch
from iq_hybrid_cpu import CompressedAttention


def main():
    torch.manual_seed(47)
    torch.set_num_threads(2)
    results=[]
    for kind in ('csa','hca'):
        module=CompressedAttention(64,kind,heads=8,kv_heads=2,local_window=16,topk=8).eval()
        torch.nn.init.normal_(module.out_proj.weight,std=0.025)
        x=torch.randn(1,264,64)
        y=module(x)
        later=x.clone();later[:,220:]+=torch.randn_like(later[:,220:])
        causal=(module(later)[:,:220]-y[:,:220]).abs().max().item()
        assert causal < 2e-6, (kind,causal)
        # Perturb far-past context outside the local 16-token window; the
        # late output can change only via the completed compressed pools.
        earlier=x.clone();earlier[:,:100]+=torch.randn_like(earlier[:,:100])
        memory_effect=(module(earlier)[:,230:]-y[:,230:]).abs().max().item()
        assert memory_effect>1e-7,(kind,memory_effect)
        results.append({'kind':kind,'ratio':module.ratio,
                        'max_future_leak':causal,'long_range_memory_delta':memory_effect,
                        'memory_path_active':True})
    print(json.dumps({'result':'PASS','note':'synthetic standalone attention tests; not a pretrained hybrid language-quality test','results':results},indent=2))

if __name__=='__main__': main()
