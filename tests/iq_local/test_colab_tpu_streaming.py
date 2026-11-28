"""Small CPU-side parity tests for the new orchestration; TPU not required."""
import torch
from iq_model.mlp import StableLatentMoEConfig
from iq_local.cpu_hybrid import FunctionalStableLatentMoELayer
from scripts.iq235_colab_tpu import moe_hidden, state_to
from iq_model.state.mamba3_cpu import CPUMamba3State

def test_host_routing_matches_reference():
    torch.manual_seed(9)
    cfg = StableLatentMoEConfig(hidden_size=8, latent_size=4,
        expert_intermediate_size=8, num_experts=4, top_k=2,
        num_shared_experts=1, expert_bias=False)
    layer = FunctionalStableLatentMoELayer(cfg).eval()
    experts = [(torch.randn(8,4), torch.randn(8,4), torch.randn(4,8))
               for _ in range(4)]
    def provider(stage, index):
        return experts[index]
    x = torch.randn(1,3,8)
    expected = layer(x, provider).hidden_states
    torch.testing.assert_close(moe_hidden(layer, x, provider), expected)

def test_state_to_preserves_fields_and_dtypes():
    state = CPUMamba3State(torch.randn(2), torch.randn(2),
                           torch.randn(2).bfloat16(), torch.randn(2).bfloat16())
    result = state_to(state, "cpu")
    for name in ("angle", "ssm", "k", "v"):
        torch.testing.assert_close(getattr(state, name), getattr(result, name))


def test_mounted_provider_reads_without_cache(tmp_path):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from scripts.iq235_colab_tpu import MountedStreamingProvider, mounted_checkpoint_root
    import pytest
    for name in ("iq_config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_text("{}")
    assert mounted_checkpoint_root(tmp_path) == tmp_path
    filename = "model-global.safetensors"
    weight = torch.arange(6, dtype=torch.bfloat16)
    save_file({"norm.weight": weight}, str(tmp_path / filename))
    index = SimpleNamespace(weight_map={"norm.weight": filename})
    provider = MountedStreamingProvider(index, tmp_path)
    torch.testing.assert_close(provider.load_global("norm.weight"), weight)
    assert provider.counters["mounted_shard_opens"] == 1
    assert provider.counters["bucket_files"] == 0
    with pytest.raises(FileNotFoundError):
        provider._shard_path("missing.safetensors")


def test_raw_provider_batches_missing_only_and_reuses_disk(tmp_path):
    from types import SimpleNamespace
    from iq_local.cache import LocalCache
    from scripts.iq235_colab_tpu import RawStreamingProvider

    class Client:
        def __init__(self):
            self.batches = []
        def download(self, files):
            self.batches.append([name for name, _ in files])
            for name, dest in files:
                dest.write_bytes(name.encode())

    client = Client()
    index = SimpleNamespace(weight_map={})
    cache = LocalCache(tmp_path, budget_bytes=10000, min_free_bytes=0)
    p = RawStreamingProvider(index, cache, client, prefetch=False)
    p._ensure_shards(["a", "b", "a"])
    p._ensure_shards(["b", "c"])
    assert client.batches == [["a", "b"], ["c"]]
    assert p._shard_path("a").read_bytes() == b"a"
    p.close()
    reopened = RawStreamingProvider(index, LocalCache(tmp_path, 10000, min_free_bytes=0),
                                    client, prefetch=False)
    reopened._ensure_shards(["a", "b", "c"])
    assert len(client.batches) == 2
    reopened.close()
