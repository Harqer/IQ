#!/usr/bin/env python3
"""Safely install the opt-in INT4 startup branch into scripts/serve_iq_local.py.

Run at repo root. Keeps a timestamped backup and never changes the default path.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

path = Path('scripts/serve_iq_local.py')
if not path.is_file() or not Path('iq_local/int4_loading.py').exists():
    raise SystemExit('Run from ~/IQ after extracting the trial archive')
src = path.read_text()
if '--cpu-int4-base' in src:
    raise SystemExit('INT4 trial is already installed; refusing to apply twice')
old_args = '    parser.add_argument("--tokenizer", default="unsloth/Meta-Llama-3.1-8B")\n'
new_args = old_args + '    parser.add_argument("--cpu-int4-base", action="store_true", help="EXPERIMENTAL: meta + CPU INT4 context/shared-MoE loading")\n' + '    parser.add_argument("--cpu-int4-group-size", type=int, default=32)\n'
if src.count(old_args) != 1:
    raise SystemExit('Argument context differs; no source modified')
old_body = '''    model = CPUHybridForCausalLM(config, dtype=torch.bfloat16, device="cpu")
    global_keys = [key for key in index.weight_map if not key.startswith("layers.")]
    state: dict[str, torch.Tensor] = {}
    for key in global_keys:
        state[key] = provider.load_global(key)
    for layer_idx in range(len(config.schedule.layers)):
        state.update(provider.load_layer_base(layer_idx))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint state mismatch: missing={sorted(missing)[:4]}... "
            f"unexpected={sorted(unexpected)[:4]}..."
        )
    print(f"IQ: base weights loaded ({len(state)} tensors)", flush=True)
    model.eval()
'''
new_body = '''    if args.cpu_int4_base:
        from iq_local.int4_loading import build_cpu_int4_model
        model, stats = build_cpu_int4_model(
            config, index, provider, group_size=args.cpu_int4_group_size
        )
        print(f"IQ: CPU INT4 experimental base loaded: {stats}", flush=True)
    else:
        model = CPUHybridForCausalLM(config, dtype=torch.bfloat16, device="cpu")
        global_keys = [key for key in index.weight_map if not key.startswith("layers.")]
        state: dict[str, torch.Tensor] = {}
        for key in global_keys:
            state[key] = provider.load_global(key)
        for layer_idx in range(len(config.schedule.layers)):
            state.update(provider.load_layer_base(layer_idx))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint state mismatch: missing={sorted(missing)[:4]}... "
                f"unexpected={sorted(unexpected)[:4]}..."
            )
        print(f"IQ: base weights loaded ({len(state)} tensors)", flush=True)
        model.eval()
'''
if src.count(old_body) != 1:
    raise SystemExit('Startup context differs; no source modified')
new_src = src.replace(old_args, new_args).replace(old_body, new_body)
ast.parse(new_src)
backup = path.with_name(path.name + '.before-int4-' + datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S'))
backup.write_text(src)
path.write_text(new_src)
print(f'Installed experimental flag. Backup: {backup}')
print('Default serving path remains unchanged.')
