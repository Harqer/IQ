#!/usr/bin/env python3
"""REAL checkpoint acceptance test for IQ full 235-layer CPU inference.

Does not mock modules, skip layers, substitute weights, or claim model quality.
Runs from ~/IQ against the configured Hugging Face checkpoint and reports
both positive evidence and unverified paths.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iq_local.bucket import BUCKET_ID, RUN_PREFIX, BucketClient
from iq_local.cpu_hybrid import CPUMamba3ResidualLayer
from iq_local.cache import LocalCache, checkpoint_cache_root
from iq_local.index import RunIndex
from iq_local.provider import StreamingWeightProvider


def gib(value: float) -> float:
    return round(value / (1 << 30), 3)


def physical_available_bytes() -> int | None:
    p = Path('/proc/meminfo')
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return None


def cgroup_headroom_bytes() -> int | None:
    # cgroup v2; avoid treating the physical host's total RAM as the pod's RAM.
    root = Path('/sys/fs/cgroup')
    cap, current = root / 'memory.max', root / 'memory.current'
    if cap.exists() and current.exists():
        m = cap.read_text().strip()
        if m != 'max':
            return max(0, int(m) - int(current.read_text().strip()))
    return None


def cgroup_peak_bytes() -> int | None:
    path = Path('/sys/fs/cgroup/memory.peak')
    return int(path.read_text().strip()) if path.exists() else None


def rss_bytes() -> int:
    p = Path('/proc/self/status')
    for line in p.read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    # Linux ru_maxrss is KiB, but this is a fallback only.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class MemorySampler:
    def __init__(self):
        self.stop = threading.Event()
        self.max_rss = rss_bytes()
        self.worker = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(0.25):
            try:
                self.max_rss = max(self.max_rss, rss_bytes())
            except OSError:
                pass

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.worker.join(timeout=2)
        self.max_rss = max(self.max_rss, rss_bytes())


def emit(label: str, data: dict):
    # JSONL captures meaningful progress even when later model load fails/OOMs.
    print(json.dumps({'event': label, 'time_unix': time.time(), **data},
                     ensure_ascii=False, default=str), flush=True)


def make_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, default=Path.home() / '.iq-local-cache')
    p.add_argument('--cache-budget-gb', type=float, default=65)
    p.add_argument('--max-ram-experts', type=int, default=4)
    p.add_argument('--bucket-id', default=BUCKET_ID)
    p.add_argument('--run-prefix', default=RUN_PREFIX)
    p.add_argument('--tokenizer', default='unsloth/Meta-Llama-3.1-8B')
    p.add_argument('--cpu-int4-group-size', type=int, default=32)
    p.add_argument('--prompt', default='What is two plus two?')
    p.add_argument('--max-new-tokens', type=int, default=8)
    p.add_argument('--expect-substring', default=None, help='Optional content acceptance check for decoded output')
    p.add_argument('--max-rss-gib', type=float, default=30.0, help='Maximum allowed peak RSS for success')
    p.add_argument('--require-free-gib', type=float, default=25,
                   help='Refuse full loading when currently available RAM or cgroup headroom is below this threshold')
    p.add_argument('--force-low-memory', action='store_true',
                   help='Override startup memory guard; may cause OS/pod OOM termination')
    p.add_argument('--manifest-only', action='store_true',
                   help='Validate only the real remote checkpoint manifest and schedule; do not load tensors')
    p.add_argument('--output', type=Path, default=Path('iq_full_inference_report.json'))
    return p.parse_args()


def attach_layer_execution_probes(layers, hits: Counter):
    """Count real residual executions, including direct Mamba prefill/step.

    The streaming runner intentionally bypasses Mamba residual.forward,
    so normal module forward hooks would falsely report skipped layers.
    All probes are removable and only count successfully returned calls.
    """
    handles = []
    patched_methods = []
    for layer_id, layer in enumerate(layers):
        if isinstance(layer, CPUMamba3ResidualLayer):
            for method_name in ("prefill", "step"):
                original = getattr(layer.mamba, method_name)

                def counted(*args, _original=original, _idx=layer_id, **kwargs):
                    output = _original(*args, **kwargs)
                    hits[_idx] += 1
                    return output

                setattr(layer.mamba, method_name, counted)
                patched_methods.append((layer.mamba, method_name, original))
        else:
            def on_forward(mod, inp, out, idx=layer_id):
                hits[idx] += 1

            handles.append(layer.register_forward_hook(on_forward))
    return handles, patched_methods


def main():
    args = make_args()
    if not 1 <= args.max_new_tokens <= 64:
        raise ValueError('max-new-tokens must be 1..64')
    if args.cpu_int4_group_size < 2 or args.cpu_int4_group_size % 2:
        raise ValueError('INT4 group size must be positive even number')
    begin = time.monotonic()
    report = {'checkpoint': args.run_prefix, 'bucket': args.bucket_id,
              'architecture_unmodified': True, 'mock_inference': False,
              'status': 'STARTED', 'finished': False}
    try:
        emit('start', {'mode': 'manifest' if args.manifest_only else 'full',
                       'checkpoint': args.run_prefix})
        client = BucketClient(args.bucket_id, args.run_prefix)
        index = RunIndex(client, checkpoint_cache_root(args.cache_dir, args.bucket_id, args.run_prefix) / 'run')
        config = index.config
        schedule = [getattr(layer, 'value', str(layer)) for layer in config.schedule.layers]
        type_counts = dict(Counter(schedule))
        assert len(schedule) == 235, f'checkpoint has {len(schedule)} layers, expected 235'
        assert len(index.weight_map) > 1000, 'checkpoint index unexpectedly small'
        report.update(schedule_counts=type_counts, schedule_length=len(schedule),
                      weight_map_keys=len(index.weight_map))
        emit('manifest_ok', {'schedule_length': 235, 'types': type_counts,
                             'weight_map_keys': len(index.weight_map)})
        if args.manifest_only:
            report.update(status='MANIFEST_PASSED_NOT_INFERRED', finished=True)
            return

        physical = physical_available_bytes()
        quota = cgroup_headroom_bytes()
        headrooms = [x for x in (physical, quota) if x is not None]
        headroom = min(headrooms) if headrooms else None
        report['startup_headroom_gib'] = gib(headroom) if headroom is not None else None
        emit('memory_preflight', {'physical_available_gib': gib(physical) if physical else None,
                                  'cgroup_headroom_gib': gib(quota) if quota else None,
                                  'effective_headroom_gib': report['startup_headroom_gib']})
        if headroom is None and not args.force_low_memory:
            raise RuntimeError('Cannot determine available RAM; use --force-low-memory to override')
        if headroom is not None and headroom < args.require_free_gib * (1 << 30) and not args.force_low_memory:
            raise RuntimeError(f'Full model loading refused: {gib(headroom)} GiB available '
                               f'< required {args.require_free_gib} GiB. '
                               'Use a >=30 GiB pod, or explicitly override with --force-low-memory.')
        # Actual model and real checkpoint; no mock or substitution allowed.
        from scripts.serve_iq_local import build_stack
        args.cpu_int4_base = True
        args.host = '127.0.0.1'
        args.port = 8000
        expert_reads = Counter()
        original_load_expert = StreamingWeightProvider.load_expert
        def counted_load_expert(provider_self, *a, **kw):
            result = original_load_expert(provider_self, *a, **kw)
            stage = a[0] if len(a) >= 1 else kw['stage']
            expert_idx = a[1] if len(a) >= 2 else kw['expert_idx']
            expert_reads[(int(stage), int(expert_idx))] += 1
            return result
        StreamingWeightProvider.load_expert = counted_load_expert
        try:
            with MemorySampler() as sample:
                started_load = time.monotonic()
                runner, tokenizer = build_stack(args)
                model = runner.model
                if len(model.layers) != 235:
                    raise AssertionError('Loaded model does not contain all 235 layers')
                meta_parameters = [n for n, p in model.named_parameters() if p.is_meta]
                meta_buffers = [n for n, b in model.named_buffers() if b.is_meta]
                if meta_parameters or meta_buffers:
                    raise AssertionError('Model retains meta tensors: ' + repr((meta_parameters+meta_buffers)[:6]))
                report['load_seconds'] = round(time.monotonic() - started_load, 3)
                report['rss_after_load_gib'] = gib(rss_bytes())
                emit('model_loaded', {'seconds': report['load_seconds'],
                                      'rss_gib': report['rss_after_load_gib']})

                # Instrument the entry points the streaming runner actually executes.
                hits = Counter()
                handles, patched_mamba = attach_layer_execution_probes(model.layers, hits)
                extra_hits = Counter()
                if model.reasoning is not None:
                    handles.append(model.reasoning.register_forward_hook(
                        lambda mod, inp, out: extra_hits.update(['reasoning'])))
                def on_lm_head(mod, inp, out):
                    extra_hits['lm_head'] += 1
                    if not torch.isfinite(out).all().item():
                        raise AssertionError('LM head produced non-finite logits')
                handles.append(model.lm_head.register_forward_hook(on_lm_head))
                # Block AttnRes has init/read/write methods rather than module.forward.
                # Observe those methods without changing their math.
                attnres = model.attnres
                original_methods = {}
                if attnres is not None:
                    for method in ('init_state', 'read', 'write'):
                        if hasattr(attnres, method):
                            orig = getattr(attnres, method)
                            original_methods[method] = orig
                            def wrapped(*a, _original=orig, _name=method, **kw):
                                extra_hits['attnres.' + _name] += 1
                                return _original(*a, **kw)
                            setattr(attnres, method, wrapped)

                try:
                    messages = [{'role': 'user', 'content': args.prompt}]
                    try:
                        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                            return_tensors='pt')
                        if isinstance(ids, dict):
                            ids = ids['input_ids']
                    except (ValueError, TypeError, AttributeError, KeyError):
                        ids = tokenizer('user: ' + args.prompt + '\nassistant:',
                                        return_tensors='pt')['input_ids']
                    if not isinstance(ids, torch.Tensor):
                        raise TypeError('Tokenizer did not return tensor input IDs')
                    prompt_ids = ids.reshape(-1).tolist()
                    if not prompt_ids:
                        raise AssertionError('empty tokenizer output')
                    eos = set()
                    if tokenizer.eos_token_id is not None:
                        eos.add(int(tokenizer.eos_token_id))
                    eot = tokenizer.convert_tokens_to_ids('<|eot_id|>')
                    if isinstance(eot, int) and 0 <= eot < len(tokenizer):
                        eos.add(eot)
                    emit('generating', {'prompt_token_count': len(prompt_ids),
                                        'max_new_tokens': args.max_new_tokens})
                    started_generation = time.monotonic()
                    with torch.inference_mode():
                        generated = runner.generate(prompt_ids, args.max_new_tokens,
                                                    stop_tokens=eos or None)
                    generated = list(generated)
                    elapsed = time.monotonic() - started_generation
                    if not generated:
                        raise AssertionError('No output tokens generated')
                    if any(not isinstance(i, int) or not 0 <= i < len(tokenizer) for i in generated):
                        raise AssertionError('Generated token IDs invalid')
                    decoded = tokenizer.decode(generated, skip_special_tokens=True)
                    if not decoded.strip():
                        raise AssertionError('Generated token IDs decode to empty visible text')
                    report.update(generated_ids=generated, decoded_text=decoded,
                                  tokens_generated=len(generated),
                                  generation_seconds=round(elapsed, 3),
                                  tokens_per_sec=round(len(generated) / max(elapsed, 1e-9), 6))
                    unseen = [i for i in range(235) if hits[i] == 0]
                    report['unexecuted_layer_indices'] = unseen
                    report['layer_forward_counts_by_type'] = {
                        name: sum(hits[i] for i, kind in enumerate(schedule) if kind == name)
                        for name in sorted(set(schedule))
                    }
                    report['component_forward_counts'] = dict(extra_hits)
                    report['provider_counters'] = dict(getattr(runner.provider, 'counters', {}))
                    emit('generated', {'output': decoded, 'token_count': len(generated),
                                       'elapsed_seconds': report['generation_seconds'],
                                       'unexecuted_layers': unseen[:12],
                                       'component_hits': report['component_forward_counts']})
                    if unseen:
                        raise AssertionError(f'Generated text, but {len(unseen)} layers did not register forward hooks')
                    if not extra_hits['lm_head']:
                        raise AssertionError('Generated tokens without confirming lm_head.forward')
                    if model.reasoning is not None and not extra_hits['reasoning']:
                        raise AssertionError('Reasoning recurrence did not register forward calls')
                    if attnres is not None and not any(k.startswith('attnres.') for k in extra_hits):
                        raise AssertionError('Block AttnRes methods not observed')
                    if not expert_reads:
                        raise AssertionError('No checkpoint routed experts were loaded during generation')
                    report['actual_routed_expert_loads'] = sum(expert_reads.values())
                    report['distinct_routed_experts_loaded'] = len(expert_reads)
                    if args.expect_substring and args.expect_substring not in decoded:
                        raise AssertionError(f'Generated output did not contain expected text: {args.expect_substring!r}')
                    report['status'] = 'ALL_235_LAYERS_EXECUTED_AND_TOKENS_GENERATED'
                finally:
                    for h in handles:
                        h.remove()
                    for obj, method, original in patched_mamba:
                        setattr(obj, method, original)
                    for method, orig in original_methods.items():
                        setattr(attnres, method, orig)
        finally:
            StreamingWeightProvider.load_expert = original_load_expert
            if 'sample' in locals():
                report['peak_rss_gib'] = gib(sample.max_rss)
        report['peak_rss_gib'] = gib(sample.max_rss)
        report['cgroup_peak_gib'] = gib(cgroup_peak_bytes()) if cgroup_peak_bytes() is not None else None
        if report['peak_rss_gib'] >= args.max_rss_gib:
            raise AssertionError(f"Peak RSS {report['peak_rss_gib']} GiB exceeded budget {args.max_rss_gib} GiB")
        report['finished'] = True
        report['content_assertion_passed'] = bool(args.expect_substring)
        report['semantic_quality_verified'] = False
        emit('accepted', {'status': report['status'], 'peak_rss_gib': report['peak_rss_gib'],
                          'routed_expert_loads': report['actual_routed_expert_loads']})
    except BaseException as exc:
        report['status'] = 'FAILED'
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc(limit=15)
        emit('failed', {'type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic()-begin, 3)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        emit('report_written', {'path': str(args.output), 'status': report['status']})


if __name__ == '__main__':
    main()
