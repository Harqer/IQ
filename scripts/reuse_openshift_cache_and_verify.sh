#!/usr/bin/env bash
# Recover the IQ 235-layer checkpoint cache without downloading base shards again.
# Refuse overlapping verification runs; preserve evidence and the checkpoint PVC.
set -euo pipefail
cd "$HOME/IQ"
ns=$(oc project -q)
job=iq-full-verification
maintenance=iq-cache-reuse-once

if oc get job "$job" -n "$ns" >/dev/null 2>&1 ||
   [[ -n $(oc get pods -n "$ns" -l app=iq-full-verification -o name) ]]; then
  echo "ABORT: an IQ verification Job/pod already exists" >&2
  exit 1
fi
if ! grep -Eq '^[[:space:]]+memory: 24Gi$' deploy/iq-full-job.yaml; then
  echo "ABORT: expected local 24Gi memory request in deploy/iq-full-job.yaml" >&2
  exit 1
fi
if oc get pod iq-cache-audit -n "$ns" >/dev/null 2>&1; then
  phase=$(oc get pod iq-cache-audit -n "$ns" -o jsonpath='{.status.phase}')
  if [[ "$phase" != Succeeded && "$phase" != Failed ]]; then
    echo "ABORT: audit pod still active: $phase" >&2
    exit 1
  fi
  oc delete pod iq-cache-audit -n "$ns" --wait=true
fi
if oc get pod "$maintenance" -n "$ns" >/dev/null 2>&1; then
  echo "ABORT: cache maintenance pod already exists" >&2
  exit 1
fi

cat <<'YAML' | oc create -n "$ns" -f -
apiVersion: v1
kind: Pod
metadata:
  name: iq-cache-reuse-once
  labels:
    app: iq-cache-maintenance
spec:
  restartPolicy: Never
  containers:
    - name: maintenance
      image: python:3.12-slim
      command: ["python", "-u", "-c"]
      args:
        - |
          import hashlib
          import json
          import os
          import shutil
          from pathlib import Path

          cache = Path('/data/cache')
          bucket = 'kofecheckmate/iq-glm53-all256'
          run = 'IQ/runs/glm53-full78-local-20261025a'
          identity = hashlib.sha256(f'{bucket}\0{run}'.encode()).hexdigest()
          new = cache / 'checkpoints' / identity
          dest = new / 'store'
          old_index = cache / 'run'
          new_index = new / 'run'
          required = ('iq_config.json', 'model.safetensors.index.json',
                      'COMPILE_COMPLETE.json', 'transplant_manifest.json')

          def digest(p):
              with p.open('rb') as f:
                  return hashlib.file_digest(f, 'sha256').digest()

          for name in required:
              a, b = old_index / name, new_index / name
              if not (a.is_file() and b.is_file() and digest(a) == digest(b)):
                  raise RuntimeError(f'ABORT: checkpoint manifests disagree: {name}')

          files = sorted(cache.glob('shards_*'))
          if len(files) != 315 or not dest.is_dir():
              raise RuntimeError(f'ABORT: expected 315 legacy shards; found {len(files)}')
          weight_map = json.loads(
              (new_index / 'model.safetensors.index.json').read_text()
          )['weight_map']
          valid = set(weight_map.values())
          moves, duplicates = [], []
          for src in files:
              if src.is_symlink() or not src.is_file() or src.name[7:] not in valid:
                  raise RuntimeError(f'ABORT: unrecognized legacy shard: {src.name}')
              target = dest / src.name
              if target.exists():
                  if (not target.is_file() or
                      src.stat().st_size != target.stat().st_size or
                      digest(src) != digest(target)):
                      raise RuntimeError(f'ABORT: duplicate shard differs: {src.name}')
                  duplicates.append((src, target))
              else:
                  moves.append((src, target))
          print(f'Validated {len(files)} legacy shards: '
                f'{len(moves)} to move, {len(duplicates)} duplicates', flush=True)
          for src, target in moves:
              os.replace(src, target)  # same PVC, zero extra weight bytes
          for src, _ in duplicates:
              src.unlink()  # identical verified copy already exists in new cache
          free = shutil.disk_usage('/data').free / 2**30
          print(f'Cache reused. PVC free: {free:.2f} GiB', flush=True)
          if free < 12:
              raise RuntimeError('ABORT: free disk below safety floor')
          print('READY: no BF16 base-shard redownload needed', flush=True)
      resources:
        requests:
          memory: 128Mi
          cpu: 100m
        limits:
          memory: 512Mi
          cpu: 500m
      volumeMounts:
        - name: checkpoint-cache
          mountPath: /data
  volumes:
    - name: checkpoint-cache
      persistentVolumeClaim:
        claimName: iq-full-cache
YAML

# Follow completion and fail closed if the audit failed.
if ! oc wait -n "$ns" "pod/$maintenance" \
     --for=jsonpath='{.status.phase}'=Succeeded --timeout=15m; then
  oc logs -n "$ns" "pod/$maintenance" || true
  echo "ABORT: migration did not complete. Do not start inference." >&2
  exit 1
fi
oc logs -n "$ns" "pod/$maintenance"
oc delete pod -n "$ns" "$maintenance" --wait=true

# Enforce one verification Job, including races with another terminal.
if oc get job "$job" -n "$ns" >/dev/null 2>&1 ||
   [[ -n $(oc get pods -n "$ns" -l app=iq-full-verification -o name) ]]; then
  echo "ABORT: another verification appeared; no new Job created" >&2
  exit 1
fi

sed "s|__NAMESPACE__|$ns|g" deploy/iq-full-job.yaml |
  oc create -n "$ns" -f -
echo "Exactly one full 235-layer verification Job created."
oc logs -n "$ns" -f "job/$job" --timestamps --pod-running-timeout=15m
