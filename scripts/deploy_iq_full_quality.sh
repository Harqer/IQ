#!/usr/bin/env bash
# Build corrected 235-layer image and launch one correctness-gated Job.
# Requires the previously working /tmp/iq-full-build context and oc login.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
ns=$(oc project -q)
job=iq-full-verification
pod_label=app=iq-full-verification
context="/tmp/iq-full-build"

if [[ ! -f "$context/Dockerfile" ]] ||
   [[ ! -d "$context/iq_model" ]] ||
   [[ ! -d "$context/iq_local" ]] ||
   [[ ! -d "$context/scripts" ]]; then
  echo "ABORT: previous build context missing at $context" >&2
  exit 1
fi

# The previous completed verification may exist. Never replace an active Job.
if oc get job "$job" -n "$ns" >/dev/null 2>&1; then
  status=$(oc get job "$job" -n "$ns" -o jsonpath='{.status.succeeded}')
  active=$(oc get job "$job" -n "$ns" -o jsonpath='{.status.active}')
  if [[ "$status" != 1 || ( -n "$active" && "$active" != 0 ) ]]; then
    echo "ABORT: verification Job is not successfully complete" >&2
    exit 1
  fi
fi
for pod in $(oc get pods -n "$ns" -l "$pod_label" -o name); do
  phase=$(oc get -n "$ns" "$pod" -o jsonpath='{.status.phase}')
  if [[ "$phase" != Succeeded && "$phase" != Failed ]]; then
    echo "ABORT: active verification pod $pod ($phase)" >&2
    exit 1
  fi
done

# Stage only runtime source; never bundle checkpoint weights or HF credentials.
cp -a iq_model/. "$context/iq_model/"
cp -a iq_local/. "$context/iq_local/"
cp scripts/serve_iq_local.py scripts/verify_iq_full_inference.py "$context/scripts/"
cp requirements-local.txt "$context/requirements-local.txt"
cp deploy/Dockerfile.iq-full "$context/Dockerfile"

if ! cmp -s iq_local/quality.py "$context/iq_local/quality.py"; then
  echo "ABORT: new quality gate was not staged" >&2
  exit 1
fi
if ! grep -q -- '--require-arithmetic-canary' deploy/iq-full-job.yaml; then
  echo "ABORT: Job lacks correctness gate" >&2
  exit 1
fi
if ! grep -q 'memory: 24Gi' deploy/iq-full-job.yaml; then
  echo "ABORT: expected 24Gi memory request" >&2
  exit 1
fi

echo "Building updated full-model image; no inference Job yet."
oc start-build iq-full -n "$ns" --from-dir="$context" --follow --wait

# Save old success logs before replacing its completed Job.
if oc get job "$job" -n "$ns" >/dev/null 2>&1; then
  log_file="/tmp/iq-full-previous-$(date +%Y%m%d-%H%M%S).log"
  oc logs -n "$ns" "job/$job" > "$log_file" 2>&1 || true
  echo "Saved previous Job logs: $log_file"
  oc delete job "$job" -n "$ns" --cascade=foreground --wait=true
fi

if [[ -n $(oc get pods -n "$ns" -l "$pod_label" -o name) ]] ||
   oc get job "$job" -n "$ns" >/dev/null 2>&1; then
  echo "ABORT: verification resources still exist" >&2
  exit 1
fi

echo "Creating exactly one real 235-layer quality-gated Job."
sed "s|__NAMESPACE__|$ns|g" deploy/iq-full-job.yaml |
  oc create -n "$ns" -f -
oc get jobs,pods -n "$ns" -l "$pod_label"
echo "The Job executes all 235 layers and records strict canary results."
oc logs -n "$ns" -f "job/$job" --timestamps --pod-running-timeout=15m
