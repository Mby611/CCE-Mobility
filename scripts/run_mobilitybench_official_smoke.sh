#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CCE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CCE_ROOT="${CCE_ROOT:-$DEFAULT_CCE_ROOT}"
MB_ROOT="${MB_ROOT:?Set MB_ROOT to a local MobilityBench checkout}"
if [[ ! -d "$MB_ROOT" ]]; then
  echo "MobilityBench checkout not found: $MB_ROOT" >&2
  exit 2
fi

if [[ -f "$CCE_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$CCE_ROOT/.env"
  set +a
fi

if [[ -n "${CCE_MODEL_API_KEY:-}" && -z "${LLM_API_KEY:-}" ]]; then
  LLM_API_KEY=$CCE_MODEL_API_KEY
fi
if [[ -z "${AIHUB_API_KEY:-}" && -z "${LLM_API_KEY:-}" ]]; then
  LOCAL_SECRET_KEY="$(
    CCE_ROOT="$CCE_ROOT" python3 - <<'PY'
import importlib
import os
import sys
from pathlib import Path

root = Path(os.environ["CCE_ROOT"])
sys.path.insert(0, str(root / "src"))
try:
    local_secrets = importlib.import_module("cce_mobility.local_secrets")
except Exception:
    print("")
else:
    print(str(
        getattr(local_secrets, "CCE_MODEL_API_KEY", "")
        or getattr(local_secrets, "AIHUB_API_KEY", "")
        or ""
    ))
PY
  )"
  if [[ -n "$LOCAL_SECRET_KEY" ]]; then
    AIHUB_API_KEY=$LOCAL_SECRET_KEY
    export AIHUB_API_KEY
  fi
fi

if [[ -z "${AIHUB_API_KEY:-}" && -z "${LLM_API_KEY:-}" ]]; then
  echo "Missing acting key. Fill CCE_MODEL_API_KEY in cce_mobility.local_secrets or set LLM_API_KEY." >&2
  exit 2
fi

LLM_API_KEY=${LLM_API_KEY:-${AIHUB_API_KEY:-}}
export LLM_API_KEY
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CCE_ROOT/.uv-cache}"
export PYTHONPATH="$CCE_ROOT/patches/mobilitybench${PYTHONPATH:+:$PYTHONPATH}"
export CCE_MOBILITYBENCH_DISABLE_THINKING="${CCE_MOBILITYBENCH_DISABLE_THINKING:-0}"
export CCE_MOBILITYBENCH_FORCE_TOOL_CHOICE_REQUIRED="${CCE_MOBILITYBENCH_FORCE_TOOL_CHOICE_REQUIRED:-0}"
export CCE_MOBILITYBENCH_LOOP_SAFE_ASYNC_CLIENT="${CCE_MOBILITYBENCH_LOOP_SAFE_ASYNC_CLIENT:-1}"

DATASET="${DATASET:?Set DATASET to a local official MobilityBench CSV}"
CONFIG="${CONFIG:?Set CONFIG to a local MobilityBench model config}"
for input in "$DATASET" "$CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "Required input not found: $input" >&2
    exit 2
  fi
done
OUTPUT_DIR="${OUTPUT_DIR:-$CCE_ROOT/outputs/mobilitybench_official_smoke_plan_and_execute}"
MODEL="${MODEL:?Set MODEL to the alias in CONFIG}"
FRAMEWORK="${FRAMEWORK:-plan_and_execute}"

cd "$MB_ROOT"
CMD=(
  uv --cache-dir "$UV_CACHE_DIR" run mbench run
  --model "$MODEL" \
  --dataset "$DATASET" \
  --framework "$FRAMEWORK" \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --sandbox
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi

"${CMD[@]}"
