#!/bin/bash
# Entry script for the on-policy distillation student, for the instantstart
# "HyperPod Job (generic .sh)" path. It reads code from the S3 mount, waits for the
# teacher vLLM Service, then runs the trainer. Sync this dir to S3 first:
#   aws s3 sync src s3://eks-hypd-workspace-07151100-s3-us-west-2/opd/src
set -euo pipefail

TEACHER_URL="${TEACHER_URL:-http://teacher-vllm.default.svc.cluster.local:8000/v1}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-4B}"
STUDENT_MODEL="${STUDENT_MODEL:-/s3/Qwen-Qwen3-0.6B}"
CODE_DIR="${CODE_DIR:-/s3/opd/src}"
RUN_DIR="${RUN_DIR:-/fsx/opd/run1}"           # checkpoints + metrics on FSx (high throughput)
STEPS="${STEPS:-200}"
GPU_PER_NODE="${GPU_PER_NODE:-1}"
export HF_HOME="${HF_HOME:-/fsx/hf_cache}"    # HF cache on FSx

pip install -q "openai>=1.40" "datasets"

echo "waiting for teacher at ${TEACHER_URL} ..."
until python3 -c "import urllib.request; urllib.request.urlopen('${TEACHER_URL}/models')" 2>/dev/null; do sleep 10; done
echo "teacher is up."

cd "${CODE_DIR}"
torchrun --nproc_per_node="${GPU_PER_NODE}" train_distill.py \
  --teacher-url "${TEACHER_URL}" \
  --teacher-model "${TEACHER_MODEL}" \
  --student-model "${STUDENT_MODEL}" \
  --steps "${STEPS}" \
  --output-dir "${RUN_DIR}"
