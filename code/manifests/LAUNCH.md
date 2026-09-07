# Launching on `hp-cluster-onpolicy-distillation` cluster
The SageMaker notebook role has been granted **EKS cluster-admin** (`AmazonEKSClusterAdminPolicy`
on its access entry), so it can create the teacher and student **directly from the notebook**
(`RUN_CLUSTER=True`) or from any admin `kubectl` context. 

## 0. One-time prerequisites
- **Download the teacher** `Qwen/Qwen3-4B` into the bucket (GUI → HF Download → S3, or the agent).
  It lands at `/s3/Qwen-Qwen3-4B`. (`Qwen/Qwen3-0.6B` is already there as the student.)
- **Sync the trainer code** so the student pod can read it:
  ```bash
  aws s3 sync src s3://<s3_bucket>/opd/src
  ```

**Storage layout:** `/fsx` (FSx for Lustre, PVC `fsx-claim`) is the high-throughput working store —
HF cache (`/fsx/hf_cache`), checkpoints and metrics (`/fsx/opd/run1`). `/s3` (PVC `s3-claim`) is the
model registry + code. Both are mounted in the teacher and student pods. *Optional, for fastest cold
starts:* stage the model weights onto FSx and point `TEACHER_MODEL`/`STUDENT_MODEL` at `/fsx/...`
(admin, e.g. `kubectl cp` into a pod's `/fsx`, or a staging pod that copies `/s3/Qwen-* → /fsx/`).

## 1. Teacher (vLLM Service)
**Admin kubectl:**
```bash
set -a && source manifests/env_vars && set +a
envsubst < manifests/teacher-vllm.yaml-template | kubectl --context "$CTX" apply -f -
kubectl --context "$CTX" -n default rollout status deploy/teacher-vllm --timeout=900s
```
**or instantstart agent (kiro-cli):**
> "Deploy `vllm serve /s3/Qwen-Qwen3-4B --served-model-name Qwen/Qwen3-4B --max-model-len 4096`
> with 1 GPU on ml.g6.12xlarge, service type clusterip, name teacher-vllm — and wait until the
> model pod is Running and /v1/models responds."

## 2. Student (on-policy distillation Job)
**Admin kubectl:**
```bash
envsubst < manifests/student-distill-job.yaml-template | kubectl --context "$CTX" apply -f -
```
**or instantstart** — launch as a **HyperPod Job (generic .sh)** pointing at
`/s3/opd/src/entry-distill.sh`, instance ml.g6.12xlarge, `nprocPerNode=1`, mounting `/s3`.
The verl RayJob path (KubeRay) is the alternative for fault-tolerant multi-node.

## 3. Monitor (works with the read-only notebook role)
```bash
kubectl --context "$CTX" -n default get pods -o wide
kubectl --context "$CTX" -n default logs -f job/opd-student
# reach the teacher from the notebook:
kubectl --context "$CTX" -n default port-forward svc/teacher-vllm 8000:8000
```
Metrics (per-step teacher KL) are written to `${RUN_DIR}/metrics_rank0.jsonl` on `/fsx`.

## 4. Cleanup
```bash
kubectl --context "$CTX" -n default delete job opd-student --ignore-not-found
kubectl --context "$CTX" -n default delete deploy,svc teacher-vllm --ignore-not-found
```
