# On-Policy Distillation on SageMaker HyperPod EKS — Code Talk

Compress a frontier **teacher** (`Qwen/Qwen3-4B`) into a lean **student** (`Qwen/Qwen3-0.6B`)
by training on the *student's own* math rollouts with **dense, per-token feedback** (reverse KL
to the teacher). No verifier, cheaper and more stable than RL with verifiable rewards.

**Target cluster:** `hp-cluster-hypd-0710-be86` (instantstart) · EKS `eks-hypd-0710-be86`,
`us-west-2`, ns `default`. Storage: **FSx `/fsx` (`fsx-claim`) = high-throughput working store**
(HF cache, checkpoints, metrics); **S3 `/s3` (`s3-claim`) = model registry**. Nodes: 2× g5.8xlarge
(1 GPU) + 2× g6.12xlarge (4 GPU).

## Contents
```
on_policy_distillation.ipynb        # the 20-min walkthrough (Parts 0–6)
slides/on-policy-distillation.pptx  # AWS-styled 5-min deck (build_pptx.py regenerates it)
slides/on-policy-distillation.md    # same deck as Marp Markdown
src/
  teacher_client.py                 # score student tokens via vLLM echo-logprobs (remote) or transformers (local)
  rollout_buffer.py                 # deque buffer: capacity, staleness, batch sampling
  distill_step.py                   # reverse-KL advantage + importance-sampling loss (core algorithm)
  train_distill.py                  # headless trainer the student Job runs
  entry-distill.sh                  # entry script for the instantstart "HyperPod Job (.sh)" path
manifests/
  env_vars                          # cluster, storage, models, run config
  teacher-vllm.yaml-template        # Deployment + ClusterIP Service (teacher pool)
  student-distill-job.yaml-template # K8s Job (student pool); no Kubeflow CRD on this cluster
  LAUNCH.md                         # how to create resources (admin context / instantstart) + monitor
```

## The method in one screen
```
advantage_t = -kl_coef * ( log π_student(y_t|y_<t) - log π_teacher(y_t|y_<t) )   # negative reverse KL
loss        = -mean_t [ stop_grad(advantage_t) * ratio_t ]                        # importance-sampling
ratio_t     = exp( log π_student_new(y_t) - stop_grad(log π_student_old(y_t)) )    # ≈1 on fresh rollouts
```
Track `mean_t reverse_kl` — it should trend **down** as the student matches the teacher.

## Running it
- **Live algorithm demo (Parts 1 & 3):** runs in the notebook kernel — needs a **GPU kernel**
  (present from the g6.12xlarge node). `TEACHER_MODE="local"` loads the teacher in-kernel (on GPU 1).
- **Cluster orchestration (Parts 2 & 4):** the notebook role has **EKS cluster-admin**, so `RUN_CLUSTER=True`
  creates the teacher `Service` and student `Job` directly from the notebook (works from a CPU kernel too;
  the cluster Job does the GPU training). The instantstart GUI / `kiro-cli` agent remains an alternative —
  see `manifests/LAUNCH.md`. Note: to reach the teacher from the notebook for the *in-kernel remote* path,
  `port-forward svc/teacher-vllm 8000:8000` and set `TEACHER_URL=http://localhost:8000/v1`.

```bash
set -a && source manifests/env_vars && set +a           # sets $CTX, storage, models
kubectl create secret generic hf-token --from-literal=token=hf_xxx -n default   # if pulling from HF
# (admin) deploy teacher + student — see manifests/LAUNCH.md:
envsubst < manifests/teacher-vllm.yaml-template       | kubectl --context "$CTX" apply -f -
aws s3 sync src s3://$S3_BUCKET/opd/src
envsubst < manifests/student-distill-job.yaml-template | kubectl --context "$CTX" apply -f -
kubectl --context "$CTX" -n default get pods -o wide    # read-only: teacher + student pods
```

## Who uses on-policy distillation (verified)
- **Qwen3** — student generates on-policy sequences, aligned to teacher (Qwen3-32B/235B) logits via
  KL divergence. *Qwen3 Technical Report, arXiv:2505.09388.*
- **Xiaomi MiMo-V2-Flash** — Multi-Teacher On-Policy Distillation (MOPD): teachers give dense,
  token-level reward on the student's rollouts. *arXiv:2601.02780.*
- **Method / recipe:** Thinking Machines Lab, *On-Policy Distillation* (2025); **GKD**: Agarwal et al.,
  *On-Policy Distillation of LMs*, ICLR 2024 (arXiv:2306.13649). Reference impl: Tinker Cookbook
  `distillation/` (`incorporate_kl_penalty`). Reverse-KL background: MiniLLM (arXiv:2306.08543).

**Not on-policy (useful contrast):** DeepSeek-R1/V3 and Kimi k1.5/k2 *distill off-policy* — SFT on
teacher-generated traces. Gemma 2/3, Llama 3.2/4, NVIDIA Minitron use token-level KD on a **fixed
corpus** (also off-policy). GLM-4.5/4.6 use off-policy expert distillation; GLM-5's report emphasizes
RL (no confirmed on-policy-distillation claim — omitted).
