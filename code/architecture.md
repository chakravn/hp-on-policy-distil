# On-Policy Distillation — Architecture

Two GPU pools, decoupled. The **teacher** is a long-lived vLLM inference `Service`
(static weights, scoring only); the **student** is a separate training `Job`. Each pod
gets a **dedicated GPU** (`nvidia.com/gpu: 1`) — they never share a GPU. Their only
coupling is the cluster network (the `Service` call) and the shared FSx/S3 mounts, so
each pool scales independently.

```mermaid
flowchart LR
  subgraph TP["TEACHER POOL — dedicated GPU · inference only"]
    T["Deployment teacher-vllm<br/>vllm serve Qwen3-4B<br/>static weights"]
    SVC(["Service teacher-vllm:8000"])
    T --> SVC
  end
  subgraph SP["STUDENT POOL — dedicated GPU · training"]
    S["Job opd-student<br/>torchrun train_distill.py<br/>generate rollouts + update"]
  end
  subgraph STO["SHARED STORAGE"]
    FSX[("FSx /fsx<br/>HF cache · checkpoints · metrics")]
    S3[("S3 /s3<br/>models · code")]
  end
  S -- "1 . student token_ids" --> SVC
  SVC -- "2 . per-token teacher logprobs" --> S
  S -. read/write .-> FSX
  S -. read .-> S3
  T -. read .-> S3
```

**Per-step loop:** ① student samples a rollout from π_student → ② teacher scores every
token (log π_teacher) → advantage = −(log π_student − log π_teacher) → student backprops
and updates θ. Repeat. The teacher never trains; scale each pool independently
(more `teacher-vllm` replicas for scoring throughput, more GPUs / FSDP for the student).
