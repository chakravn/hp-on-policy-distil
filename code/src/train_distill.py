"""Standalone on-policy distillation trainer -- the script the student Job runs.

It is the exact loop from the notebook, packaged as a CLI so it can run headless on
the student GPU pool while querying the teacher vLLM Service across the network:

    generate rollouts (student)  ->  score (remote teacher)  ->  rollout buffer
        ->  reverse-KL advantage  ->  importance-sampling update  ->  log KL / checkpoint

Launch (single GPU, the demo default; paths are on the /s3 mount of this cluster)::

    torchrun --nproc_per_node=1 train_distill.py \
        --teacher-url http://teacher-vllm.default.svc.cluster.local:8000/v1 \
        --teacher-model Qwen/Qwen3-4B \
        --student-model Qwen/Qwen3-0.6B-Base \
        --steps 200 --output-dir /fsx/opd/run1   # checkpoints/metrics on FSx (high throughput)

Scaling the *student* to a multi-GPU FSDP job is the natural next step -- wrap the
model with torch FSDP and use `summon_full_params` around `.generate()`; see
awsome-distributed-training/3.test_cases/pytorch/FSDP for the reference wiring.
This demo keeps one GPU per worker so the loop stays readable.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from teacher_client import make_teacher
from rollout_buffer import Rollout, RolloutBuffer
from distill_step import sequence_logprobs, score_rollout_for_training


def load_math_prompts(n: int) -> list[str]:
    """A few math prompts; swap for DeepMath-103K / GSM8K via `datasets.load_dataset`."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=f"train[:{n}]")
        return [row["question"] for row in ds]
    except Exception:
        base = [
            "What is 17 * 24? Show your reasoning step by step.",
            "A train travels 60 km in 45 minutes. What is its speed in km/h?",
            "If 3x + 7 = 22, what is x? Explain each step.",
            "What is the sum of the first 20 positive integers?",
        ]
        return [base[i % len(base)] for i in range(n)]


def build_prompt(tokenizer, question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return question + "\n"


def generate_rollout(student, tokenizer, prompt_text, max_new_tokens, temperature, device, step):
    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = enc.input_ids.shape[1]
    with torch.no_grad():
        out = student.generate(
            **enc, do_sample=True, temperature=temperature, top_p=1.0,
            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
        )
    seq = out[0]
    student_logp_old = sequence_logprobs(student, seq, with_grad=False)  # [T-1], detached
    return Rollout(
        token_ids=seq.tolist(),
        prompt_len=prompt_len,
        student_logp_old=student_logp_old.float().cpu().tolist(),
        teacher_logp=[],  # filled after teacher scoring
        produced_at_step=step,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-url", default=os.environ.get("TEACHER_URL", "http://teacher-vllm.default.svc.cluster.local:8000/v1"))
    ap.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "Qwen/Qwen3-4B"))
    ap.add_argument("--student-model", default=os.environ.get("STUDENT_MODEL", "Qwen/Qwen3-0.6B-Base"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=8, help="rollouts generated per step")
    ap.add_argument("--batch-size", type=int, default=8, help="rollouts sampled from buffer per update")
    ap.add_argument("--buffer-capacity", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kl-coef", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--output-dir", default=os.environ.get("RUN_DIR", "/fsx/opd/run1"))
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / f"metrics_rank{local_rank}.jsonl"

    print(f"[rank {local_rank}] loading student {args.student_model} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16
    ).to(device).train()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    teacher = make_teacher("remote", base_url=args.teacher_url, model=args.teacher_model)
    buffer = RolloutBuffer(capacity=args.buffer_capacity)
    prompts = load_math_prompts(max(args.group_size * 4, 32))

    print(f"[rank {local_rank}] teacher @ {args.teacher_url}; starting {args.steps} steps", flush=True)
    for step in range(args.steps):
        t0 = time.time()
        # 1) generate rollouts from the current student (on-policy)
        for i in range(args.group_size):
            q = prompts[(step * args.group_size + i) % len(prompts)]
            r = generate_rollout(student, tokenizer, build_prompt(tokenizer, q),
                                 args.max_new_tokens, args.temperature, device, step)
            # 2) score every token with the remote teacher
            r.teacher_logp = teacher.score(r.token_ids)
            buffer.add(r)

        # 3) sample a batch and take one importance-sampling update
        batch = buffer.sample_batch(args.batch_size)
        opt.zero_grad()
        step_kl = 0.0
        for r in batch:
            seq = torch.tensor(r.token_ids, device=device)
            logp_old = torch.tensor(r.student_logp_old, device=device)
            loss, mean_kl = score_rollout_for_training(
                student, seq, r.prompt_len, r.teacher_logp, logp_old, args.kl_coef
            )
            (loss / len(batch)).backward()
            step_kl += mean_kl / len(batch)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        rec = {"step": step, "teacher_kl": step_kl,
               "buffer": len(buffer), "staleness": buffer.mean_staleness(step),
               "sec": round(time.time() - t0, 2)}
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if step % 10 == 0:
            print(f"[rank {local_rank}] {rec}", flush=True)

        if args.save_every and (step + 1) % args.save_every == 0 and local_rank == 0:
            ckpt = out / f"student-step{step+1}"
            student.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"[rank {local_rank}] saved {ckpt}", flush=True)

    if local_rank == 0:
        student.save_pretrained(out / "student-final")
        tokenizer.save_pretrained(out / "student-final")
    print(f"[rank {local_rank}] done", flush=True)


if __name__ == "__main__":
    main()
