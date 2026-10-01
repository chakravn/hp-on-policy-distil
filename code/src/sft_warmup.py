"""Supervised warm-up: LoRA fine-tune the student on the teacher's correct, finished solutions.

Phase 1 of SFT -> on-policy distillation (see gen_teacher_data.py for why). The output is a
plain PEFT adapter, so it plugs into the rest of the pipeline unchanged:
  * eval:      EVAL_FT_MODEL_DIR=$SFT_DIR/adapter-final with EVAL_STUDENT_URL set (the
               served route renames it for vLLM via sampler.export_adapter_for_vllm)
  * next run:  INIT_ADAPTER=$SFT_DIR/adapter-final for train_distill.py

Training examples are ``prompt + completion + <|im_end|>`` with the loss on the completion
and the end-of-turn token only. The prompt is the exact chat-templated text recorded by
gen_teacher_data.py, i.e. the same prompt train_distill.py and eval_student.py use -- and
the explicit <|im_end|> is the point: it is the "stop here" the student has lost.

Loss is the token mean over each optimizer batch (sum of completion-token NLL / completion
tokens), accumulated over micro-batches of one sequence, so the update does not depend on
how the batch is split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from model_setup import (
    build_optimizer, build_scheduler, load_student, load_tokenizer, memory_report,
    trainable_parameter_report,
)


def load_examples(path: Path, tok, max_examples: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    keep = [r for r in rows if r["correct"] and r["finish_reason"] == "stop" and r["has_marker"]]
    random.Random(seed).shuffle(keep)
    if max_examples:
        keep = keep[:max_examples]
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    if eos is None or eos == tok.unk_token_id:
        eos = tok.eos_token_id
    examples = []
    for r in keep:
        p = tok(r["prompt"], add_special_tokens=False).input_ids
        c = tok(r["completion"].rstrip(), add_special_tokens=False).input_ids + [eos]
        examples.append({"ids": p + c, "prompt_len": len(p)})
    print(f"[sft] {len(keep)} of {len(rows)} teacher rows kept (correct + finished); "
          f"completion tokens mean={sum(len(e['ids']) - e['prompt_len'] for e in examples) / max(1, len(examples)):.0f}",
          flush=True)
    return examples


def completion_nll_sum(model, ex: dict, device: str) -> tuple[torch.Tensor, int]:
    """Summed NLL of the completion tokens (incl. <|im_end|>) and how many there were."""
    ids = torch.tensor([ex["ids"]], dtype=torch.long, device=device)
    # Only the completion positions need logits. With a 248k vocab the LM head over the
    # ~800-token few-shot prompt is most of the cost, so ask for the last C+1 positions only
    # (the train_distill anchor ran ~6 s/example with full logits).
    keep = len(ex["ids"]) - ex["prompt_len"] + 1
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        try:
            logits = model(input_ids=ids, logits_to_keep=keep).logits[0, :-1]
        except TypeError:  # models without logits_to_keep
            logits = model(input_ids=ids).logits[0, ex["prompt_len"] - 1 : -1]
    targets = ids[0, ex["prompt_len"] :]
    nll = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum")
    return nll, int(targets.numel())


def main() -> int:
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student-model", default=env("STUDENT_MODEL", "Qwen/Qwen3.5-0.8B"))
    ap.add_argument("--sft-dir", default=env("SFT_DIR", "/fsx/opd/sft1"))
    ap.add_argument("--epochs", type=int, default=int(env("SFT_EPOCHS", "2")))
    ap.add_argument("--batch-size", type=int, default=int(env("SFT_BATCH", "32")),
                    help="sequences per optimizer step (accumulated one at a time)")
    ap.add_argument("--lr", type=float, default=float(env("SFT_LR", "1e-4")))
    ap.add_argument("--warmup-ratio", type=float, default=float(env("WARMUP_RATIO", "0.05")))
    ap.add_argument("--weight-decay", type=float, default=float(env("WEIGHT_DECAY", "0.0")))
    ap.add_argument("--grad-clip", type=float, default=float(env("GRAD_CLIP", "1.0")))
    ap.add_argument("--lora-r", type=int, default=int(env("LORA_R", "32")))
    ap.add_argument("--lora-alpha", type=int, default=int(env("LORA_ALPHA", "64")))
    ap.add_argument("--max-examples", type=int, default=int(env("SFT_MAX_EXAMPLES", "0")))
    ap.add_argument("--val-frac", type=float, default=float(env("SFT_VAL_FRAC", "0.02")))
    ap.add_argument("--log-every", type=int, default=int(env("LOG_EVERY", "10")))
    ap.add_argument("--seed", type=int, default=int(env("SEED", "0")))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = Path(args.sft_dir)
    tok = load_tokenizer(args.student_model)
    examples = load_examples(out / "teacher_solutions.jsonl", tok, args.max_examples, args.seed)
    n_val = max(1, int(len(examples) * args.val_frac))
    val, train = examples[:n_val], examples[n_val:]

    model = load_student(args.student_model, device=device, precision="lora",
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    print(f"[sft] {trainable_parameter_report(model)} {memory_report(device)}", flush=True)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    opt = build_optimizer(model, args.lr, args.weight_decay)
    sched = build_scheduler(opt, total_steps, args.warmup_ratio)
    print(f"[sft] train={len(train)} val={len(val)} epochs={args.epochs} batch={args.batch_size} "
          f"steps={total_steps} lr={args.lr}", flush=True)

    def val_loss() -> float:
        model.eval()
        nll, n = 0.0, 0
        with torch.no_grad():
            for ex in val:
                s, k = completion_nll_sum(model, ex, device)
                nll, n = nll + float(s), n + k
        model.train()
        return nll / max(1, n)

    metrics = (out / "sft_metrics.jsonl").open("a")
    print(f"[sft] step 0 val_loss={val_loss():.4f}", flush=True)
    step, t0 = 0, time.time()
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for b in range(0, len(order), args.batch_size):
            batch = [train[i] for i in order[b : b + args.batch_size]]
            n_tok = sum(len(e["ids"]) - e["prompt_len"] for e in batch)
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for ex in batch:
                nll, _ = completion_nll_sum(model, ex, device)
                (nll / n_tok).backward()
                loss_sum += float(nll)
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            rec = {"step": step, "epoch": epoch, "loss": round(loss_sum / n_tok, 4),
                   "grad_norm": round(float(gn), 4), "lr": sched.get_last_lr()[0],
                   "sec": round(time.time() - t0, 1)}
            if step % args.log_every == 0 or step == total_steps:
                print(f"[sft] {rec}", flush=True)
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
        vl = val_loss()
        ckpt = out / f"adapter-epoch{epoch + 1}"
        model.save_pretrained(ckpt, safe_serialization=True)
        tok.save_pretrained(ckpt)
        print(f"[sft] epoch {epoch + 1} val_loss={vl:.4f} saved {ckpt}", flush=True)
        metrics.write(json.dumps({"epoch_end": epoch + 1, "val_loss": vl}) + "\n")
    final = out / "adapter-final"
    model.save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    print(f"[sft] done; adapter at {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
