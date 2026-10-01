"""Teacher solutions for the supervised warm-up -- and the teacher-solvable prompt filter.

WHY A WARM-UP
-------------
On-policy distillation only reweights tokens the student already samples. On MATH-500 with
a 0.8B student that left a gap: reasoning improved on problems the student finished, but it
stopped concluding -- the \\boxed rate fell from 69% to ~49%. A per-token loss on the sampled
token can push "keep reasoning" down but never push "\\boxed{...}<|im_end|>" up unless the
student happens to sample it. Imitating the teacher's own finished solutions teaches that
directly. This is phase 1 of the SFT -> on-policy recipe in the
HuggingFaceH4 on-policy-distillation write-up, which also filters to "correct teacher
completions only".

WHAT IT WRITES (``$SFT_DIR/teacher_solutions.jsonl``, one row per training problem)
    question, gold, prompt (exact chat-templated text the student is trained/evaluated on),
    completion, finish_reason, completion_tokens, answer, correct
Rows are appended as they finish and the script resumes from an existing file, so a
restart does not re-query the teacher. `sft_warmup.py` keeps rows that are correct AND
finished; the same rows' questions are the teacher-solvable prompt pool for the next
on-policy run.

Prompts are built exactly as `train_distill.py` builds them (same `build_prompt`, few-shot
count, chat template and thinking switch, student tokenizer), and are sent as token ids to
the completions endpoint, so the teacher continues the identical text the student sees.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from data import _load_split, _normalise_for_dedupe, _pick_field, QUESTION_FIELDS, load_eval_questions
from prompts import (
    DEFAULT_ANSWER_REL_TOL, DEFAULT_TEXT_SIMILARITY, answers_equivalent,
    build_prompt, extract_answer, get_task_spec, has_answer_marker,
)

try:  # same two-tier scorer as eval_student.py: math-verify first, tolerant fallback
    from math_verify import parse as mv_parse, verify as mv_verify   # type: ignore
except Exception:  # pragma: no cover
    mv_parse = mv_verify = None  # type: ignore


def is_correct(answer: str | None, gold: str) -> bool:
    if answer is None or not gold:
        return False
    if mv_parse is not None:
        try:
            if mv_verify(mv_parse(gold), mv_parse(answer)):
                return True
        except Exception:
            pass
    return answers_equivalent(answer, gold, DEFAULT_ANSWER_REL_TOL, DEFAULT_TEXT_SIMILARITY)[0]


def load_problems(args, spec) -> list[dict]:
    """MATH train problems with gold answers: deduped, eval-contamination-filtered, shuffled."""
    rows = _load_split(args.train_dataset, args.train_dataset_config, args.train_split)
    qfield = _pick_field(rows[0], QUESTION_FIELDS, spec.question_field)
    exclude = load_eval_questions(spec)
    seen, out = set(), []
    for row in rows:
        q = str(row[qfield]).strip()
        key = _normalise_for_dedupe(q)
        if not q or key in seen or key in exclude:
            continue
        seen.add(key)
        gold = spec.gold_fn(row)   # MATH: boxed solution; GSM8K: the number after '####'
        if gold:
            out.append({"id": key[:64], "question": q, "gold": gold})
    random.Random(args.seed).shuffle(out)
    return out[: args.limit] if args.limit else out


def main() -> int:
    env = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-url", default=env("TEACHER_URL", "http://teacher-vllm.default.svc.cluster.local:8000/v1"))
    ap.add_argument("--teacher-model", default=env("TEACHER_MODEL_ID", "Qwen/Qwen3.5-9B"))
    ap.add_argument("--student-model", default=env("STUDENT_MODEL", "Qwen/Qwen3.5-0.8B"),
                    help="tokenizer the prompts are built with -- the student's, as in training")
    ap.add_argument("--task", default=env("TASK", "math500"))
    ap.add_argument("--train-dataset", default=env("TRAIN_DATASET", "EleutherAI/hendrycks_math"))
    ap.add_argument("--train-dataset-config", default=env("TRAIN_DATASET_CONFIG", ""))
    ap.add_argument("--train-split", default=env("TRAIN_SPLIT", "train"))
    ap.add_argument("--fewshot", type=int, default=int(env("TRAIN_FEWSHOT", "4")))
    ap.add_argument("--enable-thinking", type=int, default=int(env("ENABLE_THINKING", "0") or "0"))
    ap.add_argument("--max-new-tokens", type=int, default=int(env("MAX_NEW_TOKENS", "2048")))
    ap.add_argument("--temperature", type=float, default=float(env("SFT_GEN_TEMPERATURE", "0")),
                    help="0 = greedy: the teacher's single best solution, as in its 81.8% eval")
    ap.add_argument("--limit", type=int, default=int(env("SFT_GEN_LIMIT", "4000")),
                    help="training problems to solve (0 = all ~7.5k)")
    ap.add_argument("--concurrency", type=int, default=int(env("SFT_GEN_CONCURRENCY", "32")))
    ap.add_argument("--seed", type=int, default=int(env("SEED", "0")))
    ap.add_argument("--out-dir", default=env("SFT_DIR", "/fsx/opd/sft1"))
    args = ap.parse_args()

    from openai import OpenAI
    from transformers import AutoTokenizer

    spec = get_task_spec(args.task)
    tok = AutoTokenizer.from_pretrained(args.student_model)
    out_path = Path(args.out_dir) / "teacher_solutions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text().splitlines() if l.strip()}
    problems = [p for p in load_problems(args, spec) if p["id"] not in done]
    print(f"[gen] {len(problems)} problems to solve ({len(done)} already in {out_path}); "
          f"teacher {args.teacher_model}, max_new_tokens={args.max_new_tokens}, "
          f"temperature={args.temperature}", flush=True)

    client = OpenAI(base_url=args.teacher_url, api_key="EMPTY", timeout=900)
    lock = threading.Lock()
    stats = {"n": 0, "correct": 0, "finished": 0, "kept": 0}
    t0 = time.time()

    def solve(p: dict) -> dict:
        prompt = build_prompt(spec, p["question"], args.fewshot, tok, chat=True,
                              enable_thinking=bool(args.enable_thinking))
        resp = client.completions.create(
            model=args.teacher_model,
            prompt=tok(prompt, add_special_tokens=False).input_ids,
            max_tokens=args.max_new_tokens, temperature=args.temperature,
            stop=list(spec.stop_seqs),
        )
        choice = resp.choices[0]
        text = choice.text
        answer = extract_answer(text, spec.answer_style)
        return {
            **p, "prompt": prompt, "completion": text,
            "finish_reason": choice.finish_reason,
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "has_marker": has_answer_marker(text, spec.answer_style),
            "answer": answer, "correct": is_correct(answer, p["gold"]),
        }

    with out_path.open("a") as f, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(solve, p) for p in problems]
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as exc:  # one bad request must not lose the whole pass
                print(f"[gen] request failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                stats["n"] += 1
                stats["correct"] += row["correct"]
                finished = row["finish_reason"] == "stop" and row["has_marker"]
                stats["finished"] += finished
                stats["kept"] += row["correct"] and finished
                if stats["n"] % 200 == 0 or stats["n"] == len(problems):
                    n = stats["n"]
                    print(f"[gen] {n}/{len(problems)}  correct={stats['correct']/n:.3f}  "
                          f"finished={stats['finished']/n:.3f}  kept={stats['kept']}  "
                          f"({n/(time.time()-t0)*60:.0f}/min)", flush=True)
    print(f"[gen] done: {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
