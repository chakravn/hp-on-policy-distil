# Experiments

Tracking for every on-policy distillation run after the recipe first worked: what changed, and what
it scored. Each run is `manifests/env_vars` plus the overrides in `configs/<run>.env`.

All evals: GSM8K test split (1,319 problems), 4-shot, greedy, `#### N` answer match. *z* is the paired
sign test against the base student on the same problems; "gap closed" is
`(after − before) / (teacher − before)`. Standard error of one checkpoint's accuracy is ~±1.3 pp.

## Baselines

| model | setting | accuracy |
|---|---|---|
| Qwen3.5-0.8B (student) | 4-shot `####` (shipped prompt) | **52.0%** |
| Qwen3.5-9B (teacher) | 4-shot `####` | **94.5%** |
| Qwen3.5-0.8B / 9B | zero-shot `\boxed{}` (`TASK=gsm8k_native`) | 55.1% / 94.3% |
| Qwen3.5-0.8B / 9B | thinking mode, first 500 problems, 3,500 tokens | 37.4% / 87.4% (0.8B: 78% hit the cap) |

## Runs

| run | config | change vs. previous | best checkpoint | plateau |
|---|---|---|---|---|
| run13 | [`configs/run13.env`](configs/run13.env) | first working recipe: LR 1e-5, 100 steps | 64.7% (step 100), z = +8.86 | still rising at 100 |
| run14 | [`configs/run14.env`](configs/run14.env) | 200 steps | 65.8% (step 125), z = +9.54 | 63.5–65.8% from step 50 to 200 |
| run15 | [`configs/run15.env`](configs/run15.env) | LR 2e-5 | 65.1% (step 75), z = +8.98 | 63.5–65.1% from step 25 |
| run16 | [`configs/run16.env`](configs/run16.env) | 2B student, LR 2e-5, 50 steps | *running* | |

Every checkpoint: [`results/gsm8k_evals.csv`](results/gsm8k_evals.csv).

## What the runs established

- **The 0.8B plateaus at ~65% (~30% of the teacher gap).** 200 steps scored the same as 100; doubling
  the LR reached the plateau ~4× sooner (step 25 instead of ~100) without raising it. Training stayed
  stable throughout (answer marker ~85%, capped ~14–17%).
- **The limit is capacity, not training.** `teacher_kl` falls ~11% and then flattens with accuracy; the
  plateau starts before one pass over the 7.5k training prompts. A 0.8B with a rank-32 LoRA cannot
  represent the 9B's distribution, so its reverse KL has a high floor.
- **Shipped defaults** follow from this: `LR=2e-5`, `STEPS=50` — the same accuracy in ~4.5 h instead of 9.
- **Thinking mode hurts** with this prompt: the 0.8B thinks until it hits the token cap.

Next: run16 (2B student) tests whether a larger student lifts the plateau; LoRA rank 128 on the 0.8B
would test whether the adapter, not the model size, is the limit.

## Reproducing

```bash
cd code
set -a && source manifests/env_vars && source experiments/configs/run15.env && set +a
envsubst < manifests/opd-config.yaml-template        | kubectl --context "$CTX" apply -f -
envsubst < manifests/student-distill-job.yaml-template | kubectl --context "$CTX" apply -f -

# evaluate checkpoints as they appear (each eval needs no GPU and runs alongside training)
CRED_REFRESH_CMD='<your credential refresh command>' \
  experiments/scripts/eval_steps.sh "$RUN_DIR" run15 25 50 75 100
```

Changing `STUDENT_MODEL` (run16) also means re-applying `manifests/student-sampler.yaml-template` and
measuring that student's baseline first (`EVAL_PHASE=before,teacher`, with `BASE_DIR` pointing the
scripts at it).

| script | purpose |
|---|---|
| [`scripts/eval_steps.sh`](scripts/eval_steps.sh) | waits for each `student-step<N>`, launches its eval Job, prints the comparison |
| [`scripts/summarize.py`](scripts/summarize.py) | accuracy, paired z, gap closed, marker/capped/length for any set of eval dirs |
| [`scripts/results_table.py`](scripts/results_table.py) | regenerates `results/gsm8k_evals.csv` from the eval dirs on `/fsx` |
