# Experiments

Tracking for every on-policy distillation run after the recipe first worked: what changed, and what
it scored. Each run is `manifests/env_vars` plus the overrides in `configs/<run>.env`. The shipped
`env_vars` defaults are **run17** (2B student, LoRA r=128), so `configs/run17.env` is now a no-op.

All evals: GSM8K test split (1,319 problems), 4-shot, greedy, `#### N` answer match. *z* is the paired
sign test against the base student on the same problems; "gap closed" is
`(after − before) / (teacher − before)`. Standard error of one checkpoint's accuracy is ~±1.3 pp.

## Baselines

Measured first, before any training (notebook Part 1): they pick the student and serve as the
`before` and `teacher` phases of every later eval.

| model | setting | accuracy |
|---|---|---|
| Qwen3.5-9B (teacher) | 4-shot `####` (shipped prompt) | **94.3%** (94.5–94.6% on other runs) |
| **Qwen3.5-2B (student — chosen)** | 4-shot `####` | **74.8%** — 19.6 pp gap, 276 teacher-solved problems missed |
| Qwen3.5-0.8B (candidate) | 4-shot `####` | **52.0%** — wider gap (567 missed), but it plateaus at ~65% |
| Qwen3.5-4B (candidate) | 4-shot `####` | **92.8%** — too small a gap to distil (52 missed) |
| Qwen3.5-0.8B / 9B | zero-shot `\boxed{}` (`TASK=gsm8k_native`) | 55.1% / 94.3% |
| Qwen3.5-0.8B / 9B | thinking mode, first 500 problems, 3,500 tokens | 37.4% / 87.4% (0.8B: 78% hit the cap) |

## Runs

| run | config | change vs. previous | best checkpoint | plateau |
|---|---|---|---|---|
| run13 | [`configs/run13.env`](configs/run13.env) | first working recipe: LR 1e-5, 100 steps | 64.7% (step 100), z = +8.86 | still rising at 100 |
| run14 | [`configs/run14.env`](configs/run14.env) | 200 steps | 65.8% (step 125), z = +9.54 | 63.5–65.8% from step 50 to 200 |
| run15 | [`configs/run15.env`](configs/run15.env) | LR 2e-5 | 65.1% (step 75), z = +8.98 | 63.5–65.1% from step 25 |
| run16 | [`configs/run16.env`](configs/run16.env) | **2B student**, LR 2e-5, 50 steps | **83.2%** (steps 25 and 50), z = +7.26 | 83.2% from step 25 — **43% of the gap** |
| run17 | [`configs/run17.env`](configs/run17.env) | 2B with **LoRA rank 128** (alpha 256) — **shipped** | **84.0%** (step 25), z = +7.45 | 83.9% at step 50 — **47% of the gap** |
| run18 | [`configs/run18.env`](configs/run18.env) | run17 sampled at **T=1.0**, **8 samples × 32 prompts** | 84.4% (step 50), z = +7.77 | 82.6% at step 25 — a tie with run17 within noise |
| run19 | [`configs/run19.env`](configs/run19.env) | run17 + **KL to the teacher's top-20 distribution** (`TOPK_KL=20`) | 84.2% (step 25), z = +7.87 | 83.5% at step 50 — a tie |
| run20 | [`configs/run20.env`](configs/run20.env) | run17 with Qwen's **native zero-shot `\boxed{}` prompt** (base 77.9%) | 83.9% (step 50), z = +5.87 | 83.6% at step 25 — a tie; 37% of its smaller gap |

**Decoding the shipped checkpoint with a majority vote** (run17, step 25; 8 samples at T=0.7, most
common answer scored — `EVAL_VOTES=8`):

| phases voting | base 2B | distilled 2B | teacher | gap closed |
|---|---|---|---|---|
| none (greedy) | 74.8% | 84.0% | 94.3% | 47% |
| **`after` only — shipped headline** | 74.8% greedy | **89.2%** (z = +11.8; 226 fixed / 35 broken) | 94.3% greedy | **74%** |
| all | 84.0% | 88.7% | 96.1% | 39% |

The distilled student's accuracy by number of votes: 82.8% (1), 86.3% (3), 87.8% (4), 88.9% (6),
89.2% (8); at least one of 8 is right on 94.9%. Two independent 8-vote runs agree on 94.6% of
problems (88.7% and 89.2%). In the CSV the `capped` and `mean_tokens` of a voted row average the 8
votes.

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
- **A larger student lifts the ceiling.** The 2B goes 74.8% → 83.2% (+8.5 pp, z = +7.26), closing 43% of
  its gap versus ~30% for the 0.8B, and ends 17 pp above the best distilled 0.8B. It also plateaus by
  step 25 at LR 2e-5, and it is the cleanest run: answer marker 92%, capped 7–8%, no length drift.

- **The adapter is not the main limit.** Rank 128 adds +0.8 pp over rank 32 (84.0% vs 83.2%, within
  ~1 SE), again plateauing by step 25; answer marker 94%, capped 6%.
- **The 4B is not worth distilling here**: at 92.8% it is within 1.8 pp of the teacher.
- **Exactly on-policy sampling does not raise the ceiling.** T=1.0 with 8 samples per prompt (run18)
  lags at step 25 (half as many distinct prompts seen) and ties run17 at step 50 (84.4% vs. 84.0%,
  ~5 problems). It stays stable: capped rollouts fall from 17% to 8% during training.

- **A denser teacher signal does not either.** The top-20 KL (run19) ties: 84.2% / 83.5%.
- **The prompt moves the base, not the plateau.** The native zero-shot `\boxed{}` prompt lifts the
  base 2B by 3.1 pp (77.9%), but run20 ends at 83.9% — the same as run17.
- **Every 2B variant ends at 83–84% greedy**, within ~1 SE: the ceiling is the student's capacity
  with this teacher. What still moves the reported number is decoding: an 8-vote majority on the
  distilled student gives 89.2%. A larger teacher (27B W4A16, documented in `env_vars`) is the
  remaining lever for training.

## Reproducing

```bash
cd code
set -a && source manifests/env_vars && source experiments/configs/run18.env && set +a
envsubst < manifests/opd-config.yaml-template        | kubectl --context "$CTX" apply -f -
envsubst < manifests/student-distill-job.yaml-template | kubectl --context "$CTX" apply -f -

# evaluate checkpoints as they appear (each eval needs no GPU and runs alongside training)
CRED_REFRESH_CMD='<your credential refresh command>' \
  experiments/scripts/eval_steps.sh "$RUN_DIR" run18 25 50
```

Changing `STUDENT_MODEL` or `LORA_R` also means re-applying `manifests/student-sampler.yaml-template`
(`--max-lora-rank` follows `LORA_R`), and a new student needs its baseline first (notebook Part 1,
with `BASE_DIR` pointing the scripts at it).

| script | purpose |
|---|---|
| [`scripts/eval_steps.sh`](scripts/eval_steps.sh) | waits for each `student-step<N>`, launches its eval Job, prints the comparison |
| [`scripts/summarize.py`](scripts/summarize.py) | accuracy, paired z, gap closed, marker/capped/length for any set of eval dirs |
| [`scripts/results_table.py`](scripts/results_table.py) | regenerates `results/gsm8k_evals.csv` from the eval dirs on `/fsx` |
