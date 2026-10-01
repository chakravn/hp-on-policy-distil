"""Rollout buffer for on-policy distillation.

The buffer is the seam between two rates: the *generator* (student sampling + teacher
scoring) fills it, and the *trainer* drains it. In the notebook the two alternate in one
process, so a plain deque is enough. In a real HyperPod job you would swap this for a
distributed queue (Ray / OpenRLHF engine) shared across the generation and training pools
-- the interface below is deliberately the minimum that swap would need to preserve.

Why it matters for distillation:
  * capacity      -- bounds memory; oldest rollouts are evicted (maxlen).
  * staleness     -- a rollout sampled under student weights theta_k but trained at
                     theta_{k+n} is *off-policy by n steps*. We stamp each rollout with
                     the step it was produced at so the trainer can bound staleness and
                     apply an importance-sampling correction (see distill_step).

THE STALENESS BUG THIS VERSION FIXES
------------------------------------
The original buffer held 256 rollouts, gained 8 per step and sampled 8 uniformly from the
whole deque. The expected age of a sampled rollout was therefore ~16 steps, so the loop
was not on-policy at all despite the name, and `logp_old` had drifted so far from the live
policy that the PPO ratio clip zeroed the gradient on most tokens. `mean_staleness` was
logged but never enforced.

Now: `max_age` is enforced on every read (`evict_stale`), and `max_age=0` gives a strictly
on-policy loop -- generate, train on exactly those rollouts, discard. That is the default.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List
import random


@dataclass
class Rollout:
    """One student trajectory, fully scored and ready to train on."""

    token_ids: List[int]           # prompt + student completion
    prompt_len: int                # number of prompt tokens (completion starts here)
    student_logp_old: List[float]  # student log-probs at sampling time (behaviour policy)
    teacher_logp: List[float]      # teacher log-probs for the same tokens
    produced_at_step: int = 0      # trainer step when this rollout was generated
    # True when generation stopped because it hit max_new_tokens rather than emitting EOS.
    # Training on unfinished reasoning teaches the student to never reach its final answer,
    # so the trainer drops (or masks) these -- see train_distill.--truncated-policy.
    truncated: bool = False
    prompt_text: str = ""          # kept for debugging / dumping example rollouts

    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len


class RolloutBuffer:
    """Bounded replay buffer with an enforced staleness ceiling.

    `max_age` is the maximum number of trainer steps a rollout may survive:
      * 0  -> strictly on-policy (the default): only rollouts produced this step are used.
      * n  -> allow n steps of replay, which the importance-sampling ratio corrects for.
    """

    def __init__(self, capacity: int = 1024, seed: int = 0, max_age: int = 0):
        self.buf: deque[Rollout] = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self.max_age = max_age
        self.total_added = 0
        self.total_evicted_stale = 0

    def add(self, rollout: Rollout) -> None:
        self.buf.append(rollout)
        self.total_added += 1

    def extend(self, rollouts: List[Rollout]) -> None:
        for r in rollouts:
            self.add(r)

    def __len__(self) -> int:
        return len(self.buf)

    def evict_stale(self, current_step: int) -> int:
        """Drop rollouts older than `max_age` steps. Returns how many were dropped."""
        keep = [r for r in self.buf if current_step - r.produced_at_step <= self.max_age]
        dropped = len(self.buf) - len(keep)
        if dropped:
            self.buf.clear()
            self.buf.extend(keep)
            self.total_evicted_stale += dropped
        return dropped

    def sample_batch(self, batch_size: int, current_step: int | None = None) -> List[Rollout]:
        """Uniformly sample without replacement, after enforcing the staleness ceiling."""
        if current_step is not None:
            self.evict_stale(current_step)
        n = min(batch_size, len(self.buf))
        return self.rng.sample(list(self.buf), n)

    def drain(self) -> List[Rollout]:
        """Take everything and empty the buffer -- the strictly on-policy read."""
        out = list(self.buf)
        self.buf.clear()
        return out

    def mean_staleness(self, current_step: int) -> float:
        """Average number of trainer steps since each buffered rollout was produced."""
        if not self.buf:
            return 0.0
        return sum(current_step - r.produced_at_step for r in self.buf) / len(self.buf)

    def clear(self) -> None:
        self.buf.clear()
