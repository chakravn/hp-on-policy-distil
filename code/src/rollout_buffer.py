"""Rollout buffer for on-policy distillation.

The buffer is the seam between two rates: the *generator* (student sampling +
teacher scoring) fills it, and the *trainer* drains it. In the notebook the two
alternate in one process, so a plain deque is enough. In a real HyperPod job you
would swap this for a distributed queue (Ray / OpenRLHF engine) shared across the
generation and training pools -- the interface below is deliberately the minimum
that swap would need to preserve.

Why it matters for distillation:
  * capacity      -- bounds memory; oldest rollouts are evicted (maxlen).
  * staleness     -- a rollout scored against student weights theta_k but trained
                     at theta_{k+n} is *off-policy by n steps*. We stamp each
                     rollout with the step it was produced at so the trainer can
                     measure staleness and apply an importance-sampling correction
                     (see distill_step.distill_loss). Fresh (age==0) => on-policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
import random


@dataclass
class Rollout:
    """One student trajectory, fully scored and ready to train on."""
    token_ids: List[int]          # prompt + student completion
    prompt_len: int               # number of prompt tokens (completion starts here)
    student_logp_old: List[float] # student log-probs at sampling time (behavior policy)
    teacher_logp: List[float]     # teacher log-probs for the same tokens
    produced_at_step: int = 0     # trainer step when this rollout was generated

    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len


class RolloutBuffer:
    def __init__(self, capacity: int = 1024, seed: int = 0):
        self.buf: deque[Rollout] = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        self.total_added = 0

    def add(self, rollout: Rollout) -> None:
        self.buf.append(rollout)
        self.total_added += 1

    def extend(self, rollouts: List[Rollout]) -> None:
        for r in rollouts:
            self.add(r)

    def __len__(self) -> int:
        return len(self.buf)

    def sample_batch(self, batch_size: int) -> List[Rollout]:
        """Uniformly sample without replacement (falls back to all if buffer is small)."""
        n = min(batch_size, len(self.buf))
        return self.rng.sample(list(self.buf), n)

    def mean_staleness(self, current_step: int) -> float:
        """Average number of trainer steps since each buffered rollout was produced."""
        if not self.buf:
            return 0.0
        return sum(current_step - r.produced_at_step for r in self.buf) / len(self.buf)

    def clear(self) -> None:
        self.buf.clear()
