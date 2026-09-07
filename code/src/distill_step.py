"""The core on-policy distillation math: per-token reverse KL -> advantage -> loss.

Faithful to the Thinking Machines / Tinker `incorporate_kl_penalty` recipe:

    reverse_kl_t = log pi_student(y_t | y_<t) - log pi_teacher(y_t | y_<t)
    advantage_t  = - kl_coef * reverse_kl_t                     # negative reverse KL
    loss         = - mean_t [ stop_grad(advantage_t) * ratio_t ]
    ratio_t      = exp( log pi_student_new(y_t) - stop_grad(log pi_student_old(y_t)) )

`ratio_t` is the importance weight of the *current* student against the student
that generated the rollout. On fresh (on-policy) rollouts ratio ~= 1 and the loss
reduces to the intuitive `-(advantage * logp_new).mean()`; the ratio is what lets
us reuse slightly stale rollouts from the buffer without bias (PPO-style).

Alignment (see teacher_client): `teacher_logp` has length T and is aligned to
`token_ids` (element t = teacher log-prob of token t). A model's own per-token
log-probs have length T-1 (index i = log-prob of token i+1), so we compare against
`teacher_logp[1:]`. Completion tokens y_t live at sequence positions
[prompt_len, T-1], i.e. log-prob indices [prompt_len-1, T-2].
"""

from __future__ import annotations

from typing import List
import torch


def completion_slice(prompt_len: int, seq_len: int) -> slice:
    """Indices into a length-(T-1) per-token log-prob array selecting completion tokens."""
    return slice(prompt_len - 1, seq_len - 1)


def sequence_logprobs(model, seq_ids: torch.Tensor, with_grad: bool) -> torch.Tensor:
    """Per-token log-probs of a single sequence: returns [T-1], index i = log p(token i+1)."""
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        logits = model(seq_ids.unsqueeze(0)).logits[0]          # [T, V]
        logp = logits[:-1].log_softmax(-1)                      # [T-1, V]
        tok_logp = logp.gather(-1, seq_ids[1:, None]).squeeze(-1)  # [T-1]
    return tok_logp


def reverse_kl(student_logp: torch.Tensor, teacher_logp_full: torch.Tensor) -> torch.Tensor:
    """Per-token reverse KL estimate over the completion span. Inputs already sliced/aligned."""
    return student_logp - teacher_logp_full  # log pi_student - log pi_teacher


def distill_loss(
    logp_new: torch.Tensor,        # [C] current student log-probs (requires grad), completion span
    logp_old: torch.Tensor,        # [C] student log-probs at sampling time (detached)
    advantage: torch.Tensor,       # [C] = -kl_coef * reverse_kl (detached)
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """PPO-style importance-sampling surrogate; equals -(advantage*logp_new).mean() when ratio==1."""
    ratio = (logp_new - logp_old.detach()).exp()
    adv = advantage.detach()
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    # We want to *maximize* advantage-weighted log-prob, so minimize the negative min.
    return -torch.min(unclipped, clipped).mean()


def score_rollout_for_training(
    student_model,
    seq_ids: torch.Tensor,      # [T] prompt + completion, on the student's device
    prompt_len: int,
    teacher_logp_full: List[float] | torch.Tensor,  # [T], aligned to seq_ids
    student_logp_old: torch.Tensor,                  # [T-1], from sampling time (detached)
    kl_coef: float,
):
    """One rollout -> (loss, mean_reverse_kl). Recomputes the student pass with grad."""
    device = seq_ids.device
    T = seq_ids.shape[0]
    comp = completion_slice(prompt_len, T)

    if not torch.is_tensor(teacher_logp_full):
        teacher_logp_full = torch.tensor(teacher_logp_full, device=device)
    teacher_aligned = teacher_logp_full[1:]            # [T-1], aligned to model log-probs

    rkl = reverse_kl(student_logp_old[comp], teacher_aligned[comp])   # [C], detached
    advantage = -kl_coef * rkl

    logp_new = sequence_logprobs(student_model, seq_ids, with_grad=True)[comp]  # [C], grad
    loss = distill_loss(logp_new, student_logp_old[comp], advantage)
    return loss, rkl.mean().item()
