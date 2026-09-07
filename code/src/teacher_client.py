"""Teacher scoring client for on-policy distillation.

The teacher's only job is to return, for a sequence the *student* generated, the
teacher's log-probability of every token. That single scalar per token is all we
need to compute the per-token reverse KL that supervises the student, and it is
cheap enough to send over the wire -- which is exactly why the teacher can live
on its own GPU pool (a vLLM inference Service) instead of in the trainer process.

Two backends, same interface (`score(token_ids, ...) -> list[float]`):

  * RemoteTeacher  -- talks to a vLLM OpenAI-compatible server (the HyperPod
                      two-pool setup). Uses `echo=True, logprobs=0` to read back
                      the log-prob of each *prompt* token.
  * LocalTeacher   -- loads the teacher in-process with transformers. Zero infra,
                      used as the notebook's de-risking fallback and for the unit
                      check that the remote path returns the same numbers.

Alignment convention (shared by both backends): the returned list has the same
length as `token_ids`; element t is  log p_teacher(token_ids[t] | token_ids[:t]).
Element 0 is undefined (no context) and is returned as 0.0 -- it is always masked
out downstream because we only score completion tokens (t >= prompt_len).
"""

from __future__ import annotations

from typing import List, Sequence


class RemoteTeacher:
    """Score token sequences against a vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 120.0):
        # Imported lazily so the module is importable without the openai package.
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout)

    def score(self, token_ids: Sequence[int]) -> List[float]:
        """Return per-token teacher log-probs aligned to `token_ids` (see module docstring)."""
        resp = self.client.completions.create(
            model=self.model,
            prompt=list(token_ids),  # vLLM accepts a list of token ids as the prompt
            max_tokens=1,            # must be >= 1; we ignore the generated token
            echo=True,               # echo the prompt so we get its per-token log-probs
            logprobs=0,              # 0 => just the actual token's log-prob, no top-k
            temperature=0.0,
        )
        # token_logprobs is aligned to the echoed prompt tokens (index 0 is None),
        # possibly followed by the one generated token -- keep only the prompt span.
        tlp = resp.choices[0].logprobs.token_logprobs[: len(token_ids)]
        return [0.0 if v is None else float(v) for v in tlp]

    def score_batch(self, batch: Sequence[Sequence[int]]) -> List[List[float]]:
        return [self.score(ids) for ids in batch]


class LocalTeacher:
    """Score token sequences with an in-process transformers model (fallback / unit check)."""

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype)
        ).to(device).eval()

    @property
    def _no_grad(self):
        return self.torch.no_grad()

    def score(self, token_ids: Sequence[int]) -> List[float]:
        torch = self.torch
        ids = torch.tensor(token_ids, device=self.device).unsqueeze(0)  # [1, T]
        with self._no_grad:
            logits = self.model(ids).logits[0]                          # [T, V]
            logp = logits[:-1].log_softmax(-1)                          # predict tokens 1..T-1
            tok_logp = logp.gather(-1, ids[0, 1:, None]).squeeze(-1)    # [T-1]
        # Prepend 0.0 for position 0 so the output aligns 1:1 with token_ids.
        return [0.0] + tok_logp.float().cpu().tolist()

    def score_batch(self, batch: Sequence[Sequence[int]]) -> List[List[float]]:
        return [self.score(ids) for ids in batch]


def make_teacher(mode: str, **kwargs):
    """Factory: mode='remote' -> RemoteTeacher, mode='local' -> LocalTeacher."""
    if mode == "remote":
        return RemoteTeacher(base_url=kwargs["base_url"], model=kwargs["model"])
    if mode == "local":
        return LocalTeacher(model_name=kwargs["model_name"], device=kwargs.get("device", "cuda"))
    raise ValueError(f"unknown teacher mode: {mode!r}")
