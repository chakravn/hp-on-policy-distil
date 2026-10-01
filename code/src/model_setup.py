"""Student model / optimizer construction for the distillation trainer.

THE BUG THIS FILE EXISTS TO PREVENT
-----------------------------------
The original trainer did::

    student = AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-5)

i.e. AdamW writing updates directly into **bfloat16** parameters. bf16 has an 8-bit
mantissa, so its relative resolution is about 2^-8 ~= 0.4%. An Adam step has magnitude
~= lr = 1e-5 *absolute*, while a typical Qwen weight is ~1e-2, making the relative update
~1e-3 -- an order of magnitude below what bf16 can represent. Momentum accumulated
correctly and then the write-back rounded away to nothing. The model barely moved, while
`teacher_kl` still drifted down from the few large-gradient tensors that did clear the
threshold, so the run *looked* healthy.

`assert_trainable_dtype` now fails loudly on any trainable parameter that is not fp32.

WHY LoRA IS THE DEFAULT AT 4B
-----------------------------
The obvious fix -- keep everything in fp32 -- costs, for a 4B student: 16 GB params +
16 GB grads + 32 GB Adam state = ~68 GB. That does not fit a 24 GB A10G and would need
FSDP across 4 GPUs, plus `summon_full_params` wiring around `.generate()` that this
codebase does not have. LoRA on a frozen bf16 base costs ~13 GB and fits one GPU:

    | strategy                        | total (+act.) | fits 1x A10G |
    | full fp32 AdamW                 | ~68 GB        | no           |
    | fp32 + 8-bit Adam               | ~44 GB        | no           |
    | LoRA r=32, bf16 base, fp32 adapter | ~13 GB     | yes          |

LoRA also fixes the precision bug in the right place: the frozen base stays bf16 (it is
never updated, so rounding cannot hurt it) while the small trainable adapter is fp32,
where a 1e-5 update resolves cleanly. And a LoRA adapter can be hot-swapped into a vLLM
sampler (`sampler.VLLMSampler.sync_adapter`), which removes the need for a full-weight
synchronisation protocol between trainer and sampler.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import torch

# Qwen-family attention + MLP projections. `all-linear` is the safer generic choice, but
# naming them keeps the adapter small and the target set reproducible across versions.
DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def load_tokenizer(model_ref: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_student(
    model_ref: str,
    device: str = "cuda:0",
    precision: str = "lora",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    lora_targets: Iterable[str] = DEFAULT_LORA_TARGETS,
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = False,
    adapter_path: str | None = None,
):
    """Load the student for training.

    `precision`:
      * ``lora``      -- frozen bf16 base + fp32 LoRA adapter (default; fits one 24 GB GPU)
      * ``full-fp32`` -- every parameter fp32 and trainable (needs ~68 GB at 4B, so FSDP)
    `adapter_path` (``INIT_ADAPTER``) starts from an existing adapter instead of the base
    weights -- how a crashed run resumes from a ``${RUN_DIR}/student-step<N>`` checkpoint.
    """
    from transformers import AutoModelForCausalLM

    base_dtype = torch.bfloat16 if precision == "lora" else torch.float32
    # FlashAttention2 cuts attention time by 2-4x at seq_len ~1000 on Qwen3.5. Try FA2 first;
    # fall back to SDPA if the wheel is not installed, then to eager as a last resort so this
    # keeps loading on machines without flash-attn.
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_ref, torch_dtype=base_dtype,
                trust_remote_code=trust_remote_code, attn_implementation=attn_impl,
            )
            break
        except (ImportError, ValueError):
            continue
    else:  # pragma: no cover - defensive; from_pretrained normally succeeds with eager
        model = AutoModelForCausalLM.from_pretrained(
            model_ref, torch_dtype=base_dtype, trust_remote_code=trust_remote_code,
        )
    # Checkpointing needs the cache off during training; generation re-enables it itself.
    model.config.use_cache = not gradient_checkpointing

    if precision == "lora":
        from peft import LoraConfig, PeftModel, get_peft_model

        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        else:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=list(lora_targets),
                ),
            )
        # The adapter must be fp32 or the optimizer update rounds away (see module docstring).
        for _, param in model.named_parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()
    elif precision != "full-fp32":
        raise ValueError(f"unknown precision {precision!r}; use 'lora' or 'full-fp32'")

    model = model.to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            # Without this, checkpointing + a frozen embedding layer yields no grad path.
            model.enable_input_require_grads()
    model.train()
    assert_trainable_dtype(model)
    return model


def assert_trainable_dtype(model, expected: torch.dtype = torch.float32) -> None:
    """Fail unless every trainable parameter is `expected` dtype.

    Guards the exact class of bug described in the module docstring: a low-precision
    parameter silently swallowing small optimizer updates.
    """
    offenders = [
        (name, str(p.dtype))
        for name, p in model.named_parameters()
        if p.requires_grad and p.dtype != expected
    ]
    if offenders:
        shown = ", ".join(f"{n}:{d}" for n, d in offenders[:5])
        raise RuntimeError(
            f"{len(offenders)} trainable parameters are not {expected} (e.g. {shown}). "
            "Optimizer updates of size ~lr would round away below this dtype's resolution "
            "and training would silently stall."
        )


def trainable_parameter_report(model) -> dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
    }


def build_optimizer(model, lr: float, weight_decay: float = 0.0, adam8bit: bool = False):
    """AdamW over the trainable parameters, optionally with 8-bit state.

    Weight decay is applied only to matrices, never to norms/biases -- decaying a LayerNorm
    gain is a well-known way to quietly degrade a model.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 or "norm" in name.lower() else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if adam8bit:
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.05,
                    min_lr_ratio: float = 0.1):
    """Linear warm-up then cosine decay to `min_lr_ratio` of peak.

    The original run used a constant 1e-5 with no warm-up. With only a few hundred steps
    that wastes the early steps on a badly-scaled first update; with a few thousand it
    leaves the model oscillating instead of settling.
    """
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_student(
    model, tokenizer, out_dir: str | Path, merge: bool = False,
    trust_remote_code: bool = False,
) -> Path:
    """Save the student. For LoRA this writes the adapter; `merge=True` writes full weights.

    The eval and the vLLM sampler want different things: the sampler hot-loads an *adapter*,
    while `eval_student.py` compares a pristine base against a standalone checkpoint, so the
    final save is merged.

    MERGE PRECISION. `merge_and_unload` computes `W = W_base + B @ A * scaling` in the base's
    dtype. With a bf16 base and an fp32 adapter, the add rounds the adapter's contribution
    to bf16 -- silently discarding much of what `assert_trainable_dtype` was built to
    preserve. So the merge runs in fp32, then downcasts to bf16 for storage (the eval loads
    in bf16 anyway).

    MERGE FROM A FRESH fp32 BASE (same path as `merge_debug.py`). Rather than upcasting the
    live training model, the adapter is written to ``<out_dir>-adapter`` and merged into a
    base reloaded from its source straight in fp32. That (a) keeps tensors the checkpoint
    stores above bf16 at full precision until the single final rounding, (b) saves the
    base's pristine `config.json` instead of the training one (`use_cache=False` from
    gradient checkpointing), and (c) leaves the live PeftModel untouched and trainable.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if merge and hasattr(model, "peft_config"):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        adapter_dir = out.parent / f"{out.name}-adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        base_ref = model.peft_config["default"].base_model_name_or_path

        # MERGE ON CPU. The fp32 4B base is 16 GiB -- more than the A10G has spare next to
        # the live training model. Node RAM is 384 GiB, so CPU is comfortable.
        base = AutoModelForCausalLM.from_pretrained(
            base_ref, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged = merged.to(torch.bfloat16).eval()   # the fp32 merge result is baked in
        merged.config.torch_dtype = torch.bfloat16
        merged.save_pretrained(out, safe_serialization=True)
        del base, merged
    else:
        model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    return out


def memory_report(device: str = "cuda:0") -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    idx = torch.device(device).index or 0
    return {
        "gpu_alloc_gb": round(torch.cuda.memory_allocated(idx) / 2**30, 2),
        "gpu_reserved_gb": round(torch.cuda.memory_reserved(idx) / 2**30, 2),
        "gpu_total_gb": round(torch.cuda.get_device_properties(idx).total_memory / 2**30, 2),
    }
