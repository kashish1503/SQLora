"""LoRA implemented from scratch in plain PyTorch.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).

For a frozen linear layer  y = W x + b,  LoRA learns a low-rank update

    y = W x + b + (alpha / r) * B A x

with A in R^{r x in}, B in R^{out x r}, and r << min(in, out).
B is zero-initialised, so at step 0 the adapted model is exactly the base model.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps an existing ``nn.Linear`` and adds a trainable low-rank update.

    Args:
        base: the pretrained linear layer to adapt (its weights get frozen).
        r: rank of the update.
        alpha: scaling numerator. Effective scale is ``alpha / r``
            (or ``alpha / sqrt(r)`` with ``use_rslora``).
        dropout: dropout applied to the input of the LoRA branch only.
        use_rslora: rank-stabilised scaling (Kalajdzievski, 2023), which keeps
            the update magnitude stable as ``r`` grows.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_rslora: bool = False,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}")

        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / math.sqrt(r) if use_rslora else alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Adapter weights are kept in fp32 even when the base model is bf16/fp16:
        # the update is small, and low-precision optimiser steps on it are lossy.
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=device))
        # Same init as the reference implementation / HF PEFT.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.merged = False
        for p in self.base.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        h = self.lora_dropout(x).to(self.lora_A.dtype)
        update = F.linear(F.linear(h, self.lora_A), self.lora_B) * self.scaling
        return out + update.to(out.dtype)

    # ------------------------------------------------------------ merge / unmerge
    def delta_weight(self) -> torch.Tensor:
        """The dense update  ΔW = scaling * B @ A  (shape: out x in)."""
        return (self.lora_B @ self.lora_A) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold ΔW into the base weight so inference costs nothing extra."""
        if self.merged:
            return
        w = self.base.weight
        if not w.is_floating_point():
            raise TypeError("Cannot merge into a quantized weight; dequantize first.")
        w.add_(self.delta_weight().to(w.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        w = self.base.weight
        w.sub_(self.delta_weight().to(w.dtype))
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in={self.base.in_features}, out={self.base.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4g}, merged={self.merged}"
        )


# ---------------------------------------------------------------------------
# Model surgery helpers
# ---------------------------------------------------------------------------


def _is_target(full_name: str, target_modules: Iterable[str]) -> bool:
    """Match on the last path component, e.g. 'model.layers.0.self_attn.q_proj' -> 'q_proj'."""
    return full_name.rsplit(".", 1)[-1] in set(target_modules)


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str],
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    use_rslora: bool = False,
) -> list[str]:
    """Replace every matching ``nn.Linear`` in ``model`` with a ``LoRALinear``, in place.

    Returns the fully-qualified names of the replaced layers.
    """
    target_modules = list(target_modules)
    replaced: list[str] = []
    # Snapshot the module list first: we mutate the tree while walking it.
    for parent_name, parent in list(model.named_modules()):
        if isinstance(parent, LoRALinear):
            continue
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and _is_target(full_name, target_modules):
                setattr(parent, child_name, LoRALinear(child, r, alpha, dropout, use_rslora))
                replaced.append(full_name)
    if not replaced:
        raise ValueError(f"No nn.Linear layers matched target_modules={target_modules}")
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze everything except the LoRA A/B matrices."""
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)


def lora_parameters(model: nn.Module, which: str = "both") -> list[nn.Parameter]:
    """Return LoRA params; ``which`` in {'A', 'B', 'both'} (used for LoRA+ param groups)."""
    keys = {"A": ("lora_A",), "B": ("lora_B",), "both": ("lora_A", "lora_B")}[which]
    return [p for n, p in model.named_parameters() if n.rsplit(".", 1)[-1] in keys]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter weights: this is the whole checkpoint (a few MB)."""
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise KeyError(f"Unexpected keys in LoRA checkpoint: {unexpected[:5]} ...")
    missing_lora = [k for k in missing if "lora_" in k]
    if missing_lora:
        raise KeyError(f"LoRA checkpoint is missing keys: {missing_lora[:5]} ...")


def merge_lora(model: nn.Module, unwrap: bool = True) -> nn.Module:
    """Merge every adapter into its base layer.

    With ``unwrap=True`` the ``LoRALinear`` wrappers are replaced by the plain
    (now updated) ``nn.Linear``, giving back a standard architecture that can
    be saved with ``save_pretrained`` and served by anything.
    """
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                child.merge()
                if unwrap:
                    setattr(parent, child_name, child.base)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
