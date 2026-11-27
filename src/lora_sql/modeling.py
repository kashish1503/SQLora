"""Loading the base model and attaching / saving / restoring adapters.

Two interchangeable backends so results can be compared head-to-head:
  * ``scratch`` - this repo's ``LoRALinear`` (src/lora_sql/lora.py)
  * ``peft``    - Hugging Face PEFT, the reference implementation
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .lora import inject_lora, load_lora_state_dict, lora_state_dict, mark_only_lora_trainable, merge_lora

SCRATCH_WEIGHTS = "lora_adapter.safetensors"
SCRATCH_CONFIG = "lora_config.json"


def load_base(model_name: str, dtype: torch.dtype, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    return model, tokenizer


def attach_adapter(model, backend: str, lora_cfg: dict):
    """Add trainable LoRA adapters. Returns the (possibly wrapped) model."""
    if backend == "scratch":
        inject_lora(
            model,
            target_modules=lora_cfg["target_modules"],
            r=lora_cfg["r"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg.get("dropout", 0.0),
            use_rslora=lora_cfg.get("use_rslora", False),
        )
        mark_only_lora_trainable(model)
        return model
    if backend == "peft":
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg.get("dropout", 0.0),
            target_modules=lora_cfg["target_modules"],
            use_rslora=lora_cfg.get("use_rslora", False),
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, config)
    raise ValueError(f"Unknown backend: {backend!r} (expected 'scratch' or 'peft')")


def save_adapter(model, backend: str, out_dir: Path, base_model: str, lora_cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if backend == "peft":
        model.save_pretrained(out_dir)
        return
    save_file(lora_state_dict(model), out_dir / SCRATCH_WEIGHTS)
    (out_dir / SCRATCH_CONFIG).write_text(json.dumps({"base_model": base_model, **lora_cfg}, indent=2))


def load_adapter(model, adapter_dir: str | Path, merge: bool = False):
    """Restore an adapter saved by either backend onto a freshly loaded base model."""
    adapter_dir = Path(adapter_dir)
    if (adapter_dir / SCRATCH_CONFIG).exists():
        cfg = json.loads((adapter_dir / SCRATCH_CONFIG).read_text())
        inject_lora(
            model,
            target_modules=cfg["target_modules"],
            r=cfg["r"],
            alpha=cfg["alpha"],
            dropout=0.0,
            use_rslora=cfg.get("use_rslora", False),
        )
        device = next(model.parameters()).device
        load_lora_state_dict(model, load_file(adapter_dir / SCRATCH_WEIGHTS, device=str(device)))
        return merge_lora(model) if merge else model
    if (adapter_dir / "adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        return model.merge_and_unload() if merge else model
    raise FileNotFoundError(f"No scratch or PEFT adapter found in {adapter_dir}")
