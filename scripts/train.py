"""Fine-tune a causal LM for text-to-SQL with LoRA, using a hand-written PyTorch loop.

    python scripts/train.py --config configs/qwen05b_sql.yaml
    python scripts/train.py --config configs/qwen05b_sql.yaml --override lora.r=64 backend=peft
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lora_sql.data import SFTCollator, SQLSFTDataset, load_create_context_splits
from lora_sql.lora import count_parameters
from lora_sql.modeling import attach_adapter, load_base, save_adapter
from lora_sql.utils import get_device, load_config, peak_memory_gb, resolve_dtype, set_seed


def build_optimizer(model, train_cfg: dict) -> torch.optim.Optimizer:
    """AdamW over trainable params. With ``lora_plus_ratio`` > 1, B gets a higher LR (LoRA+)."""
    lr = float(train_cfg["lr"])
    ratio = float(train_cfg.get("lora_plus_ratio", 1.0))
    wd = float(train_cfg.get("weight_decay", 0.0))
    params_b, params_other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (params_b if "lora_B" in name else params_other).append(p)
    groups = [
        {"params": params_other, "lr": lr},
        {"params": params_b, "lr": lr * ratio},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=wd)


@torch.no_grad()
def evaluate_loss(model, loader, device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        n_tokens = int((batch["labels"] != -100).sum())
        total_loss += model(**batch).loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="e.g. lora.r=32 train.lr=1e-4")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    tcfg, dcfg, lcfg = cfg["train"], cfg["data"], cfg["lora"]
    set_seed(tcfg["seed"])
    device = get_device()
    dtype = resolve_dtype(cfg.get("dtype", "auto"), device)
    out_dir = Path(tcfg["output_dir"])
    print(f"device={device} dtype={dtype} backend={cfg['backend']}")

    # ---------------------------------------------------------------- model
    model, tokenizer = load_base(cfg["model_name"], dtype, device)
    model.config.use_cache = False
    if tcfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Embeddings are frozen, so make their output require grad or checkpointed
        # blocks would see no grad-requiring inputs and skip backprop entirely.
        model.enable_input_require_grads()
    model = attach_adapter(model, cfg["backend"], lcfg)
    trainable, total = count_parameters(model)
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    # ----------------------------------------------------------------- data
    train_rows, eval_rows, _ = load_create_context_splits(
        dcfg["dataset"], dcfg["eval_samples"], dcfg["test_samples"], seed=dcfg.get("split_seed", 42)
    )
    if dcfg.get("train_samples"):
        train_rows = train_rows.select(range(min(dcfg["train_samples"], len(train_rows))))
    train_ds = SQLSFTDataset(train_rows, tokenizer, dcfg["max_length"])
    eval_ds = SQLSFTDataset(eval_rows, tokenizer, dcfg["max_length"])
    print(f"train={len(train_ds)} (skipped {train_ds.skipped} too long), eval={len(eval_ds)}")

    collate = SFTCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=tcfg["batch_size"], collate_fn=collate)

    # ------------------------------------------------------------ optimiser
    accum = tcfg["grad_accum"]
    total_steps = math.ceil(len(train_loader) / accum) * tcfg["epochs"]
    optimizer = build_optimizer(model, tcfg)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    wandb_run = None
    if tcfg.get("wandb_project"):
        import wandb

        wandb_run = wandb.init(project=tcfg["wandb_project"], config=cfg, name=out_dir.name)

    # ----------------------------------------------------------------- loop
    model.train()
    step, tokens_seen, history = 0, 0, []
    running_loss = 0.0
    t0 = time.perf_counter()
    for epoch in range(tcfg["epochs"]):
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()
            running_loss += loss.item()
            tokens_seen += int(batch["attention_mask"].sum())

            is_last_batch = i + 1 == len(train_loader)
            if (i + 1) % accum != 0 and not is_last_batch:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % tcfg["log_every"] == 0:
                elapsed = time.perf_counter() - t0
                record = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": running_loss / tcfg["log_every"],
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens_per_s": tokens_seen / elapsed,
                }
                running_loss = 0.0
                print(json.dumps(record))
                history.append(record)
                if wandb_run:
                    wandb_run.log(record, step=step)

            if step % tcfg["eval_every"] == 0 or step == total_steps:
                eval_loss = evaluate_loss(model, eval_loader, device)
                print(json.dumps({"step": step, "eval_loss": eval_loss}))
                history.append({"step": step, "eval_loss": eval_loss})
                if wandb_run:
                    wandb_run.log({"eval_loss": eval_loss}, step=step)

    # ----------------------------------------------------------------- save
    save_adapter(model, cfg["backend"], out_dir, cfg["model_name"], lcfg)
    summary = {
        "backend": cfg["backend"],
        "trainable_params": trainable,
        "total_params": total,
        "train_time_s": time.perf_counter() - t0,
        "peak_memory_gb": peak_memory_gb(device),
        "final_eval_loss": next((h["eval_loss"] for h in reversed(history) if "eval_loss" in h), None),
        "config": cfg,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history))
    print(f"saved adapter + summary to {out_dir}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
