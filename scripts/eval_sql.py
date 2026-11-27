"""Evaluate a base model or a LoRA adapter on text-to-SQL.

Held-out split of sql-create-context (exact match + valid-SQL rate):
    python scripts/eval_sql.py --config configs/qwen05b_sql.yaml                       # base model
    python scripts/eval_sql.py --config configs/qwen05b_sql.yaml --adapter outputs/qwen05b-r16

Spider dev set (execution accuracy against the real databases):
    python scripts/eval_sql.py --config configs/qwen05b_sql.yaml --adapter outputs/qwen05b-r16 \
        --benchmark spider --spider-dir data/spider
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lora_sql.data import build_prompt, load_create_context_splits
from lora_sql.modeling import load_adapter, load_base
from lora_sql.sql_exec import execution_match, extract_sql, is_valid_on_ddl, normalize_sql, schema_from_sqlite
from lora_sql.utils import get_device, load_config, resolve_dtype


def load_examples(args, cfg) -> list[dict]:
    if args.benchmark == "create_context":
        d = cfg["data"]
        _, _, test = load_create_context_splits(
            d["dataset"], d["eval_samples"], d["test_samples"], seed=d.get("split_seed", 42)
        )
        return [{"schema": r["context"], "question": r["question"], "gold": r["answer"]} for r in test]

    spider = Path(args.spider_dir)
    rows = json.loads((spider / "dev.json").read_text())
    schemas: dict[str, str] = {}
    examples = []
    for r in rows:
        db_path = str(spider / "database" / r["db_id"] / f"{r['db_id']}.sqlite")
        if r["db_id"] not in schemas:
            schemas[r["db_id"]] = schema_from_sqlite(db_path)
        examples.append(
            {"schema": schemas[r["db_id"]], "question": r["question"], "gold": r["query"], "db_path": db_path}
        )
    return examples


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, device) -> list[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", help="Adapter dir (scratch or PEFT). Omit to evaluate the base model.")
    ap.add_argument("--merge", action="store_true", help="Merge LoRA into the weights before inference.")
    ap.add_argument("--benchmark", choices=["create_context", "spider"], default="create_context")
    ap.add_argument("--spider-dir", help="Folder with dev.json and database/ (for --benchmark spider)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default=None, help="Where to write predictions.jsonl")
    args = ap.parse_args()
    if args.benchmark == "spider" and not args.spider_dir:
        ap.error("--spider-dir is required for --benchmark spider")

    cfg = load_config(args.config)
    device = get_device()
    model, tokenizer = load_base(cfg["model_name"], resolve_dtype(cfg.get("dtype", "auto"), device), device)
    tokenizer.padding_side = "left"  # decoder-only batched generation needs left padding
    if args.adapter:
        model = load_adapter(model, args.adapter, merge=args.merge)
    model.eval()

    examples = load_examples(args, cfg)[: args.limit]
    out_path = Path(args.out or (Path(args.adapter or "outputs/base") / f"predictions_{args.benchmark}.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = exact = valid = exec_hits = exec_total = 0
    with out_path.open("w") as f:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            prompts = [build_prompt(tokenizer, ex["schema"], ex["question"]) for ex in batch]
            for ex, raw in zip(batch, generate(model, tokenizer, prompts, args.max_new_tokens, device)):
                pred = extract_sql(raw)
                rec = {"question": ex["question"], "gold": ex["gold"], "pred": pred}
                rec["exact_match"] = normalize_sql(pred) == normalize_sql(ex["gold"])
                if "db_path" in ex:
                    res = execution_match(ex["db_path"], ex["gold"], pred)
                    rec.update(valid=res["valid"], exec_match=res["match"], error=res["error"])
                    if res["match"] is not None:
                        exec_total += 1
                        exec_hits += int(res["match"])
                else:
                    rec["valid"] = is_valid_on_ddl(ex["schema"], pred)
                n += 1
                exact += int(rec["exact_match"])
                valid += int(rec["valid"])
                f.write(json.dumps(rec) + "\n")
            print(f"{n}/{len(examples)}  exact={exact / n:.3f}  valid={valid / n:.3f}", flush=True)

    metrics = {
        "model": cfg["model_name"],
        "adapter": args.adapter,
        "benchmark": args.benchmark,
        "n": n,
        "exact_match": exact / n,
        "valid_sql": valid / n,
    }
    if exec_total:
        metrics["execution_accuracy"] = exec_hits / exec_total
    print(json.dumps(metrics, indent=2))
    out_path.with_name(f"metrics_{args.benchmark}.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
