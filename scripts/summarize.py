"""Collect train_summary.json + metrics_*.json from run folders into a Markdown table.

    python scripts/summarize.py outputs/sweep
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(root: str) -> None:
    rows = []
    for run in sorted(p for p in Path(root).iterdir() if (p / "train_summary.json").exists()):
        s = json.loads((run / "train_summary.json").read_text())
        m_path = run / "metrics_create_context.json"
        m = json.loads(m_path.read_text()) if m_path.exists() else {}
        lora = s["config"]["lora"]
        rows.append(
            [
                run.name,
                s["backend"],
                str(lora["r"]),
                f"{s['trainable_params']:,}",
                f"{100 * s['trainable_params'] / s['total_params']:.3f}%",
                f"{s['final_eval_loss']:.4f}" if s["final_eval_loss"] is not None else "-",
                f"{m['exact_match']:.3f}" if m else "-",
                f"{m['valid_sql']:.3f}" if m else "-",
                f"{s['peak_memory_gb']:.2f}" if s["peak_memory_gb"] else "-",
                f"{s['train_time_s'] / 60:.1f}",
            ]
        )
    header = ["run", "backend", "r", "trainable", "% of model", "eval loss", "exact match", "valid SQL", "peak GB", "min"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs")
