"""Text-to-SQL data: prompt formatting, tokenisation with prompt masking, collation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a SQLite database schema and a question, "
    "write one SQLite query that answers it. Reply with the SQL only."
)
IGNORE_INDEX = -100


def build_messages(schema: str, question: str, sql: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    if sql is not None:
        messages.append({"role": "assistant", "content": sql})
    return messages


def build_prompt(tokenizer, schema: str, question: str) -> str:
    return tokenizer.apply_chat_template(
        build_messages(schema, question), tokenize=False, add_generation_prompt=True
    )


def tokenize_example(tokenizer, schema: str, question: str, sql: str, max_length: int) -> dict | None:
    """Tokenise one example so the loss is computed on the answer tokens only.

    Prompt and answer are tokenised separately and concatenated, so the
    boundary between them is exact (no BPE merge across it).
    Returns None if the example does not fit in ``max_length``.
    """
    prompt = build_prompt(tokenizer, schema, question)
    full = tokenizer.apply_chat_template(build_messages(schema, question, sql), tokenize=False)
    # The answer is whatever the chat template appends after the generation prompt
    # (the SQL plus the end-of-turn token), so it adapts to any model's template.
    answer = full[len(prompt):] if full.startswith(prompt) else sql + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
    }


# ---------------------------------------------------------------------------
# b-mc2/sql-create-context: {question, context (CREATE TABLE ...), answer}
# ---------------------------------------------------------------------------


def load_create_context_splits(dataset_name: str, eval_samples: int, test_samples: int, seed: int = 42):
    """The dataset ships one split; carve out fixed test/eval sets with a seeded shuffle.

    train.py and eval_sql.py both call this, so the test set is never trained on.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train").shuffle(seed=seed)
    test = ds.select(range(test_samples))
    eval_ = ds.select(range(test_samples, test_samples + eval_samples))
    train = ds.select(range(test_samples + eval_samples, len(ds)))
    return train, eval_, test


class SQLSFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int) -> None:
        self.examples = []
        self.skipped = 0
        for row in rows:
            ex = tokenize_example(tokenizer, row["context"], row["question"], row["answer"], max_length)
            if ex is None:
                self.skipped += 1
            else:
                self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]


@dataclass
class SFTCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            input_ids[i, :n] = torch.tensor(ex["input_ids"])
            labels[i, :n] = torch.tensor(ex["labels"])
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
