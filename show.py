"""Сводка журнала calls.jsonl: исходы, токены и цена."""

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    path = Path("calls.jsonl")
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    outcomes = Counter(row.get("outcome") for row in rows)
    prompt_tokens = 0
    completion_tokens = 0
    usd = 0.0
    for row in rows:
        usage = row.get("usage")
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
        price = row.get("usd")
        if isinstance(price, (int, float)):
            usd += price
    print(f"строк: {len(rows)}")
    if outcomes:
        print("исходы: " + ", ".join(f"{name} {count}" for name, count in sorted(outcomes.items(), key=lambda item: str(item[0]))))
    else:
        print("исходы: нет")
    print(f"prompt_tokens: {prompt_tokens}")
    print(f"completion_tokens: {completion_tokens}")
    print(f"usd: {usd:.4f}")


if __name__ == "__main__":
    main()
