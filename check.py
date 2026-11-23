"""Прогон queries.json через pick и сводка исходов."""

import json
from collections import Counter
from pathlib import Path

from kino import make_client, model_name, pick


def _rows() -> list[dict]:
    path = Path("calls.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    films = json.loads(Path("films.json").read_text(encoding="utf-8"))
    queries = json.loads(Path("queries.json").read_text(encoding="utf-8"))
    known = {str(film["id"]) for film in films}
    client = make_client()
    model = model_name()
    before = len(_rows())
    counts: Counter[str] = Counter()
    must_total = 0
    must_passed = 0
    slipped = 0
    for item in queries:
        outcome, ids = pick(client, model, item["query"], films)
        counts[outcome] += 1
        slipped += sum(1 for film_id in ids if str(film_id) not in known)
        expect = item.get("expect") or {}
        must = [str(film_id) for film_id in expect.get("must") or []]
        forbid = {str(film_id) for film_id in expect.get("forbid") or []}
        if must:
            must_total += 1
            got = {str(film_id) for film_id in ids}
            if outcome == "ok" and got & set(must) and not got & forbid:
                must_passed += 1
    caught = 0
    for row in _rows()[before:]:
        rejected = row.get("rejected") or []
        if isinstance(rejected, list):
            caught += len(rejected)
    print(f"ok: {counts['ok']}")
    print(f"no_match: {counts['no_match']}")
    print(f"invalid: {counts['invalid']}")
    print(f"чужих id: {caught + slipped}")
    print(f"must пройдено: {must_passed} из {must_total}")
    others = {name: count for name, count in counts.items() if name not in {"ok", "no_match", "invalid"}}
    if others:
        print("другие исходы: " + ", ".join(f"{name} {count}" for name, count in sorted(others.items())))


if __name__ == "__main__":
    main()
