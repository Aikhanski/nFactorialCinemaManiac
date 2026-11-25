"""Три прогона queries.json на каждой настроенной модели."""

import argparse
import json
from pathlib import Path

from kino import _build_client, bind_model_kind, cloud_config, local_config, pick

RUNS = 3


def _rows() -> list[dict]:
    path = Path("calls.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _passed(outcome: str, ids: list[str], expect: dict) -> bool:
    if expect.get("no_match"):
        return outcome == "no_match"
    must = {str(film_id) for film_id in expect.get("must") or []}
    forbid = {str(film_id) for film_id in expect.get("forbid") or []}
    got = {str(film_id) for film_id in ids}
    return outcome == "ok" and bool(got & must) and not (got & forbid)


def _run(kind: str, base_url: str, api_key: str, model: str, films: list[dict], queries: list[dict], runs: int) -> None:
    known = {str(film["id"]) for film in films}
    client = _build_client(base_url, api_key, local=kind == "local")
    before = len(_rows())
    passed = 0
    for _ in range(runs):
        for item in queries:
            with bind_model_kind(kind):
                outcome, ids = pick(client, model, item["query"], films)
            if _passed(outcome, ids, item["expect"]):
                passed += 1
    fresh = _rows()[before:]
    choice = [row for row in fresh if row.get("step") != "filters"]
    aliens = 0
    violations = 0
    seconds = 0.0
    usd = 0.0
    for row in fresh:
        seconds += float(row.get("seconds") or 0)
        price = row.get("usd")
        if isinstance(price, (int, float)):
            usd += price
        for film_id in row.get("rejected") or []:
            if str(film_id) in known:
                violations += 1
            else:
                aliens += 1
        for film_id in row.get("ids") or []:
            if str(film_id) not in known:
                aliens += 1
    total = runs * len(queries)
    print(f"{model}: пройдено {passed} из {total}, доля {passed / total:.0%}")
    print(f"  чужих id: {aliens}")
    print(f"  нарушения условий: {violations}")
    print(f"  секунды: {seconds:.1f}")
    print(f"  usd: {usd:.4f}")
    print(f"  строк выбора: {len(choice)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон queries.json")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--only", choices=("local", "cloud", "both"), default="both")
    args = parser.parse_args()
    films = json.loads(Path("films.json").read_text(encoding="utf-8"))
    queries = json.loads(Path("queries.json").read_text(encoding="utf-8"))
    if args.only in {"local", "both"}:
        local_base, local_key, local_model = local_config()
        _run("local", local_base, local_key, local_model, films, queries, args.runs)
    if args.only in {"cloud", "both"}:
        cloud_base, cloud_key, cloud_model = cloud_config()
        if cloud_model and cloud_key:
            _run("cloud", cloud_base, cloud_key, cloud_model, films, queries, args.runs)
        else:
            print("облачная модель: не настроена")


if __name__ == "__main__":
    main()
