"""Условия из схемы Filters применяет код, а не модель."""

import json

from kino import Choice, FilmPick, Filters, apply_filters, pick


FILMS = [
    {
        "id": "long",
        "title": "Длинная картина",
        "year": 1980,
        "minutes": 200,
        "genres": ["драма"],
        "countries": ["Япония"],
        "languages": ["японский"],
        "description": "История идёт больше трёх часов.",
    },
    {
        "id": "short",
        "title": "Короткая картина",
        "year": 2019,
        "minutes": 80,
        "genres": ["комедия"],
        "countries": ["США"],
        "languages": ["английский"],
        "description": "История успевает закончиться за вечер.",
    },
]


class _Message:
    def __init__(self, parsed):
        self.parsed = parsed
        self.refusal = None


class _Completion:
    def __init__(self, parsed):
        self.choices = [type("Choice", (), {"message": _Message(parsed)})()]
        self.usage = None


class Client:
    def __init__(self, filters, parsed):
        self.filters = filters
        self.parsed = parsed
        self.calls = []
        self.chat = self
        self.completions = self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs.get("response_format")
        if "max_minutes" in getattr(schema, "model_fields", {}):
            return _Completion(self.filters)
        return _Completion(self.parsed)


def test_long_film_is_removed_before_choice(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(
        status="ok",
        reason="оба",
        picks=[
            FilmPick(film_id="long", why="долгий"),
            FilmPick(film_id="short", why="короткий"),
        ],
    )
    client = Client(Filters(max_minutes=100), parsed)
    outcome, ids = pick(client, "local-model", "кино до ста минут", FILMS)
    assert outcome == "ok"
    assert ids == ["short"]
    choice = client.calls[1]["messages"][0]["content"]
    assert "Короткая картина" in choice
    assert "Длинная картина" not in choice
    assert "Длинная картина" not in capsys.readouterr().out
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in rows] == ["filters", "choice"]
    assert "long" in rows[1]["rejected"]


def test_empty_filters_keep_every_film():
    kept = apply_filters(FILMS, Filters())
    assert kept == FILMS


def test_excluded_genre_drops_film():
    kept = apply_filters(FILMS, Filters(exclude_genres=["драма"]))
    assert [film["id"] for film in kept] == ["short"]
