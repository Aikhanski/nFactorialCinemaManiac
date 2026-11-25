"""Контракт pick на заглушке. Журнал пишется во временный каталог, не в сдаваемый calls.jsonl."""

from kino import Choice, FilmPick, Filters, pick


FILMS = [
    {
        "id": "f-001",
        "title": "Матрица",
        "year": 1999,
        "minutes": 136,
        "genres": ["фантастика"],
        "countries": ["США"],
        "languages": ["английский"],
        "description": "Хакер узнаёт, что город вокруг него — симуляция.",
    },
    {
        "id": "f-002",
        "title": "Унесённые призраками",
        "year": 2001,
        "minutes": 124,
        "genres": ["мультфильм"],
        "countries": ["Япония"],
        "languages": ["японский"],
        "description": "Девочка работает в бане духов, чтобы спасти родителей.",
    },
]


class _Usage:
    prompt_tokens = 12
    completion_tokens = 4
    total_tokens = 16


class _Message:
    def __init__(self, parsed=None, refusal=None):
        self.parsed = parsed
        self.refusal = refusal


class _Choice:
    def __init__(self, message):
        self.message = message


class _Completion:
    def __init__(self, message):
        self.choices = [_Choice(message)]
        self.usage = _Usage()


class _Completions:
    def __init__(self, completion):
        self.completion = completion

    def parse(self, **kwargs):
        schema = kwargs.get("response_format")
        if schema is not None and "max_minutes" in getattr(schema, "model_fields", {}):
            return _Completion(_Message(parsed=Filters()))
        return self.completion


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, completion):
        self.chat = _Chat(_Completions(completion))


def _client(parsed):
    return FakeClient(_Completion(_Message(parsed=parsed)))


def test_alien_id_invalidates_whole_answer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(
        status="ok",
        reason="смесь",
        picks=[
            FilmPick(film_id="f-001", why="подходит"),
            FilmPick(film_id="f-001999", why="чужой"),
        ],
    )
    outcome, ids = pick(_client(parsed), "qwen3.5-9b-vlm", "что посмотреть", FILMS)
    assert outcome == "invalid"
    assert ids == []
    journal = (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    assert "f-001999" in journal
    assert '"step": "filters"' in journal
    assert '"step": "choice"' in journal


def test_duplicate_is_shown_once(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(
        status="ok",
        reason="повтор",
        picks=[
            FilmPick(film_id="f-001", why="первый"),
            FilmPick(film_id="f-001", why="ещё раз"),
        ],
    )
    outcome, ids = pick(_client(parsed), "qwen3.5-9b-vlm", "что посмотреть", FILMS)
    assert outcome == "ok"
    assert ids == ["f-001"]
    printed = capsys.readouterr().out
    assert printed.count("Матрица") == 1
    assert "1999" in printed
    assert FILMS[0]["description"] in printed


def test_no_match(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(status="no_match", reason="пусто", picks=[])
    outcome, ids = pick(_client(parsed), "qwen3.5-9b-vlm", "мультфильм 2031 года", FILMS)
    assert outcome == "no_match"
    assert ids == []
    assert capsys.readouterr().out == ""
