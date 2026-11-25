"""Страна проверяется кодом. Оба клиента — заглушки, без сети и без ключей."""

import json

import openai

from kino import Choice, FilmPick, Filters, films_for_query, pick, required_country, run_models


CATALOG = [
    {
        "id": "f-jp",
        "title": "Мой сосед Тоторо",
        "year": 1988,
        "minutes": 86,
        "genres": ["мультфильм", "семейный"],
        "countries": ["Япония"],
        "languages": ["японский"],
        "description": "Две сестры встречают лесного духа.",
    },
    {
        "id": "f-kz",
        "title": "Жёлтая кошка",
        "year": 2020,
        "minutes": 90,
        "genres": ["комедия", "драма"],
        "countries": ["Казахстан"],
        "languages": ["русский", "казахский"],
        "description": "Бывший заключённый хочет открыть кинотеатр в степи.",
    },
    {
        "id": "f-us",
        "title": "Матрица",
        "year": 1999,
        "minutes": 136,
        "genres": ["фантастика"],
        "countries": ["США"],
        "languages": ["английский"],
        "description": "Хакер узнаёт, что город — симуляция.",
    },
]


class _Message:
    def __init__(self, parsed):
        self.parsed = parsed
        self.refusal = None


class _ChoiceWrap:
    def __init__(self, message):
        self.message = message


class _Completion:
    def __init__(self, parsed):
        self.choices = [_ChoiceWrap(_Message(parsed))]
        self.usage = None


class RecordingClient:
    def __init__(self, parsed=None, error=None, filters=None):
        self.parsed = parsed
        self.error = error
        self.filters = Filters() if filters is None else filters
        self.calls = []
        self.chat = self
        self.completions = self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        schema = kwargs.get("response_format")
        if schema is not None and "max_minutes" in getattr(schema, "model_fields", {}):
            return _Completion(self.filters)
        return _Completion(self.parsed)


def _both(parsed):
    return RecordingClient(parsed), RecordingClient(parsed)


def _titles(client: RecordingClient) -> str:
    for call in client.calls:
        schema = call.get("response_format")
        if schema is not None and "picks" in getattr(schema, "model_fields", {}):
            return call["messages"][0]["content"]
    return ""


def test_kazakh_request_drops_japan(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(
        status="ok",
        reason="оба",
        picks=[
            FilmPick(film_id="f-jp", why="семья"),
            FilmPick(film_id="f-kz", why="степь"),
        ],
    )
    client = RecordingClient(parsed)
    outcome, ids = pick(client, "local-model", "Казахстанский семейный фильм", CATALOG)
    assert outcome == "ok"
    assert ids == ["f-kz"]
    printed = capsys.readouterr().out
    assert "Тоторо" not in printed
    assert "Жёлтая кошка" in printed
    assert "Мой сосед Тоторо" not in _titles(client)


def test_japanese_request_drops_kazakhstan(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(
        status="ok",
        reason="оба",
        picks=[
            FilmPick(film_id="f-kz", why="степь"),
            FilmPick(film_id="f-jp", why="лес"),
        ],
    )
    client = RecordingClient(parsed)
    outcome, ids = pick(client, "local-model", "Японский мультфильм", CATALOG)
    assert outcome == "ok"
    assert ids == ["f-jp"]
    assert "Жёлтая кошка" not in capsys.readouterr().out
    assert "Жёлтая кошка" not in _titles(client)


def test_query_without_country_keeps_catalog(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(status="ok", reason="все", picks=[FilmPick(film_id="f-jp", why="лес")])
    client = RecordingClient(parsed)
    outcome, ids = pick(client, "local-model", "что посмотреть вечером", CATALOG)
    assert outcome == "ok"
    assert ids == ["f-jp"]
    prompt = _titles(client)
    assert "Мой сосед Тоторо" in prompt
    assert "Жёлтая кошка" in prompt
    assert "Матрица" in prompt
    for query in (
        "фильм как Интерстеллар, но не про космос",
        "доброе кино про Алматы, только не мелодрама",
        "что посмотреть с мамой на казахском, до двух часов",
        "комедия для вечера с друзьями, без ужасов",
        "қазақша отбасылық фильм",
        "что-нибудь старое и спокойное, кроме вестернов",
    ):
        assert required_country(query) is None
        assert films_for_query(query, CATALOG) == CATALOG


def test_missing_country_returns_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    only_japan = [CATALOG[0]]
    client = RecordingClient(Choice(status="ok", reason="нет", picks=[FilmPick(film_id="f-jp", why="лес")]))
    outcome, ids = pick(client, "local-model", "Казахстанский семейный фильм", only_japan)
    assert outcome == "no_match"
    assert ids == []
    assert len(client.calls) == 1
    assert capsys.readouterr().out == ""


def test_alien_id_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(status="ok", reason="чужой", picks=[FilmPick(film_id="нет-такого", why="выдумка")])
    outcome, ids = pick(RecordingClient(parsed), "local-model", "что посмотреть вечером", CATALOG)
    assert outcome == "invalid"
    assert ids == []
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any("нет-такого" in (row.get("rejected") or []) for row in rows)


def test_wrong_country_from_model_is_dropped(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(status="ok", reason="япония", picks=[FilmPick(film_id="f-jp", why="семья")])
    outcome, ids = pick(RecordingClient(parsed), "local-model", "Фильм из Казахстана без комедии", CATALOG)
    assert outcome == "no_match"
    assert ids == []
    assert "Тоторо" not in capsys.readouterr().out


def test_local_failure_does_not_stop_cloud(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(status="ok", reason="да", picks=[FilmPick(film_id="f-kz", why="степь")])
    local = RecordingClient(error=RuntimeError("local down"))
    cloud = RecordingClient(parsed)
    results = run_models(
        "Казахстанский семейный фильм",
        CATALOG,
        [("local", local, "local-model"), ("cloud", cloud, "cloud-model")],
    )
    assert local.calls
    assert results[0] == ("error", [])
    assert results[1] == ("ok", ["f-kz"])
    assert cloud.calls[0]["model"] == "cloud-model"


def test_cloud_failure_does_not_stop_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(status="ok", reason="да", picks=[FilmPick(film_id="f-kz", why="степь")])
    local = RecordingClient(parsed)
    cloud = RecordingClient(error=openai.APIConnectionError(request=None))
    results = run_models(
        "Казахстанский семейный фильм",
        CATALOG,
        [("local", local, "local-model"), ("cloud", cloud, "cloud-model")],
    )
    assert results[0] == ("ok", ["f-kz"])
    assert results[1][0] == "no_connection"
    assert results[1][1] == []


def test_both_models_get_the_same_query_and_catalog(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parsed = Choice(status="ok", reason="да", picks=[FilmPick(film_id="f-kz", why="степь")])
    local, cloud = _both(parsed)
    query = "Американская комедия"
    run_models(query, CATALOG, [("local", local, "local-model"), ("cloud", cloud, "cloud-model")])
    assert local.calls[0]["messages"] == cloud.calls[0]["messages"]
    assert _titles(local) == _titles(cloud)
    prompt = _titles(local)
    assert query in prompt
    assert "Матрица" in prompt
    assert "Мой сосед Тоторо" not in prompt
    assert "Жёлтая кошка" not in prompt
