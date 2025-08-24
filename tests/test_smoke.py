from corpusloom import OllamaClient


def test_import_and_init():
    c = OllamaClient(model="gpt-oss:20b")
    assert c.model == "gpt-oss:20b"
