from src.preprocessing.text import normalize_text


def test_normalize_text_replaces_url_and_number():
    result = normalize_text("Visit https://example.com and call 12345")
    assert "URL".lower() in result
    assert "NUMBER".lower() in result
