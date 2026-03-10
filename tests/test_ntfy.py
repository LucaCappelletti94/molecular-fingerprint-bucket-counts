import re
from urllib.error import URLError

import fp_bucket_counts.ntfy as ntfy


def test_generate_topic_returns_uuid7_like_value():
    topic = ntfy.generate_topic()

    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        topic,
    )


def test_notify_posts_message_with_headers(monkeypatch):
    captured: dict[str, object] = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(ntfy.urllib.request, "urlopen", fake_urlopen)

    ntfy.notify("demo-topic", "hello world", title="Pipeline Started", tags="rocket")

    assert captured == {
        "url": "https://ntfy.sh/demo-topic",
        "data": b"hello world",
        "method": "POST",
        "headers": {"Title": "Pipeline Started", "Tags": "rocket"},
        "timeout": 10,
    }


def test_notify_swallows_delivery_errors(monkeypatch, caplog):
    def fake_urlopen(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr(ntfy.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level("DEBUG", logger="fp_bucket_counts.ntfy"):
        ntfy.notify("demo-topic", "hello world", title="Pipeline Started", tags="rocket")

    assert "Failed to send ntfy notification" in caplog.text
