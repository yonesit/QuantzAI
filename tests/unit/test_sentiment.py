"""
Unit-Tests fuer SentimentFeature.
FinBERT/transformers wird NICHT geladen – ein Fake-SentimentModel
implementiert dasselbe Protocol. RSS-HTTP-Requests werden gemockt.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.data.sentiment import (
    SentimentFeature,
    NewsArticle,
    SentimentScore,
    FinBERTModel,
)


# ---------------------------------------------------------------------------
# Fake-Modell (erfuellt das SentimentModel-Protocol ohne ML-Libraries)
# ---------------------------------------------------------------------------

class FakeSentimentModel:
    """Gibt fuer jeden Text einen festen oder gemappten Score zurueck."""

    def __init__(self, fixed_score: float = 0.5, text_score_map: dict | None = None):
        self.fixed_score = fixed_score
        self.text_score_map = text_score_map or {}
        self.calls: list[list[str]] = []

    def predict(self, texts: list[str]) -> list[float]:
        self.calls.append(texts)
        return [self.text_score_map.get(t, self.fixed_score) for t in texts]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _rss_xml(items: list[dict]) -> str:
    """Baut ein minimales RSS-XML aus einer Liste von dicts."""
    entries = ""
    for item in items:
        entries += f"""
        <item>
            <title>{item['title']}</title>
            <link>{item['link']}</link>
            <pubDate>{item['pubDate']}</pubDate>
            <description>{item.get('description', '')}</description>
        </item>
        """
    return f"""<?xml version="1.0"?>
    <rss version="2.0">
    <channel>
        <title>Test Feed</title>
        {entries}
    </channel>
    </rss>"""


def _rfc822(dt: datetime) -> str:
    """Formatiert ein datetime als RFC822-String (RSS pubDate Format)."""
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


def _mock_response(status: int = 200, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status >= 400:
        import requests
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    return resp


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Tests: RSS-Abruf & Deduplication
# ---------------------------------------------------------------------------

class TestFetchArticles:

    def test_fetches_articles_from_feeds(self):
        model = FakeSentimentModel()
        feature = SentimentFeature(model=model, feeds={"test": "http://fake/feed.rss"})

        xml = _rss_xml([
            {"title": "Fed raises rates", "link": "http://a.com/1", "pubDate": _rfc822(_now())},
        ])

        with patch("requests.get", return_value=_mock_response(200, xml)):
            articles = feature.fetch_articles()

        assert len(articles) == 1
        assert articles[0].title == "Fed raises rates"

    def test_deduplicates_by_url(self):
        model = FakeSentimentModel()
        feature = SentimentFeature(model=model, feeds={"test": "http://fake/feed.rss"})

        xml = _rss_xml([
            {"title": "Same article", "link": "http://a.com/dup", "pubDate": _rfc822(_now())},
        ])

        with patch("requests.get", return_value=_mock_response(200, xml)):
            first  = feature.fetch_articles()
            second = feature.fetch_articles()

        assert len(first) == 1
        assert len(second) == 0   # bereits gesehen -> nicht erneut hinzugefuegt

    def test_handles_feed_error_gracefully(self):
        model = FakeSentimentModel()
        feature = SentimentFeature(model=model, feeds={"broken": "http://fake/broken.rss"})

        import requests
        with patch("requests.get", side_effect=requests.ConnectionError("down")):
            articles = feature.fetch_articles()

        assert articles == []

    def test_old_articles_pruned(self):
        model = FakeSentimentModel()
        feature = SentimentFeature(
            model=model, feeds={"test": "http://fake/feed.rss"},
            window=timedelta(hours=2),
        )

        old_date = _now() - timedelta(hours=5)
        xml = _rss_xml([
            {"title": "Old news", "link": "http://a.com/old", "pubDate": _rfc822(old_date)},
        ])

        with patch("requests.get", return_value=_mock_response(200, xml)):
            feature.fetch_articles()

        # Artikel ausserhalb des Fensters sollten entfernt sein
        assert len(feature._articles) == 0

    def test_multiple_feeds_combined(self):
        model = FakeSentimentModel()
        feature = SentimentFeature(
            model=model,
            feeds={
                "feed1": "http://fake/1.rss",
                "feed2": "http://fake/2.rss",
            },
        )

        xml1 = _rss_xml([{"title": "A", "link": "http://a.com/1", "pubDate": _rfc822(_now())}])
        xml2 = _rss_xml([{"title": "B", "link": "http://a.com/2", "pubDate": _rfc822(_now())}])

        responses = [_mock_response(200, xml1), _mock_response(200, xml2)]
        with patch("requests.get", side_effect=responses):
            articles = feature.fetch_articles()

        assert len(articles) == 2


# ---------------------------------------------------------------------------
# Tests: Sentiment-Aggregation
# ---------------------------------------------------------------------------

class TestGetSentimentScore:

    def test_no_articles_returns_zero(self):
        model = FakeSentimentModel()
        feature = SentimentFeature(model=model, feeds={})
        result = feature.get_sentiment_score("EURUSD")
        assert result.score == 0.0
        assert result.article_count == 0

    def test_relevant_article_used(self):
        model = FakeSentimentModel(fixed_score=0.8)
        feature = SentimentFeature(model=model, feeds={})

        feature._articles.append(NewsArticle(
            title="Federal Reserve raises interest rates",
            url="http://a.com/1",
            published_at=_now(),
            source="test",
        ))

        result = feature.get_sentiment_score("EURUSD")
        assert result.score == 0.8
        assert result.article_count == 1

    def test_irrelevant_article_filtered_out(self):
        model = FakeSentimentModel(fixed_score=0.9)
        feature = SentimentFeature(model=model, feeds={})

        feature._articles.append(NewsArticle(
            title="Local sports team wins championship",
            url="http://a.com/sports",
            published_at=_now(),
            source="test",
        ))

        result = feature.get_sentiment_score("EURUSD")
        assert result.article_count == 0
        assert result.score == 0.0

    def test_average_of_multiple_articles(self):
        # fixed_score=None-Ersatz: 0.0 als Fallback, damit ein Key-Mismatch
        # sofort auffaellt statt sich hinter dem Default zu verstecken
        model = FakeSentimentModel(fixed_score=0.0, text_score_map={
            "Fed raises rates aggressively.":  1.0,
            "Federal Reserve signals caution.": -0.5,
        })
        feature = SentimentFeature(model=model, feeds={})

        feature._articles.append(NewsArticle(
            title="Fed raises rates aggressively", url="http://a.com/1",
            published_at=_now(), source="test",
        ))
        feature._articles.append(NewsArticle(
            title="Federal Reserve signals caution", url="http://a.com/2",
            published_at=_now(), source="test",
        ))

        result = feature.get_sentiment_score("EURUSD")
        assert result.article_count == 2
        assert result.score == pytest.approx(0.25, abs=0.01)

    def test_batch_inference_called_once(self):
        model = FakeSentimentModel(fixed_score=0.1)
        feature = SentimentFeature(model=model, feeds={})

        for i in range(5):
            feature._articles.append(NewsArticle(
                title=f"Fed announcement {i}", url=f"http://a.com/{i}",
                published_at=_now(), source="test",
            ))

        feature.get_sentiment_score("EURUSD")

        # Batch-Inferenz: genau EIN predict()-Aufruf mit allen Texten, nicht 5 einzelne
        assert len(model.calls) == 1
        assert len(model.calls[0]) == 5

    def test_outside_window_excluded(self):
        model = FakeSentimentModel(fixed_score=0.9)
        feature = SentimentFeature(model=model, feeds={}, window=timedelta(hours=2))

        feature._articles.append(NewsArticle(
            title="Fed news from yesterday",
            url="http://a.com/old",
            published_at=_now() - timedelta(hours=5),
            source="test",
        ))

        result = feature.get_sentiment_score("EURUSD")
        assert result.article_count == 0

    def test_eur_keyword_matches_eurusd(self):
        model = FakeSentimentModel(fixed_score=0.3)
        feature = SentimentFeature(model=model, feeds={})

        feature._articles.append(NewsArticle(
            title="ECB signals rate cut", url="http://a.com/ecb",
            published_at=_now(), source="test",
        ))

        result = feature.get_sentiment_score("EURUSD")
        assert result.article_count == 1


# ---------------------------------------------------------------------------
# Tests: build_feature (Pipeline-Integration)
# ---------------------------------------------------------------------------

class TestBuildFeature:

    def test_returns_feature_dict(self):
        model = FakeSentimentModel(fixed_score=0.5)
        feature = SentimentFeature(model=model, feeds={})

        feature._articles.append(NewsArticle(
            title="Fed update", url="http://a.com/1",
            published_at=_now(), source="test",
        ))

        result = feature.build_feature("EURUSD")
        assert result == {"sentiment_score": 0.5}

    def test_score_within_bounds(self):
        model = FakeSentimentModel(fixed_score=0.95)
        feature = SentimentFeature(model=model, feeds={})
        feature._articles.append(NewsArticle(
            title="Fed update", url="http://a.com/1",
            published_at=_now(), source="test",
        ))
        result = feature.build_feature("EURUSD")
        assert -1.0 <= result["sentiment_score"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: NewsArticle
# ---------------------------------------------------------------------------

class TestNewsArticle:

    def test_url_hash_consistent(self):
        a1 = NewsArticle(title="A", url="http://x.com/1", published_at=_now(), source="s")
        a2 = NewsArticle(title="Different title", url="http://x.com/1", published_at=_now(), source="s")
        assert a1.url_hash == a2.url_hash   # gleicher URL -> gleicher Hash

    def test_url_hash_differs_for_different_urls(self):
        a1 = NewsArticle(title="A", url="http://x.com/1", published_at=_now(), source="s")
        a2 = NewsArticle(title="A", url="http://x.com/2", published_at=_now(), source="s")
        assert a1.url_hash != a2.url_hash

    def test_text_combines_title_and_summary(self):
        a = NewsArticle(
            title="Title", url="http://x.com", published_at=_now(),
            source="s", summary="Summary text",
        )
        assert "Title" in a.text
        assert "Summary text" in a.text


# ---------------------------------------------------------------------------
# Tests: FinBERTModel (Lazy-Loading, ohne tatsaechlichen Download)
# ---------------------------------------------------------------------------

class TestFinBERTModel:

    def test_empty_texts_returns_empty_list(self):
        model = FinBERTModel()
        # Sollte NICHT versuchen das Modell zu laden bei leerer Liste
        assert model.predict([]) == []

    def test_lazy_loading_raises_clear_error_without_transformers(self):
        model = FinBERTModel()
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(RuntimeError, match="transformers"):
                model.predict(["some text"])

    def test_label_mapping_positive(self):
        model = FinBERTModel()
        model._pipeline = MagicMock(return_value=[{"label": "positive", "score": 0.9}])
        scores = model.predict(["good news"])
        assert scores == [0.9]

    def test_label_mapping_negative(self):
        model = FinBERTModel()
        model._pipeline = MagicMock(return_value=[{"label": "negative", "score": 0.7}])
        scores = model.predict(["bad news"])
        assert scores == [-0.7]

    def test_label_mapping_neutral(self):
        model = FinBERTModel()
        model._pipeline = MagicMock(return_value=[{"label": "neutral", "score": 0.6}])
        scores = model.predict(["neutral news"])
        assert scores == [0.0]
