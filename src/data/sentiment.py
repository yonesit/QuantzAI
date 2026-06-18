"""
src/data/sentiment.py
SentimentFeature – RSS-News-Sentiment via FinBERT als Pipeline-Feature.

Optionales Modul (siehe Issue #8). Wird nur aktiv wenn
config.yaml: features.include_sentiment: true

FinBERT (transformers/torch) wird lazy geladen, damit das Modul
importierbar bleibt auch wenn die ML-Libraries (noch) nicht installiert sind.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Protocol

import requests
from loguru import logger


# ─────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────

_DEFAULT_RSS_FEEDS: dict[str, str] = {
    "reuters_business":    "https://feeds.reuters.com/reuters/businessNews",
    "investing_forex":     "https://www.investing.com/rss/news_1.rss",
    "bloomberg_markets":   "https://feeds.bloomberg.com/markets/news.rss",
}

_AGGREGATION_WINDOW = timedelta(hours=2)

# Sehr grobe Heuristik um Artikel grob einem Waehrungspaar zuzuordnen
_CURRENCY_KEYWORDS: dict[str, list[str]] = {
    "USD": ["federal reserve", "fed ", "dollar", "fomc", "powell", "u.s. economy", "nonfarm"],
    "EUR": ["ecb", "eurozone", "euro area", "lagarde", "european central bank"],
    "GBP": ["bank of england", "boe ", "pound sterling", "sterling", "uk inflation"],
    "JPY": ["bank of japan", "boj ", "yen", "japan inflation"],
    "AUD": ["rba ", "reserve bank of australia", "aussie dollar"],
    "CAD": ["bank of canada", "boc ", "canadian dollar"],
    "CHF": ["swiss national bank", "snb ", "swiss franc"],
}


# ─────────────────────────────────────────────
#  Datenklassen
# ─────────────────────────────────────────────

@dataclass
class NewsArticle:
    title: str
    url: str
    published_at: datetime
    source: str
    summary: str = ""

    @property
    def url_hash(self) -> str:
        """Hash zur Deduplication."""
        return hashlib.sha256(self.url.encode("utf-8")).hexdigest()

    @property
    def text(self) -> str:
        """Text der fuer die Sentiment-Analyse verwendet wird."""
        return f"{self.title}. {self.summary}".strip()


@dataclass
class SentimentScore:
    symbol: str
    score: float            # -1.0 (negativ) bis +1.0 (positiv)
    article_count: int
    window_start: datetime
    window_end: datetime


# ─────────────────────────────────────────────
#  Sentiment-Modell-Adapter (austauschbar)
# ─────────────────────────────────────────────

class SentimentModel(Protocol):
    """Schnittstelle die jedes Sentiment-Backend erfuellen muss."""

    def predict(self, texts: list[str]) -> list[float]:
        """Gibt fuer jeden Text einen Score zwischen -1.0 und +1.0 zurueck."""
        ...


class FinBERTModel:
    """
    FinBERT-Adapter via HuggingFace transformers.

    Lazy-Loading: das Modell (~500MB) wird erst beim ersten predict()-Aufruf
    geladen, nicht beim Import oder bei __init__.

    Parameters
    ----------
    model_name : HuggingFace Model-ID
    device     : "cpu", "cuda", oder None (= automatisch erkennen)
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._device_pref = device
        self._pipeline = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT benoetigt 'transformers' und 'torch'. "
                "Installiere mit: pip install transformers torch"
            ) from exc

        device = self._device_pref
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        device_idx = 0 if device == "cuda" else -1

        logger.info("FinBERT wird geladen | model={model} device={device}",
                     model=self._model_name, device=device)

        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self._model_name,
            device=device_idx,
        )

    def predict(self, texts: list[str]) -> list[float]:
        """
        Batch-Inferenz. Gibt pro Text einen Score zwischen -1.0 und +1.0 zurueck.

        FinBERT-Labels: positive, negative, neutral.
        Mapping: positive -> +score, negative -> -score, neutral -> 0.0
        """
        if not texts:
            return []

        self._ensure_loaded()
        results = self._pipeline(texts, batch_size=16, truncation=True)

        scores = []
        for r in results:
            label = r["label"].lower()
            conf  = float(r["score"])
            if label == "positive":
                scores.append(conf)
            elif label == "negative":
                scores.append(-conf)
            else:
                scores.append(0.0)
        return scores


# ─────────────────────────────────────────────
#  SentimentFeature
# ─────────────────────────────────────────────

class SentimentFeature:
    """
    Konsumiert RSS-Feeds, berechnet Sentiment-Scores via FinBERT (oder
    einem anderen SentimentModel) und liefert ein aggregiertes Feature
    pro Waehrungspaar fuer die letzten 2 Stunden.

    Parameters
    ----------
    model        : SentimentModel-Implementierung (z.B. FinBERTModel())
    feeds        : dict {name: rss_url}, Standard: Reuters/Bloomberg/Investing
    window       : Aggregationsfenster (Standard: 2 Stunden)
    timeout      : HTTP-Timeout fuer RSS-Abruf
    """

    def __init__(
        self,
        model: Optional[SentimentModel] = None,
        feeds: Optional[dict[str, str]] = None,
        window: timedelta = _AGGREGATION_WINDOW,
        timeout: int = 10,
    ) -> None:
        self._model   = model or FinBERTModel()
        self._feeds   = feeds or _DEFAULT_RSS_FEEDS
        self._window  = window
        self._timeout = timeout

        self._seen_hashes: set[str] = set()
        self._articles: list[NewsArticle] = []

    # ── Feed-Abruf ───────────────────────────────────

    def fetch_articles(self) -> list[NewsArticle]:
        """
        Holt neue Artikel aus allen konfigurierten RSS-Feeds.
        Bereits gesehene Artikel (per URL-Hash) werden nicht erneut hinzugefuegt.

        Returns
        -------
        list[NewsArticle]: nur die NEU hinzugefuegten Artikel dieses Aufrufs
        """
        new_articles: list[NewsArticle] = []

        for name, url in self._feeds.items():
            try:
                resp = requests.get(url, timeout=self._timeout)
                resp.raise_for_status()
                parsed = self._parse_rss(resp.text, source=name)

                for article in parsed:
                    if article.url_hash in self._seen_hashes:
                        continue
                    self._seen_hashes.add(article.url_hash)
                    self._articles.append(article)
                    new_articles.append(article)

            except requests.RequestException as exc:
                logger.warning("RSS-Feed nicht erreichbar | {name} | {exc}", name=name, exc=exc)

        self._prune_old_articles()
        return new_articles

    def _parse_rss(self, xml_text: str, source: str) -> list[NewsArticle]:
        """Minimaler RSS-Parser (nur <item> mit title/link/pubDate)."""
        import xml.etree.ElementTree as ET

        articles = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("RSS-Parsing fehlgeschlagen | {source} | {exc}", source=source, exc=exc)
            return []

        for item in root.iter("item"):
            title_el = item.find("title")
            link_el  = item.find("link")
            date_el  = item.find("pubDate")
            desc_el  = item.find("description")

            if title_el is None or link_el is None:
                continue

            published_at = self._parse_pubdate(date_el.text if date_el is not None else None)

            articles.append(NewsArticle(
                title=title_el.text or "",
                url=link_el.text or "",
                published_at=published_at,
                source=source,
                summary=(desc_el.text or "") if desc_el is not None else "",
            ))

        return articles

    @staticmethod
    def _parse_pubdate(raw: Optional[str]) -> datetime:
        """Parst RFC822-Datum (Standard fuer RSS pubDate). Fallback: jetzt."""
        if not raw:
            return datetime.now(timezone.utc)
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return datetime.now(timezone.utc)

    def _prune_old_articles(self) -> None:
        """Entfernt Artikel die ausserhalb des Aggregationsfensters liegen."""
        cutoff = datetime.now(timezone.utc) - self._window
        self._articles = [a for a in self._articles if a.published_at >= cutoff]

    # ── Sentiment-Berechnung ─────────────────────────

    def get_sentiment_score(self, symbol: str) -> SentimentScore:
        """
        Aggregierter Sentiment-Score fuer ein Waehrungspaar
        ueber die Artikel der letzten `window` (Standard: 2h).

        Parameters
        ----------
        symbol : z.B. "EURUSD"

        Returns
        -------
        SentimentScore mit score=0.0 wenn keine relevanten Artikel vorhanden sind.
        """
        now = datetime.now(timezone.utc)
        window_start = now - self._window

        relevant = self._filter_relevant_articles(symbol, window_start, now)

        if not relevant:
            return SentimentScore(
                symbol=symbol, score=0.0, article_count=0,
                window_start=window_start, window_end=now,
            )

        texts = [a.text for a in relevant]
        scores = self._model.predict(texts)  # Batch-Inferenz, nicht pro Artikel einzeln

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return SentimentScore(
            symbol=symbol,
            score=round(avg_score, 4),
            article_count=len(relevant),
            window_start=window_start,
            window_end=now,
        )

    def _filter_relevant_articles(
        self, symbol: str, window_start: datetime, window_end: datetime
    ) -> list[NewsArticle]:
        """Filtert Artikel nach Zeitfenster und thematischer Relevanz fuer das Symbol."""
        currencies = self._extract_currencies(symbol)
        keywords: list[str] = []
        for cur in currencies:
            keywords.extend(_CURRENCY_KEYWORDS.get(cur, []))

        relevant = []
        for article in self._articles:
            if not (window_start <= article.published_at <= window_end):
                continue
            text_lower = article.text.lower()
            if any(kw in text_lower for kw in keywords):
                relevant.append(article)

        return relevant

    @staticmethod
    def _extract_currencies(symbol: str) -> set[str]:
        s = symbol.upper().replace("_", "")
        if len(s) == 6:
            return {s[:3], s[3:]}
        return {s}

    # ── Feature-Export ───────────────────────────────

    def build_feature(self, symbol: str) -> dict:
        """
        Liefert das Feature-Dict fuer die Pipeline-Integration.

        Returns
        -------
        dict: {"sentiment_score": float}
        """
        result = self.get_sentiment_score(symbol)
        return {"sentiment_score": result.score}
