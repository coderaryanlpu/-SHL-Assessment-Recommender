"""
Retrieval module for SHL catalog.
Uses TF-IDF + BM25-style keyword matching — no torch/sentence-transformers needed.
Falls back to Gemini embeddings if available.
"""
import json
import re
import math
from pathlib import Path
from typing import Optional
import os

CATALOG_PATH = Path(__file__).parent / "catalog.json"

def load_catalog() -> list[dict]:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Tokenisation ──────────────────────────────────────────────────────────────

STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "to","of","in","on","at","for","with","by","from","as","into","through",
    "and","or","but","not","if","then","this","that","these","those","it","its",
    "i","we","you","he","she","they","our","your","their","my","me","him","her","us",
    "what","which","who","how","when","where","why","need","want","looking","hiring",
    "role","position","candidate","candidates","assessment","assessments","test","tests",
}

def tokenise(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9#+]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    K1 = 1.5
    B  = 0.75

    def __init__(self, catalog: list[dict]):
        self.catalog = catalog
        self.n = len(catalog)
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: list[dict[str,int]] = []
        self.df: dict[str,int] = {}
        self.avgdl: float = 0.0
        self._build(catalog)

    def _build(self, catalog):
        total_len = 0
        for item in catalog:
            tokens = tokenise(item["text"])
            self.doc_tokens.append(tokens)
            freq: dict[str,int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.doc_freq.append(freq)
            total_len += len(tokens)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = total_len / self.n if self.n else 1

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        freq = self.doc_freq[doc_idx]
        dl = len(self.doc_tokens[doc_idx])
        score = 0.0
        for t in query_tokens:
            if t not in freq:
                continue
            tf = freq[t]
            df = self.df.get(t, 0)
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self.K1 + 1)) / (tf + self.K1 * (1 - self.B + self.B * dl / self.avgdl))
            score += idf * tf_norm
        return score

    def search(self, query: str, top_k: int = 20, filters: Optional[dict] = None) -> list[dict]:
        query_tokens = tokenise(query)
        if not query_tokens:
            return []

        scores = []
        for i, item in enumerate(self.catalog):
            # Apply filters
            if filters:
                if "test_types" in filters and filters["test_types"]:
                    item_types = set(item.get("test_type_full", []))
                    if not item_types.intersection(filters["test_types"]):
                        continue
                if "job_levels" in filters and filters["job_levels"]:
                    item_levels = set(item.get("job_levels", []))
                    if not item_levels.intersection(filters["job_levels"]):
                        continue
            sc = self.score(query_tokens, i)
            if sc > 0:
                scores.append((sc, i))

        scores.sort(reverse=True)
        results = []
        for sc, idx in scores[:top_k]:
            entry = dict(self.catalog[idx])
            entry["_score"] = round(sc, 3)
            results.append(entry)
        return results


# ── Singleton ─────────────────────────────────────────────────────────────────
_index: Optional[BM25Index] = None

def get_index() -> BM25Index:
    global _index
    if _index is None:
        catalog = load_catalog()
        _index = BM25Index(catalog)
    return _index


def retrieve(query: str, top_k: int = 15, filters: Optional[dict] = None) -> list[dict]:
    """Main retrieval entry point. Returns top_k most relevant assessments."""
    idx = get_index()
    return idx.search(query, top_k=top_k, filters=filters)


def get_by_name(name: str) -> Optional[dict]:
    """Find an assessment by (partial) name match."""
    catalog = get_index().catalog
    name_lower = name.lower()
    # Exact first
    for item in catalog:
        if item["name"].lower() == name_lower:
            return item
    # Partial
    for item in catalog:
        if name_lower in item["name"].lower():
            return item
    return None


def format_for_prompt(items: list[dict], max_items: int = 20) -> str:
    """Format catalog items into a compact string for the LLM prompt context."""
    lines = []
    for i, item in enumerate(items[:max_items], 1):
        langs = item.get("languages", [])
        lang_str = ", ".join(langs[:3])
        if len(langs) > 3:
            lang_str += f" (+{len(langs)-3} more)"
        if not lang_str:
            lang_str = "English"
        lines.append(
            f"{i}. [{item['test_type']}] {item['name']} | "
            f"{item['duration']} | {lang_str} | "
            f"Levels: {', '.join(item.get('job_levels', ['All'])) or 'All'} | "
            f"URL: {item['url']}"
        )
        if item.get("description"):
            desc = item["description"][:120].replace("\n", " ")
            lines.append(f"   Desc: {desc}...")
    return "\n".join(lines)
