"""Algorithm 3 of the manuscript, the RAG insight distillation pipeline.

Four stages, run once per domain and reused for every trajectory in it:

  1. retrieval     arXiv API, and Google Scholar via SerpAPI when a key exists
  2. ranking       Eq. (rag_rank), the LLM scores each abstract on [0, 100]
  3. extraction    methods, metrics and OOD limits from the top K_doc papers
  4. compression   two to three actionable bulleted constraints -> I_domain

Retrieval, ranking and distillation are executed once per application domain,
not once per trajectory, and the result is cached on disk per domain.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import quote

import requests

METHOD_TERMS = (
    "out-of-distribution detection, multi-scale temporal analysis, "
    "density-based forecasting"
)


@dataclass
class Candidate:
    title: str
    authors: str
    abstract: str
    url: str
    source: str
    date: str
    relevance: float = 0.0


# --------------------------------------------------------------------------
# Stage 1, retrieval
# --------------------------------------------------------------------------

def query_arxiv(domain, max_results=6, timeout=30):
    query = f"{domain} {METHOD_TERMS} time series regression OOD"
    url = (
        "http://export.arxiv.org/api/query?search_query=all:"
        f"{quote(query)}&start=0&max_results={max_results}"
        "&sortBy=relevance&sortOrder=descending"
    )
    ns = "{http://www.w3.org/2005/Atom}"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        print(f"[rag] arXiv query failed: {exc}")
        return []

    out = []
    for entry in root.findall(f"{ns}entry"):
        authors = [a.find(f"{ns}name").text for a in entry.findall(f"{ns}author")]
        out.append(
            Candidate(
                title=entry.find(f"{ns}title").text.strip(),
                authors=", ".join(authors),
                abstract=entry.find(f"{ns}summary").text.strip(),
                url=entry.find(f"{ns}id").text,
                source="arXiv",
                date=entry.find(f"{ns}published").text,
            )
        )
    print(f"[rag] arXiv returned {len(out)} candidates")
    return out


def query_scholar(domain, serpapi_key, max_results=6, timeout=30):
    """Google Scholar via SerpAPI. Returns an empty list without a key."""
    if not serpapi_key:
        return []
    params = {
        "engine": "google_scholar",
        "q": f"{domain} {METHOD_TERMS}",
        "num": max_results,
        "api_key": serpapi_key,
    }
    try:
        r = requests.get("https://serpapi.com/search", params=params, timeout=timeout)
        r.raise_for_status()
        results = r.json().get("organic_results", [])
    except Exception as exc:
        print(f"[rag] Scholar query failed: {exc}")
        return []

    out = []
    for item in results[:max_results]:
        info = item.get("publication_info", {})
        out.append(
            Candidate(
                title=item.get("title", ""),
                authors=info.get("summary", ""),
                abstract=item.get("snippet", ""),
                url=item.get("link", ""),
                source="Scholar",
                date=info.get("summary", ""),
            )
        )
    print(f"[rag] Scholar returned {len(out)} candidates")
    return out


# --------------------------------------------------------------------------
# Stage 2, LLM ranking, Eq. (rag_rank)
# --------------------------------------------------------------------------

RANK_PROMPT = """Domain: {domain}

Paper title: {title}
Abstract: {abstract}

Rate the relevance of this paper for TIME SERIES FORECASTING and REGRESSION
with out-of-distribution detection, on a scale of 0 to 100. Reward only:
- forecasting accuracy and prediction methodology
- multi-scale temporal analysis for regression
- out-of-distribution detection in time series
- density-based approaches and reconstruction error analysis
- RMSE, MAE or MASE evaluation

Penalise heavily: binary classification, AUC optimisation, sensitivity and
specificity, and general anomaly detection with no forecasting target.

Respond with a single integer between 0 and 100 and nothing else."""


def rank_candidates(candidates, domain, llm, pause=0.1):
    for c in candidates:
        prompt = RANK_PROMPT.format(
            domain=domain, title=c.title, abstract=c.abstract[:1500]
        )
        try:
            text = llm.generate(prompt)
            match = re.search(r"\d+", text)
            c.relevance = float(match.group()) if match else 30.0
        except Exception as exc:
            print(f"[rag] ranking failed for {c.title[:48]!r}: {exc}")
            c.relevance = 30.0
        if pause:
            time.sleep(pause)
    return sorted(candidates, key=lambda c: c.relevance, reverse=True)


# --------------------------------------------------------------------------
# Stages 3 and 4, extraction then compression
# --------------------------------------------------------------------------

EXTRACT_PROMPT = """Domain: {domain}

From the papers below, extract only what transfers to time series regression
under distribution shift. Cover: multi-scale methods, regression metrics,
out-of-distribution detection, and density or reconstruction-error approaches.
One or two bullets per heading, under 300 words in total. Do not mention
classification metrics.

{papers}"""

COMPRESS_PROMPT = """Compress the text below into two or three actionable
constraints for judging whether a geometric deviation in a time series is a
structurally meaningful event or benign noise. Bullet format, under 120 words,
no classification metrics.

{raw}"""


def distill(ranked, domain, llm, k_doc=3):
    top = ranked[:k_doc]
    if not top:
        return ""
    papers = "\n---\n".join(
        f"Paper {i}: {c.title}\nAbstract: {c.abstract[:400]}\n"
        f"Relevance: {c.relevance:.0f}/100"
        for i, c in enumerate(top, 1)
    )
    raw = llm.generate(EXTRACT_PROMPT.format(domain=domain, papers=papers))
    return llm.generate(COMPRESS_PROMPT.format(raw=raw)).strip()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def domain_insights(
    domain,
    llm,
    k_cand=6,
    k_doc=3,
    serpapi_key=None,
    cache_path=None,
):
    """Algorithm 3. Returns I_domain, cached on disk per domain.

    Retrieval, ranking and distillation happen once per domain, not once per
    trajectory.
    """
    cache_path = Path(cache_path) if cache_path else (
        Path(__file__).resolve().parent / "cache" / "rag_insights.json"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    store = {}
    if cache_path.exists():
        try:
            store = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            store = {}
    if domain in store:
        print(f"[rag] using cached insights for {domain!r}")
        return store[domain]["insights"]

    candidates = query_arxiv(domain, max_results=k_cand)
    candidates += query_scholar(domain, serpapi_key, max_results=k_cand)
    if not candidates:
        print("[rag] no candidates retrieved, I_domain is empty")
        return ""

    ranked = rank_candidates(candidates, domain, llm)
    insights = distill(ranked, domain, llm, k_doc=k_doc)

    store[domain] = {
        "insights": insights,
        "corpus": [asdict(c) for c in ranked],
    }
    cache_path.write_text(json.dumps(store, indent=2))
    print(f"[rag] I_domain is {len(insights)} characters")
    return insights
