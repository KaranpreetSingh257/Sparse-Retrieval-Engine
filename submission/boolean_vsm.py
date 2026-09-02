"""
submission/boolean_vsm.py — Boolean retrieval + vector-space ranking.

Required component (assignment Section 4.1): "supports conjunctive/
disjunctive Boolean queries and a cosine-similarity vector-space ranking
with a TF-IDF weighting scheme of your choice."

Two independent pieces to implement:

1. Boolean retrieval: given a query, treat it as an AND (conjunctive) or
   OR (disjunctive) combination of terms and return the matching document
   set — no ranking, just set membership. Useful as a fast candidate
   filter and as a sanity check ("does my index even find the right
   documents for this query?").

2. Vector-space ranking: represent the query and each candidate document
   as TF-IDF weighted vectors and rank by cosine similarity. A standard
   TF-IDF weight for term t in document d:

       w(t, d) = tf(t, d) * log( N / df(t) )

   (log base is your choice — just be consistent), and cosine similarity
   between query vector q and document vector d:

       sim(q, d) = (q . d) / (||q|| * ||d||)

Both pieces should read from the same InvertedIndex you build in
indexer.py.
"""
import math
from typing import Dict, List, Optional, Set, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None
_DOC_NORMS: Dict[str, float] = {}
_IDF_CACHE: Dict[str, float] = {}


def _ensure_doc_norms() -> None:
    """Lazily compute document vector norms if VSM scoring is called."""
    global _DOC_NORMS
    if _DOC_NORMS or not _INDEX or _INDEX.N == 0:
        return
    doc_sq_sums: Dict[str, float] = {}
    for term, post in _INDEX.postings.items():
        idf = _IDF_CACHE.get(term, 0.0)
        for doc_id, tf in post.items():
            weight = tf * idf
            doc_sq_sums[doc_id] = doc_sq_sums.get(doc_id, 0.0) + (weight * weight)
    for doc_id, sq_sum in doc_sq_sums.items():
        _DOC_NORMS[doc_id] = math.sqrt(sq_sum)


def build(index: InvertedIndex) -> None:
    """Precompute IDF cache for VSM."""
    global _INDEX, _DOC_NORMS, _IDF_CACHE
    _INDEX = index
    _DOC_NORMS = {}
    _IDF_CACHE = {}

    if not _INDEX or _INDEX.N == 0:
        return

    for term, post in _INDEX.postings.items():
        df = len(post)
        if df > 0:
            _IDF_CACHE[term] = math.log(_INDEX.N / df)


def boolean_search(query: str, mode: str = "and") -> List[str]:
    """Return matching doc_ids for conjunctive ('and') or disjunctive ('or') query."""
    if not _INDEX:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    doc_sets: List[Set[str]] = [
        set(_INDEX.postings.get(t, {}).keys()) for t in tokens
    ]

    mode_lower = mode.strip().lower()
    if mode_lower == "and":
        matching = set.intersection(*doc_sets) if doc_sets else set()
    elif mode_lower == "or":
        matching = set.union(*doc_sets) if doc_sets else set()
    else:
        raise ValueError(f"Unknown boolean search mode: {mode}")

    return sorted(list(matching))


def vsm_score(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return top-k (doc_id, score) pairs ranked by TF-IDF cosine similarity."""
    if not _INDEX or _INDEX.N == 0:
        return []

    _ensure_doc_norms()

    tokens = tokenize(query)
    if not tokens:
        return []

    # Count query term frequencies
    query_tf: Dict[str, int] = {}
    for token in tokens:
        query_tf[token] = query_tf.get(token, 0) + 1

    # Compute query vector weights and query norm
    query_weights: Dict[str, float] = {}
    q_sq_sum = 0.0
    for term, tf in query_tf.items():
        idf = _IDF_CACHE.get(term, 0.0)
        if idf > 0:
            weight = tf * idf
            query_weights[term] = weight
            q_sq_sum += weight * weight

    if q_sq_sum == 0.0:
        return []

    q_norm = math.sqrt(q_sq_sum)

    # Accumulate dot products across matching postings
    doc_dots: Dict[str, float] = {}
    for term, q_w in query_weights.items():
        post = _INDEX.postings.get(term, {})
        idf = _IDF_CACHE.get(term, 0.0)
        for doc_id, tf in post.items():
            d_w = tf * idf
            doc_dots[doc_id] = doc_dots.get(doc_id, 0.0) + (q_w * d_w)

    # Calculate cosine similarity
    scored_docs: List[Tuple[str, float]] = []
    for doc_id, dot in doc_dots.items():
        d_norm = _DOC_NORMS.get(doc_id, 0.0)
        if d_norm > 0:
            cos_sim = dot / (q_norm * d_norm)
            scored_docs.append((doc_id, float(cos_sim)))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:k]
