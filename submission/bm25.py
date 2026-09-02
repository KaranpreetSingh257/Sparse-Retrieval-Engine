"""
submission/bm25.py — Okapi BM25 ranking.

Required component (assignment Section 4.1): "a BM25 implementation with
tunable k1 and b." See the assignment background (Section 3) for the
Robertson & Walker / Robertson & Zaragoza references this is based on.

BM25 score for a query Q = q1...qn against document D:

    score(D, Q) = sum_i  IDF(qi) * ( tf(qi, D) * (k1 + 1) )
                                   / ( tf(qi, D) + k1 * (1 - b + b * |D| / avgdl) )

A standard IDF variant (Robertson-Sparck Jones, +1-smoothed so it stays
non-negative even for terms occurring in more than half the corpus):

    IDF(qi) = ln( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )

where:
    N        = number of documents in the corpus
    df(qi)   = number of documents containing qi
    tf(qi,D) = term frequency of qi in D
    |D|      = length of D in tokens
    avgdl    = average document length across the corpus

k1 (typically 1.2-2.0) controls term-frequency saturation; b (in [0, 1])
controls document-length normalisation strength. Both must be exposed as
parameters, not hard-coded — you need to sweep them for your report
(assignment Section 8, "parameter search procedure for k1, b").
"""
import math
from typing import Dict, List, Optional, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None
_IDF_CACHE: Dict[str, float] = {}
_DOC_L_RATIO: Dict[str, float] = {}


def build(index: InvertedIndex) -> None:
    """Precompute Robertson-Sparck Jones smoothed IDF cache and document length ratios."""
    global _INDEX, _IDF_CACHE, _DOC_L_RATIO
    _INDEX = index
    _IDF_CACHE = {}
    _DOC_L_RATIO = {}

    if not _INDEX or _INDEX.N == 0:
        return

    N = _INDEX.N
    for term, post in _INDEX.postings.items():
        df = len(post)
        _IDF_CACHE[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    avg_dl = _INDEX.avg_doc_len if _INDEX.avg_doc_len > 0 else 1.0
    for doc_id, length in _INDEX.doc_len.items():
        _DOC_L_RATIO[doc_id] = length / avg_dl


def score(query: str, k: int = 10, k1: float = 1.5, b: float = 0.50) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs for `query`, BM25-ranked, highest score first."""
    if not _INDEX or _INDEX.N == 0:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    doc_scores: Dict[str, float] = {}

    for term in tokens:
        postings = _INDEX.postings.get(term)
        if not postings:
            continue

        idf = _IDF_CACHE.get(term)
        if idf is None:
            df = len(postings)
            idf = math.log((_INDEX.N - df + 0.5) / (df + 0.5) + 1.0)

        for doc_id, tf in postings.items():
            l_ratio = _DOC_L_RATIO.get(doc_id, 1.0)
            denom = tf + k1 * (1.0 - b + b * l_ratio)
            term_score = idf * ((tf * (k1 + 1.0)) / denom)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, float(s)) for doc_id, s in ranked[:k]]
