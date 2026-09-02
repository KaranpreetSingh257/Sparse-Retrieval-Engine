"""
submission/indexer.py — High-Efficiency Inverted Index for COL 7364/764 Assignment 1.
"""
import array
import gzip
import os
import pickle
import re
from typing import Dict, List, Tuple

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
    _STEM_FN = _STEMMER.stem
except ImportError:
    _STEMMER = None
    _STEM_FN = None

_TOKEN_RE_FIND = re.compile(r"[a-z0-9]+").findall
_STEM_CACHE: Dict[str, str] = {}


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric tokenization with memoized Porter stemming."""
    if not text:
        return []
    raw_tokens = _TOKEN_RE_FIND(text.lower())
    if _STEM_FN is None:
        return raw_tokens

    stem_cache_get = _STEM_CACHE.get
    stemmed = []
    for token in raw_tokens:
        st = stem_cache_get(token)
        if st is None:
            st = _STEM_FN(token)
            _STEM_CACHE[token] = st
        stemmed.append(st)
    return stemmed


class InvertedIndex:
    """Inverted index data structure storing postings lists, document lengths,
    and collection-level statistics.
    """

    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: term_freq}
        self.doc_len: Dict[str, int] = {}  # doc_id -> number of tokens
        self.N: int = 0  # number of documents
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        """Tokenize each document, construct postings lists, and compute statistics."""
        self.N = len(corpus)
        total_tokens = 0
        postings = self.postings
        doc_len = self.doc_len

        for doc_id, text in corpus:
            tokens = tokenize(text)
            length = len(tokens)
            doc_len[doc_id] = length
            total_tokens += length

            tf: Dict[str, int] = {}
            # Title 3x weighting for first 25 tokens, 1x for body
            for t in tokens[:25]:
                tf[t] = tf.get(t, 0) + 3
            for t in tokens[25:]:
                tf[t] = tf.get(t, 0) + 1

            for t, count in tf.items():
                if t not in postings:
                    postings[t] = {doc_id: count}
                else:
                    postings[t][doc_id] = count

        self.avg_doc_len = (total_tokens / self.N) if self.N > 0 else 0.0

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once."""
        return len(self.postings.get(term, {}))

    def save(self, index_dir: str) -> None:
        """Persist postings and statistics to disk using compact gzip-compressed pickle."""
        os.makedirs(index_dir, exist_ok=True)
        file_path = os.path.join(index_dir, "index.pkl.gz")
        state = {
            "postings": self.postings,
            "doc_len": self.doc_len,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
        }
        with gzip.open(file_path, "wb", compresslevel=6) as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct the inverted index from index_dir."""
        file_path = os.path.join(index_dir, "index.pkl.gz")
        if not os.path.exists(file_path):
            legacy_path = os.path.join(index_dir, "index.pkl")
            if os.path.exists(legacy_path):
                with open(legacy_path, "rb") as f:
                    state = pickle.load(f)
            else:
                raise FileNotFoundError(f"No index file found in {index_dir}")
        else:
            with gzip.open(file_path, "rb") as f:
                state = pickle.load(f)

        idx = cls()
        idx.postings = state["postings"]
        idx.doc_len = state["doc_len"]
        idx.N = state["N"]
        idx.avg_doc_len = state["avg_doc_len"]
        return idx
