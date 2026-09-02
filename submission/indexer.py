"""
submission/indexer.py — build your inverted index here.
"""
import gzip
import os
import pickle
import re
from typing import Dict, List, Tuple

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
except ImportError:
    _STEMMER = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEM_CACHE: Dict[str, str] = {}


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric tokenization with memoized Porter stemming."""
    if not text:
        return []
    raw_tokens = _TOKEN_RE.findall(text.lower())
    if _STEMMER is None:
        return raw_tokens

    stemmed = []
    for token in raw_tokens:
        st = _STEM_CACHE.get(token)
        if st is None:
            st = _STEMMER.stem(token)
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

        for doc_id, text in corpus:
            tokens = tokenize(text)
            length = len(tokens)
            self.doc_len[doc_id] = length
            total_tokens += length

            term_counts: Dict[str, int] = {}
            for idx, token in enumerate(tokens):
                weight = 3 if idx < 25 else 1
                term_counts[token] = term_counts.get(token, 0) + weight

            for term, count in term_counts.items():
                if term not in self.postings:
                    self.postings[term] = {}
                self.postings[term][doc_id] = count

        self.avg_doc_len = (total_tokens / self.N) if self.N > 0 else 0.0

    def document_frequency(self, term: str) -> int:
        """Number of documents containing `term` at least once."""
        return len(self.postings.get(term, {}))

    def save(self, index_dir: str) -> None:
        """Persist postings and statistics to disk using gzip-compressed pickle."""
        os.makedirs(index_dir, exist_ok=True)
        file_path = os.path.join(index_dir, "index.pkl.gz")
        state = {
            "postings": self.postings,
            "doc_len": self.doc_len,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
        }
        with gzip.open(file_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Reconstruct InvertedIndex purely from disk."""
        index = cls()
        file_path = os.path.join(index_dir, "index.pkl.gz")
        with gzip.open(file_path, "rb") as f:
            state = pickle.load(f)
        index.postings = state["postings"]
        index.doc_len = state["doc_len"]
        index.N = state["N"]
        index.avg_doc_len = state["avg_doc_len"]
        return index
