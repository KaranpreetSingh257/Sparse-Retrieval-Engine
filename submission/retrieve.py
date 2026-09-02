"""
submission/retrieve.py — THE REQUIRED COMPETITION ENTRYPOINT.
"""
from typing import List, Optional, Tuple

from submission import bm25, boolean_vsm, custom_scorer
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

_INDEX: Optional[InvertedIndex] = None


def build_index(corpus_path: str, index_dir: str) -> None:
    """Load the corpus, construct the inverted index, and serialize to index_dir."""
    corpus = load_corpus(corpus_path)
    index = InvertedIndex()
    index.build(corpus)
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    """Load the inverted index from disk and precompute scorer caches."""
    global _INDEX
    _INDEX = InvertedIndex.load(index_dir)
    custom_scorer.build(_INDEX)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    """Return top-k ranked documents for the query."""
    if _INDEX is None:
        raise RuntimeError(
            "retrieve() called before load_index(); the harness always "
            "calls build_index(corpus_path, index_dir) and then "
            "load_index(index_dir) — in that order, in two separate "
            "processes — before any retrieve() calls. If you're testing "
            "manually, do the same."
        )

    return custom_scorer.score(query, k=k)
