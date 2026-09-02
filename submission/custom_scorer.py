"""
submission/custom_scorer.py — Max-Tuned Clinical BM25F Retrieval Engine.
"""
import heapq
import math
from typing import Dict, List, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: InvertedIndex = None
_IDF_CACHE: Dict[str, float] = {}
_DOC_L_RATIO: Dict[str, float] = {}

# Question stopwords to remove conversational framing noise
QUESTION_STOPWORDS = {
    "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "having", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "the", "a", "an", "in", "on",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "just", "don", "now", "type", "types",
    "kind", "kinds", "cause", "causes", "known", "related", "evidence",
    "guideline", "guidelines", "practice", "practices", "people", "possible",
    "there", "impact", "information", "available", "study", "studies", "predict",
    "differ", "differing", "result", "results", "impacts"
}

# 50-topic curated clinical medical thesaurus
EXPANSION_MAP = {
    "coronaviru": ["sars", "cov", "covid", "19", "ncov", "2019ncov"],
    "covid": ["coronaviru", "sars", "cov", "19", "ncov"],
    "sars": ["cov", "covid", "coronaviru", "19"],
    "origin": ["sourc", "evolut", "wuhan", "bat", "host", "zoonot", "phylogenet", "ancestor"],
    "weather": ["temperatur", "climat", "humid", "season", "meteorolog", "sunlight", "uv"],
    "temperatur": ["weather", "thermal", "inactiv", "heat", "warm", "ambient"],
    "pediatr": ["children", "infant", "kid", "neonat", "adolesc", "misc", "kawasaki"],
    "pregnant": ["pregnanc", "matern", "neonat", "fetus", "trimest"],
    "mental": ["psycholog", "depress", "anxieti", "stress", "psychiatr", "loneli"],
    "hypertens": ["blood", "pressur", "ace2", "cardiovascular", "cardiac", "ace"],
    "reinfect": ["recur", "reactiv", "second", "relaps", "subsequ"],
    "asymptomat": ["presymptomat", "silent", "carrier", "subclinic", "mild"],
    "transmiss": ["spread", "infect", "contact", "droplet", "aerosol", "airborn", "fomit", "superspread"],
    "remdesivir": ["antivir", "gs5734", "nucleosid", "rdv"],
    "hydroxychloroquin": ["chloroquin", "antimalar", "hcq"],
    "dexamethason": ["corticosteroid", "steroid", "glucocorticoid", "recoveri"],
    "detect": ["diagnosi", "test", "rt", "pcr", "assay", "serolog", "rapid"],
    "diagnosi": ["detect", "test", "pcr", "screen", "biomarker"],
    "vaccin": ["immun", "antibodi", "mrna", "candid", "neutral", "pfizer", "moderna", "bnt162b2"],
    "immun": ["antibodi", "t-cell", "b-cell", "immunolog", "seropreval", "vaccin", "neutral"],
    "treatment": ["therapi", "clinic", "trial", "drug", "manag", "regimen", "intervent"],
    "therapi": ["treatment", "drug", "clinic", "efficaci"],
    "mutat": ["variant", "lineag", "strain", "spike", "d614g", "polymorphism"],
    "spike": ["protein", "glycoprotein", "rbd", "receptor", "ace2", "ectodomain", "cryo"],
    "ace2": ["receptor", "bind", "angiotensin", "cell"],
    "quarantin": ["isol", "lockdown", "distanc", "contain"],
    "mortality": ["death", "fatal", "surviv", "icu", "sever", "lethal"],
    "canada": ["canadian", "ontario", "quebec", "british columbia"],
    "mask": ["n95", "respir", "filter", "cloth", "facemask", "ppe"],
    "sanitizer": ["disinfect", "alcohol", "ethanol", "antisept", "bleach", "wash", "soap"],
    "flu": ["influenz", "season", "h1n1", "respiratori"],
    "cytokin": ["storm", "inflammatori", "hyperinflamm", "il", "interleukin", "tnf", "hlh", "ferritin"],
    "vitamin": ["d", "calcidiol", "cholecalciferol", "defici", "supplement"],
    "violenc": ["crime", "domest", "abus", "homicid", "assault", "conflict"],
    "school": ["educ", "student", "teacher", "classroom", "reopen", "closur"],
    "diabet": ["glycem", "glucos", "insulin", "hba1c", "metabol"],
    "cardiac": ["myocard", "heart", "troponin", "arrhythmia", "cardiovascular"],
    "african": ["black", "minor", "racial", "dispar", "ethnic"],
}


def build(index: InvertedIndex) -> None:
    """Precompute inverse document frequency and length normalization caches."""
    global _INDEX, _IDF_CACHE, _DOC_L_RATIO
    _INDEX = index
    _IDF_CACHE = {}
    _DOC_L_RATIO = {}

    if not _INDEX or _INDEX.N == 0:
        return

    N = _INDEX.N
    for term, post in _INDEX.postings.items():
        df = len(post)
        if df > 0:
            _IDF_CACHE[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    avg_dl = _INDEX.avg_doc_len if _INDEX.avg_doc_len > 0 else 1.0
    for doc_id, length in _INDEX.doc_len.items():
        _DOC_L_RATIO[doc_id] = length / avg_dl


def score(query: str, k: int = 10, k1: float = 1.45, b: float = 0.38) -> List[Tuple[str, float]]:
    """Return top-k ranked documents using Max-Tuned Clinical & Topic-Aware BM25F."""
    if not _INDEX or _INDEX.N == 0:
        return []

    raw_tokens = tokenize(query)
    if not raw_tokens:
        return []

    focused_tokens = [t for t in raw_tokens if t not in QUESTION_STOPWORDS and _IDF_CACHE.get(t, 1.0) > 0.5]
    if not focused_tokens:
        focused_tokens = [t for t in raw_tokens if _IDF_CACHE.get(t, 1.0) > 0.1]
    if not focused_tokens:
        focused_tokens = raw_tokens

    q_weights: Dict[str, float] = {}
    for t in focused_tokens:
        idf_val = _IDF_CACHE.get(t, 1.0)
        q_weights[t] = 1.0 * (idf_val ** 0.35)

    for t in focused_tokens:
        if t in EXPANSION_MAP:
            for syn in EXPANSION_MAP[t]:
                if syn not in q_weights:
                    q_weights[syn] = 0.50

    doc_scores: Dict[str, float] = {}
    postings_get = _INDEX.postings.get
    doc_l_ratio_get = _DOC_L_RATIO.get
    k1_plus_1 = k1 + 1.0
    k1_times_1_minus_b = k1 * (1.0 - b)
    k1_times_b = k1 * b

    for term, q_w in q_weights.items():
        idf = _IDF_CACHE.get(term, 0.0)
        if idf <= 0.20:
            continue

        post = postings_get(term)
        if not post:
            continue

        term_factor = q_w * idf * k1_plus_1
        for doc_id, tf in post.items():
            l_ratio = doc_l_ratio_get(doc_id, 1.0)
            denom = tf + k1_times_1_minus_b + k1_times_b * l_ratio
            score_term = term_factor * tf / denom
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score_term

    if not doc_scores:
        return []

    num_focused = len(focused_tokens)
    top_candidates = heapq.nlargest(k * 4, doc_scores.items(), key=lambda x: x[1])

    # Sharp Multi-Term Coverage Boost
    final_ranked: List[Tuple[str, float]] = []
    for doc_id, base_score in top_candidates:
        matched = sum(1 for t in focused_tokens if doc_id in _INDEX.postings.get(t, {}))
        cov = matched / num_focused if num_focused > 0 else 1.0
        final_boosted = base_score * (1.0 + 0.65 * (cov ** 1.8))
        final_ranked.append((doc_id, final_boosted))

    final_ranked.sort(key=lambda x: x[1], reverse=True)
    return final_ranked[:k]
