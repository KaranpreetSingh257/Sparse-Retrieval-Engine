# 🎓 COL 7364/764 Assignment 1: Sparse Retrieval Engine
## Comprehensive Viva & Oral Defense Preparation Guide (Top 25 Q&As)

> **⚠️ Course Staff Warning (From Assignment PDF):**
> *"Leaderboard marks are capped at 50% of earned value if the oral defense indicates the team cannot explain their own submission."*

---

# 📑 TABLE OF CONTENTS
1. [Core Fundamentals & Data Structures (Q1 – Q5)](#1-core-fundamentals--data-structures-q1--q5)
2. [BM25 & Mathematical Formulas (Q6 – Q10)](#2-bm25--mathematical-formulas-q6--q10)
3. [Custom Scorer & Advanced Optimizations (Q11 – Q15)](#3-custom-scorer--advanced-optimizations-q11--q15)
4. [Storage, Index Size & Latency Optimization (Q16 – Q18)](#4-storage-index-size--latency-optimization-q16--q18)
5. [Docker, Conformance & Debugging Fixes (Q19 – Q21)](#5-docker-conformance--debugging-fixes-q19--q21)
6. [Evaluation Metrics & Leaderboard Math (Q22 – Q25)](#6-evaluation-metrics--leaderboard-math-q22--q25)

---

# 1. Core Fundamentals & Data Structures (Q1 – Q5)

### ❓ Q1: What is an Inverted Index, and why do we use it instead of a Forward Index?
* **Answer:**
  * A **Forward Index** maps `Document ID -> List of words`. Searching for a query term requires a slow linear scan of all $171,332$ documents.
  * An **Inverted Index** maps `Term -> Postings List {Doc ID: Term Frequency}`.
  * **Why we use it:** It provides instant $O(1)$ dictionary lookups to immediately retrieve only the subset of documents containing the query terms, skipping $99.9\%$ of irrelevant documents.

---

### ❓ Q2: What is Tokenization and Stemming? Which stemmer did you use?
* **Answer:**
  * **Tokenization:** Splitting continuous text into lowercase alphanumeric word tokens (`[a-z0-9]+`) while stripping punctuation and noise.
  * **Stemming:** Stripping grammatical suffixes (`-ing`, `-ed`, `-es`, `-s`) to reduce words to their base root (e.g., *"vaccinating"*, *"vaccinated"* $\rightarrow$ *"vaccin"*).
  * We used NLTK's **`PorterStemmer`** with an in-memory memoization cache (`_STEM_CACHE`) to avoid re-stemming identical words, speeding up indexing by $3\times$.

---

### ❓ Q3: What is Zipf's Law and how does it relate to your Inverted Index?
* **Answer:**
  * **Zipf's Law** states that the frequency of any word is inversely proportional to its rank in the frequency table ($f \propto 1/r$).
  * A few stop words (*"the"*, *"is"*, *"in"*) account for $>30\%$ of all word occurrences, while $>60\%$ of unique words appear only once (Hapax Legomena).
  * We leverage this by filtering high-frequency stop words to save scoring time and pruning rare noisy singletons to compress index size.

---

### ❓ Q4: What is the difference between Boolean Retrieval and Ranked Retrieval?
* **Answer:**
  * **Boolean Retrieval:** Uses exact binary logic (`AND`, `OR`, `NOT`). It returns an unranked set of matching documents (a document either matches or does not).
  * **Ranked Retrieval (VSM / BM25):** Assigns a continuous numerical relevance score to every matching document and sorts them, returning the Top-$K$ highest-scoring documents.

---

### ❓ Q5: What is the Vector Space Model (VSM) and Cosine Similarity?
* **Answer:**
  * VSM represents both queries and documents as high-dimensional vectors in a shared vocabulary space, weighted by $\text{TF} \times \text{IDF}$.
  * The relevance score is the **Cosine Angle** between the query vector $\vec{q}$ and document vector $\vec{d}$:
    $$\cos(\theta) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2} = \frac{\sum_{t \in Q \cap D} \text{TF}(t, Q)\text{IDF}(t) \cdot \text{TF}(t, D)\text{IDF}(t)}{\sqrt{\sum \text{weight}(q)^2} \sqrt{\sum \text{weight}(d)^2}}$$

---

# 2. BM25 & Mathematical Formulas (Q6 – Q10)

### ❓ Q6: Why is Robertson BM25 superior to classical TF-IDF?
* **Answer:** Classical $\text{TF} \times \text{IDF}$ has two major structural flaws:
  1. **Linear TF scaling:** Repeating a term 100 times gives 100x score (unrealistic).
  2. **Document Length Bias:** Long documents naturally accumulate high raw TF scores.
* **BM25 fixes both** by introducing **Term Frequency Saturation** ($k_1$) and **Length Normalization** ($b$).

---

### ❓ Q7: What does the $k_1$ parameter do in BM25, and what value did you choose?
* **Answer:**
  * **$k_1$ (Term Frequency Saturation parameter):** Controls how quickly the score saturates as TF increases.
  * When $\text{TF} \to \infty$, the term frequency component $\frac{\text{TF}(k_1+1)}{\text{TF} + \dots}$ asymptotes to $k_1 + 1$.
  * If $k_1 = 0$, TF is completely ignored (pure binary presence). If $k_1 \to \infty$, it behaves like linear TF.
  * **Our tuned value:** **`k1 = 1.45`**, found via 2D grid search on the dev set.

---

### ❓ Q8: What does the $b$ parameter do in BM25, and what value did you choose?
* **Answer:**
  * **$b$ (Length Normalization parameter):** Controls the penalty applied to long documents based on their ratio to average document length ($\frac{|D|}{\text{avgdl}}$).
  * If $b = 1.0$, full length normalization is applied (score is strictly penalized by document length).
  * If $b = 0.0$, length normalization is turned off completely.
  * **Our tuned value:** **`b = 0.38`**. A lower $b$ prevents over-penalizing comprehensive biomedical abstracts that naturally contain background context.

---

### ❓ Q9: What is BM25F, and why did you use it?
* **Answer:**
  * BM25F extends BM25 to support **structured Fields** (e.g., Title vs Body/Abstract).
  * In scientific research papers, the **Title** contains the core intent. An occurrence in the Title is significantly more informative than an occurrence buried in the body.
  * We assigned a **$4.2\times$ score weight to Title occurrences** relative to body text.

---

### ❓ Q10: How is IDF calculated in BM25?
* **Answer:**
  $$\text{IDF}(t) = \ln\left(\frac{N - \text{DF}(t) + 0.5}{\text{DF}(t) + 0.5} + 1.0\right)$$
  * Where $N = 171,332$ total documents, and $\text{DF}(t)$ is the number of documents containing term $t$.
  * The $+0.5$ provides Robertson-Spärck Jones smoothing to prevent division by zero, and $+1.0$ ensures IDF is always non-negative.

---

# 3. Custom Scorer & Advanced Optimizations (Q11 – Q15)

### ❓ Q11: What is Question Stopword Filtering, and why is it critical?
* **Answer:**
  * Many TREC-COVID queries are framed as natural language questions (e.g., *"What is the origin of COVID-19?"*, *"Are there guidelines for practice?"*).
  * Words like *"what"*, *"is"*, *"causes"*, *"evidence"*, *"practice"* appear across hundreds of papers and dilute the scoring.
  * We built a focused list of 60+ question stopwords to strip out conversational noise and keep only clinical keywords.

---

### ❓ Q12: How does your 50-Topic Clinical Thesaurus (Query Expansion) work?
* **Answer:**
  * Medical authors use diverse synonyms for the same concept (e.g., *"origin"* vs *"wuhan, bat, host, zoonotic"*; *"pediatric"* vs *"children, infant, MIS-C, kawasaki"*).
  * We implemented a curated 50-topic medical expansion dictionary.
  * When a seed term matches, synonyms are added to the query weighting map with a discounted weight ($w = 0.50$) to prevent semantic drift while capturing relevant papers that use alternate clinical terms.

---

### ❓ Q13: What is Power-Law IDF Weighting ($IDF^{1.35}$)?
* **Answer:**
  * Standard BM25 multiplies term weight linearly by IDF.
  * In medical retrieval, highly specific terms (*"remdesivir"*, *"dexamethasone"*, *"asymptomatic"*) carry far more discriminative intent than general medical terms (*"patient"*, *"study"*, *"disease"*).
  * Raising IDF to the power of $1.35$ exponentially elevates rare specific clinical terms over moderately common terms.

---

### ❓ Q14: What is the Sharp Multi-Term Coverage Boost ($coverage^{1.8}$)?
* **Answer:**
  * If a query has 4 keywords (e.g., *"remdesivir"*, *"treatment"*, *"covid"*, *"hospitalized"*), a document matching all 4 terms is vastly more relevant than a document mentioning *"covid"* 20 times and the others 0 times.
  * We calculate coverage ratio: $\text{cov} = \frac{\text{matched query terms}}{\text{total query terms}}$.
  * We apply a non-linear multiplier: $\text{Final Score} = \text{Base Score} \times (1.0 + 0.65 \times \text{cov}^{1.8})$.

---

### ❓ Q15: How does your Candidate Selection / Heap Ranking work?
* **Answer:**
  * Instead of sorting all $171,332$ documents (which is $O(N \log N)$), we accumulate candidate scores in a sparse dictionary and use Python's **`heapq.nlargest(k * 4, ...)`** (which runs in $O(M \log K)$).
  * We then apply coverage boosting only to the top candidates, yielding sub-20ms query latency.

---

# 4. Storage, Index Size & Latency Optimization (Q16 – Q18)

### ❓ Q16: How did you achieve an index size of 19.6 MB for 1.71 lakh documents?
* **Answer:**
  1. **Standard Library Binary Arrays:** Document IDs and term frequencies are stored in compact `array.array('I')` (4-byte unsigned integers) and `array.array('B')` (1-byte unsigned chars).
  2. **Gzip DEFLATE Compression:** Serialized using Python's highest pickle protocol + Gzip compression (`compresslevel=6`), achieving $< 20\text{ MB}$ on disk and securing the **Full 10/10 Index Size Marks**.

---

### ❓ Q17: How did you optimize query latency from 483ms down to < 20ms?
* **Answer:**
  1. Precomputed length normalization ratios `_DOC_L_RATIO[doc_id] = len / avg_dl` at load time.
  2. Precomputed all IDF values in `_IDF_CACHE`.
  3. Replaced slow nested Python dictionary lookups with local variable bindings and flat arithmetic loops.

---

### ❓ Q18: How fast is your Index Build on the full dataset?
* **Answer:**
  * The index build processes all **171,332 research papers in 117 seconds** ($\approx 1,460$ papers/sec).
  * This is **8x faster** than the official 900-second (15-minute) timeout limit.

---

# 5. Docker, Conformance & Debugging Fixes (Q19 – Q21)

### ❓ Q19: What was the Day 2 Docker build error, and how did you fix it?
* **Answer:**
  * **Error:** `error: Multiple top-level modules discovered in a flat-layout: ['indexer', 'bm25', 'custom_scorer', 'retrieve']`.
  * **Root Cause:** Modern `setuptools` ($\ge 61.0$) auto-discovers all `.py` files in flat directories and throws `PackageDiscoveryError` if `setup()` is called without explicit module lists.
  * **Fix:** In `submission/setup.py`, we added `py_modules=[]` and `packages=[]` to prevent setuptools auto-discovery crashes.

---

### ❓ Q20: What was the Day 3 packaging error, and how did you fix it?
* **Answer:**
  * **Error:** `ValueError: expected exactly one submission/retrieve.py, found 2`.
  * **Root Cause:** The zip archive had a duplicate `retrieve.py` at root level as well as inside `./2026MCS2250/submission/retrieve.py`.
  * **Fix:** Cleaned the packaging script to generate a single wrapper folder `./2026MCS2250/` with exactly ONE `submission/retrieve.py`.

---

### ❓ Q21: How do you guarantee offline execution in `--network none` mode?
* **Answer:**
  * In `submission/setup.py`, NLTK datasets (`punkt`, `stopwords`, `punkt_tab`) are downloaded during image build time.
  * Custom self-contained stopword sets in `custom_scorer.py` ensure 0 external network requests during grading.

---

# 6. Evaluation Metrics & Leaderboard Math (Q22 – Q25)

### ❓ Q22: What is the difference between Precision@10, MAP@10, and MRR?
* **Answer:**
  * **Precision@10:** Fraction of returned top-10 documents that are relevant ($\frac{\text{relevant in top 10}}{10}$).
  * **MRR (Mean Reciprocal Rank):** Evaluates where the *first* relevant document appears ($\frac{1}{\text{rank}_1}$).
  * **MAP@10 (Mean Average Precision):** Evaluates the entire ranking curve up to rank 10 by averaging precision at every relevant document position, normalized by total relevant documents.

---

### ❓ Q23: What is nDCG@10, and why is it the primary metric (70% weight)?
* **Answer:**
  * **nDCG@10 (Normalized Discounted Cumulative Gain):** Measures graded relevance with logarithmic rank discount:
    $$\text{DCG@10} = \sum_{i=1}^{10} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG@10} = \frac{\text{DCG@10}}{\text{IDCG@10}}$$
  * **Why it's primary:** In search engines, user attention drops rapidly down the page. A gold-standard relevant paper at Rank 1 gives high utility; at Rank 10 it gives low utility. nDCG explicitly rewards placing the best documents at top ranks.

---

### ❓ Q24: What is your score on the real 1.71 lakh dataset?
* **Answer:**
  * **nDCG@10:** **`0.6745`** (vs 0.6004 baseline).
  * **Precision@10:** **`0.7580`** (76% of top-10 hits are relevant).
  * **MRR:** **`0.8980`** (90% top-1 accuracy).
  * **Unit Tests:** **`17 / 17 PASSED (100%)`**.

---

### ❓ Q25: How does the final course grade breakdown work?
* **Answer:**
  * **Leaderboard Performance (50%):** Percentile score on held-out private test set.
  * **Correctness of Baseline Components (20%):** `boolean_vsm.py` and `bm25.py` verified via unit tests.
  * **Report Quality (20%):** Methodology, parameter sweeps, and failure analysis (Due Sept 6).
  * **Interface Conformance (5%):** Passing CI smoke test and offline Docker container execution.
  * **Code Quality & Reproducibility (5%):** Clean, deterministic execution from README.
