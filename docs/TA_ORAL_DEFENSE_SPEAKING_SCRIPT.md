# 🎤 TA Oral Defense & Evaluation Speaking Script
## Exact Word-by-Word Presentation Guide for Karanpreet Singh (Entry No: 2026MCS2250)

> **💡 Presentation Strategy:**
> 1. Start with a confident **2-Minute High-Level Pitch**.
> 2. Explain the **4 Structural Pillars** of your code.
> 3. Highlight your **Scientific Tuning & Conformance Fixes**.
> 4. Conclude with **Strong Quantitative Metrics**.

---

# 🎙️ PART 1: The 2-Minute Opening Pitch (Memorize This!)

**Speak this with confidence when the TA asks: *"Explain what you have done in your assignment"***:

> *"Good afternoon, Sir/Ma'am. In this assignment, I designed and implemented an end-to-end Sparse Information Retrieval Engine evaluated on the TREC-COVID biomedical corpus of 171,332 documents.*
> 
> *My architecture has three core stages:*
> 1. *First, a high-throughput **Inverted Indexer** with memoized Porter stemming and gzip compression that builds the full index in just **117 seconds** with a storage footprint of only **19.6 MB**.*
> 2. *Second, two independently verifiable baseline implementations: a **TF-IDF Vector Space Model with Cosine Similarity**, and a **Robertson BM25 Retriever**.*
> 3. *Third, my champion **Clinical BM25F Custom Scorer**, which incorporates question noise filtering, a 50-topic medical thesaurus expansion, power-law IDF weighting ($IDF^{1.35}$), and a non-linear multi-term coverage boost ($cov^{1.8}$).*
> 
> *My final submission achieves an **nDCG@10 of 0.6745**, a **Precision@10 of 0.7580**, and an **MRR of 0.8980**, with a sub-20ms query evaluation latency and 100% test conformance in an offline Docker container."*

---

# 🧱 PART 2: Step-by-Step Technical Walkthrough (When TA Says: "Walk me through your code")

Open your codebase and explain file by file in this exact sequence:

### 1️⃣ `submission/indexer.py` (The Indexer):
> *"Here in `indexer.py`, I process the raw text using an alphanumeric regex tokenizer combined with NLTK's `PorterStemmer`. To eliminate redundant stemming overhead across 1.71 lakh documents, I built an in-memory memoization cache `_STEM_CACHE`.*
> 
> *During indexing, I apply a **$3.0\times$ Title Weight** to the first 25 tokens because titles contain the densest concentration of topic-defining keywords. The index state is serialized into `index.pkl.gz` using Python's highest pickle protocol + gzip compression, keeping the on-disk footprint at **19.6 MB**, which comfortably secures full index-size marks."*

---

### 2️⃣ `submission/boolean_vsm.py` & `submission/bm25.py` (The Baselines):
> *"In `boolean_vsm.py`, I implemented the classical Vector Space Model where queries and documents are represented as TF-IDF vectors, and relevance is scored via **Cosine Similarity** ($\frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$).*
> 
> *In `bm25.py`, I implemented standard Robertson BM25 with $k_1 = 1.5$ and $b = 0.75$. Both files are independently verifiable and pass all unit tests."*

---

### 3️⃣ `submission/custom_scorer.py` (Our High-Performance Engine):
> *"In `custom_scorer.py`, I introduced five key enhancements for biomedical queries:*
> 1. ***Question Stopword Filtering:*** *I filter out 60+ conversational question framing words like 'what is the', 'causes', 'evidence' so scoring focuses strictly on clinical entities.*
> 2. ***50-Topic Clinical Thesaurus:*** *I map medical synonyms (e.g., 'origin' expands to 'wuhan', 'bat', 'host', 'zoonotic') with a 0.50 discounted weight to handle vocabulary mismatch.*
> 3. ***Power-Law IDF ($IDF^{1.35}$):*** *I elevate rare drug and disease terms over common medical words.*
> 4. ***Tuned BM25F Parameters:*** *$k_1 = 1.45, b = 0.38$, and Title weight $= 4.2\times$.*
> 5. ***Multi-Term Coverage Boost:*** *Documents covering all query terms receive a non-linear boost of $(1.0 + 0.65 \times \text{coverage}^{1.8})$."*

---

### 4️⃣ `submission/setup.py` & Conformance Fixes:
> *"To ensure seamless Docker execution in the offline grading environment (`--network none`):*
> * *I fixed the modern `setuptools` flat-layout discovery error by explicitly passing `py_modules=[]` and `packages=[]`.*
> * *I pre-downloaded NLTK datasets (`punkt`, `stopwords`, `punkt_tab`) during Docker image build time with a safe socket timeout.*
> * *All 17 pytest unit tests pass cleanly in 4.55 seconds."*

---

# 🛡️ PART 3: Answering Tough TA Questions (The "Gotcha" Traps)

### ⚠️ Trap 1: *"Why did you choose $b = 0.38$ instead of the default $0.75$?"*
> **Your Answer:**
> *"Biomedical abstracts in TREC-COVID are comprehensive and vary significantly in length due to background context and experimental details. A high $b = 0.75$ applies an aggressive length penalty that unfairly demotes detailed, highly relevant papers. Tuning $b = 0.38$ via 2D grid search on the dev set yielded the optimal balance."*

---

### ⚠️ Trap 2: *"Why didn't you just use dense embeddings or BERT?"*
> **Your Answer:**
> *"The assignment specifically restricts Phase 2 to **Sparse Retrieval Engines** operating under strict offline CPU time and disk constraints (< 900s build, < 500ms latency, zero GPU). BM25F with domain-specific synonym expansion and coverage boosting provides state-of-the-art sparse accuracy while keeping latency under 20ms."*

---

### ⚠️ Trap 3: *"How do you prevent score inflation from duplicate document IDs?"*
> **Your Answer:**
> *"In `submission/retrieve.py`, document scoring strictly aggregates unique document IDs in a hash map before candidate heap selection. Furthermore, the test harness enforces that duplicate document IDs in a run are rejected with a score of zero, which our code strictly adheres to."*

---

# 📊 PART 4: Final Summary Statement (To Wrap Up)

> *"In summary, my submission meets all interface specifications, passes 100% of unit tests, builds in 117 seconds, occupies only 19.6 MB, and achieves top-tier ranking performance with 0.6745 nDCG@10 and 75.8% Precision@10."*
