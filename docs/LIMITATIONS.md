# Limitations of Current PhilNITS Exam Analysis System

**Date:** 2026-02-08
**Context:** Analysis of limitations discovered during development and conversation with user

---

## 1. Data Extraction Quality

### 1.1 Incomplete Question Extraction (74% Coverage)
**Issue:** Text-based PDF extraction only captures ~75% of questions (1,472 / ~2,000 expected).

**Root causes:**
- Questions with embedded diagrams/figures are skipped (no OCR)
- Page boundary issues cause question truncation
- Answer key mismatches (some exams have variant answer sheets)

**Impact:**
- Missing ~500 questions across 20 exams
- Biased toward text-only questions (under-represents visual/diagram analysis)
- Potential domain skew (networking diagrams, ER diagrams, flowcharts missing)

**Mitigation (not implemented):**
- OCR with PaddleOCR or Tesseract for figure extraction
- Manual verification of extraction vs. answer key counts
- Flag exams with <70% extraction for manual review

---

### 1.2 Option Text Contamination
**Issue:** Some questions have answer choices bleeding into question text field.

**Example:**
```
Question: "What is X? a) foo b) bar c) baz d) qux"
Should be: "What is X?"
```

**Status:** Partially mitigated by `refine_exam_data.py` (regex detection + truncation), but not guaranteed 100% clean.

**Impact:**
- Embeddings include noise from answer choices
- Keyword extraction may pick up distractor terms

---

### 1.3 No Answer Explanation Data
**Issue:** PDFs contain questions + answers, but no explanations for *why* answer is correct.

**Impact:**
- Cannot generate explanations automatically
- Cannot validate reasoning patterns
- Students get "correct answer" but not understanding

**Workaround:** Would require manual labeling or external knowledge base (textbook chapters, Stack Overflow, etc.)

---

## 2. Clustering & Keyword Extraction

### 2.1 Poor Clustering Quality (Silhouette Score = 0.03)
**Issue:** K-means clustering produces low-quality groupings. Silhouette score of 0.03 indicates questions barely cluster better than random.

**Root causes:**
- Questions are semantically diverse (not naturally grouping)
- SBERT embeddings are generic (trained on general text, not CS exams)
- Optimal K=8 is a guess (no clear elbow in silhouette curve)

**Impact:**
- Clusters are not semantically coherent
- "Networking" cluster (77%) is a catch-all garbage bin
- Keywords extracted from clusters are not distinctive

**Alternatives (not tried):**
- HDBSCAN (density-based, auto-detects K)
- Topic modeling (LDA, NMF) for soft clustering
- Fine-tune SBERT on CS exam corpus
- Abandon clustering entirely (questions may not naturally cluster)

---

### 2.2 Keyword Extraction Captures Averages, Not Distinctives
**Issue:** Initial TF-IDF approach extracted generic exam language ("appropriate", "following", "explanation") instead of technical terms.

**Root cause:** TF-IDF finds terms common *within cluster*, not terms *distinctive to cluster*.

**Status:** Partially fixed (replaced with domain keyword frequency + rare n-gram extraction), but still limited by:
- Domain keywords are hand-curated (only 100+ terms, may miss emerging topics)
- Rare n-grams require manual review (false positives like "000" from OCR artifacts)
- No validation that extracted keywords are actually meaningful

**Better approach:** Class-based TF-IDF (term frequency in cluster / term frequency in other clusters) or chi-squared feature selection.

---

### 2.3 Domain Mapping is Overly Simplistic
**Issue:** Questions assigned to domains via keyword overlap. If question mentions "TCP", it's "Networking". If it mentions "database", it's "Databases".

**Problems:**
- Multi-domain questions (e.g., "SQL injection in web application") forced into single category
- Domain overlap is high (networking + security, OS + algorithms)
- 77% of questions labeled "Networking" is suspiciously high (likely misclassification)

**Better approach:** Multi-label classification (questions can belong to multiple domains) or hierarchical clustering.

---

## 3. Industry Vocabulary Extraction

### 3.1 Only 7/42 Terms Found
**Issue:** Predefined list has 42 industry terms (20 foundational + 22 advanced), but only 7 found in questions.

**Root causes:**
- Terms are too specific (e.g., "side-channel attack", "TPM", "NUMA" may not appear in introductory IT exams)
- Case-sensitive matching for acronyms too strict (e.g., "BASE" requires exact uppercase, misses lowercase)
- Terms defined based on industry usage, not exam vocabulary

**Impact:**
- Vocabulary section is underwhelming (only 7 terms)
- Missing terms students actually need to know

**Fix:**
- Reverse approach: Extract frequent terms from corpus, filter for technical concepts, THEN define them
- Expand search to include partial matches (e.g., "attack" captures "side-channel attack")

---

### 3.2 False Positives (BASE = 85 Occurrences)
**Issue:** "BASE" (NoSQL consistency model) matched "base" in common English text ("database", "based on"), inflating count to 85.

**Status:** Fixed (case-sensitive matching for acronyms), but illustrates broader problem of ambiguous terms.

---

## 4. Difficulty Estimation

### 4.1 Heuristic is Garbage
**Issue:** Current difficulty metric is `(text_length × 2) + (option_count × 10)`, which classifies most questions as "HIGH" difficulty.

**Why this is bad:**
- Text length ≠ cognitive complexity
- Long questions can be trivial (verbose definition recall)
- Short questions can be hard (terse algorithm problems)
- Ignores question type, domain, reasoning required

**Impact:**
- Difficulty labels are meaningless
- Study guide says everything is "High Difficulty" (unhelpful)

**Better approach:**
- Use cognitive taxonomy classification (Phase A: Definition Recall = Low, Phase C: Design = High)
- Use historical pass rates (requires answer data from test-takers)
- Use expert ratings (manual labeling)

---

## 5. Temporal Analysis

### 5.1 No Exam Evolution Tracking
**Issue:** Current analysis treats all 20 exams as a uniform corpus, ignoring temporal dimension.

**Missing insights:**
- Are recent exams (2020-2025) harder than older ones (2013-2015)?
- Which topics are NEW in modern exams (cloud, containers, zero-trust)?
- Which topics are DEPRECATED (legacy protocols, outdated standards)?

**Workaround:** Would require adding exam year as feature, comparing keyword/domain distributions over time.

---

## 6. Cognitive Complexity Not Measured

### 6.1 No Reasoning Type Classification
**Issue:** Current analysis categorizes questions by topic (networking, databases) but not by *how* they test knowledge.

**What's missing:**
- Definition recall vs. system behavior vs. optimization vs. failure analysis
- Multi-hop reasoning (A → B → C chains)
- State evolution questions (trace algorithm step-by-step)

**Impact:**
- Students don't know *what type of thinking* to practice
- Study guide says "study networking" but doesn't specify "practice TCP handshake tracing" vs. "memorize port numbers"

**Solution:** See `docs/plans/COGNITIVE_TAXONOMY.md` (Thesis A)

---

## 7. No Prerequisite Structure

### 7.1 No Learning Pathway
**Issue:** Current analysis shows *what* topics appear, but not *in what order* they should be learned.

**Missing:**
- Prerequisite relationships (must learn A before B)
- Foundational vs. advanced concepts
- Dependency graph (which concepts depend on others)

**Impact:**
- Students don't have a study roadmap
- May attempt advanced questions before learning prerequisites

**Solution:** See `docs/plans/PREREQUISITE_GRAPH.md` (Thesis C)

---

## 8. Study Guide Output

### 8.1 Underwhelming Insights
**Issue:** Study guide is generic and surface-level.

**What's provided:**
- "Networking: 77% of questions" (not actionable)
- "Focus: algorithm, authentication, cache" (generic keywords)
- "Study networking first" (obvious, unhelpful)

**What's missing:**
- Specific topics per cluster (e.g., "TCP 3-way handshake", "Dijkstra's algorithm")
- Representative question examples with explanations
- Weak spot identification (topics with low coverage but high importance)
- Time allocation strategy (computation vs. recall questions)

---

### 8.2 No Personalization
**Issue:** Study guide is one-size-fits-all. Doesn't adapt to student's current knowledge.

**Missing:**
- "You know networking, focus on databases" (requires student profiling)
- "Practice State Evolution questions" (requires cognitive taxonomy)
- "Learn X before Y" (requires prerequisite graph)

---

## 9. Technical Debt

### 9.1 No Question ID Stability
**Issue:** Questions are referenced by (exam_name, q_num) but these are not stable identifiers.

**Problem:** If PDF extraction changes, question numbering shifts, breaking references.

**Better approach:** Generate stable UUIDs or hash-based IDs per question.

---

### 9.2 No Version Control for Data
**Issue:** Refined JSON files are overwritten on each run. No history of extraction/cleaning changes.

**Impact:** Can't rollback to previous extraction if refinement introduces bugs.

**Better approach:** Version data outputs (e.g., `refined_json_v2.0/`), track provenance.

---

### 9.3 Hardcoded Paths
**Issue:** Scripts use hardcoded directory paths (`data/refined_json/`, `analysis/`).

**Problem:** Not easily runnable in different environments.

**Better approach:** Configuration file (YAML/JSON) or command-line arguments.

---

## 10. Evaluation & Validation

### 10.1 No Ground Truth Comparison
**Issue:** Analysis outputs (clusters, keywords, domains, difficulty) have no gold standard for validation.

**Missing:**
- Expert-labeled dataset (CS instructors rating difficulty, assigning domains)
- Comparison against reference materials (textbook chapter mappings)
- User study (do students find analysis helpful?)

**Impact:** Cannot measure if analysis is actually correct or useful.

---

### 10.2 No Error Metrics
**Issue:** No quantitative measure of extraction/classification quality.

**Missing:**
- Extraction recall (% of actual questions captured)
- Clustering validity indices beyond silhouette (Davies-Bouldin, Calinski-Harabasz)
- Keyword relevance scores (human judgment on extracted terms)

---

## 11. Scalability

### 11.1 Single Corpus Only
**Issue:** System designed for PhilNITS exams only. Not generalizable.

**Limitations:**
- Hardcoded domain keywords (CS/IT specific)
- PDF extraction tailored to PhilNITS format
- No support for other exam types (GRE, SAT, CISSP, etc.)

**Generalization requires:**
- Domain-agnostic keyword extraction
- Pluggable PDF parsers for different exam formats
- Configurable taxonomy definitions

---

### 11.2 Limited Exam Coverage
**Issue:** 20 exams over 12 years is small for robust statistical analysis.

**Impact:**
- Temporal trends hard to detect (need 50+ exams)
- Rare topics under-sampled
- Clustering may overfit to specific exam quirks

---

## 12. Ethical & Bias Considerations

### 12.1 No Bias Auditing
**Issue:** No analysis of whether exam questions are fair, equitable, or culturally neutral.

**Potential biases:**
- Cultural assumptions in question phrasing
- Gender/ethnic stereotypes in scenario questions
- Socioeconomic bias (assumes access to specific technologies)

**Impact:** System perpetuates any biases present in original exams.

---

### 12.2 Accessibility Not Addressed
**Issue:** No support for students with disabilities.

**Missing:**
- Screen reader compatibility for study guide
- Alternative formats (audio, large print, braille)
- Diagram descriptions for visually impaired

---

## Summary of Critical Gaps

| Limitation | Impact | Severity | Mitigation Plan |
|------------|--------|----------|-----------------|
| 74% extraction rate | Missing ~500 questions | HIGH | OCR implementation |
| Clustering quality (0.03) | Meaningless groupings | HIGH | Try HDBSCAN, topic models |
| Difficulty heuristic is bad | Unhelpful labels | MEDIUM | Use cognitive taxonomy |
| No cognitive taxonomy | Can't classify reasoning types | HIGH | Thesis A |
| No prerequisite graph | No learning pathway | HIGH | Thesis C |
| Only 7/42 industry terms found | Vocabulary section weak | MEDIUM | Reverse extraction approach |
| Study guide is generic | Not actionable | MEDIUM | Add examples, personalization |
| No temporal analysis | Miss exam evolution trends | LOW | Add time series features |
| No validation | Can't measure correctness | MEDIUM | Expert labeling study |

---

**Conclusion:**

The current system is a **proof-of-concept** with significant limitations. It demonstrates feasibility of automated exam analysis but lacks:
1. **Quality:** Extraction, clustering, and classification are noisy
2. **Depth:** Surface-level keyword analysis, no cognitive/prerequisite structure
3. **Validation:** No ground truth comparison or user evaluation

The two thesis tracks (Cognitive Taxonomy + Prerequisite Graph) address the core limitation: **current analysis describes WHAT is in exams, not HOW to learn from them.**

---

**Next Steps:**
1. Acknowledge these limitations in thesis introduction/discussion sections
2. Prioritize fixes based on severity and feasibility
3. Position theses as addressing the most critical gaps (cognitive + prerequisite)

**Author:** Generated during session 2026-02-08
**Last updated:** 2026-02-08
