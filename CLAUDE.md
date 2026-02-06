# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**jays-philnits-vault** is a PhilNITS exam analysis pipeline that extracts, cleans, analyzes, and vectorizes 20+ years of Philippine National IT Scholarship exam questions. The goal is to identify patterns, cluster questions by cognitive type (not topic), and surface curriculum gaps and trivia.

**Core Philosophy (from INTENT.md):**
- Top-down, intuition-first analysis, not rote memorization
- Cluster questions by *thinking type* required (definition recall, state evolution, resource tradeoff, algorithm reasoning), not by subject
- Build decision engines: fast recognition layer (90%) → deliberate reasoning layer (DSA/tracing) → exception handling (curriculum gaps + trivia)

## Data Pipeline (Sequential, 5 Stages)

```
PDFs (exams_and_answers)
  ↓
1. extract_exams.py         → data/extracted_json/        (raw text from PDFs)
  ↓
2. clean_exam_data.py       → data/cleaned_json/          (remove markers, normalize)
  ↓
3. refine_exam_data.py      → data/refined_json/          (remove option bleed)
  ↓
4. vectorize_exam_data.py   → data/vectorized/            (SBERT embeddings + metadata)
  ↓
5. analyze_exam_patterns.py → analysis/                   (clustering, keywords, study guide)
```

Each stage is independent; outputs feed into the next. All scripts are in `scripts/`. Data flows through `data/` subdirectories.

## Common Commands

### Activating the environment
```bash
source /home/jay/philnits-venv/bin/activate
```

### Running the full pipeline (all 5 stages)
```bash
cd "/home/jay/Desktop/Coding Stuff/jays-philnits-vault"
python3 scripts/extract_exams.py && \
python3 scripts/clean_exam_data.py && \
python3 scripts/refine_exam_data.py && \
python3 scripts/vectorize_exam_data.py && \
python3 scripts/analyze_exam_patterns.py
```

### Running individual stages
```bash
# Extract PDFs to raw JSON
python3 scripts/extract_exams.py

# Clean extracted data (remove page markers, normalize whitespace)
python3 scripts/clean_exam_data.py

# Refine questions (remove option text bleeding into question field)
python3 scripts/refine_exam_data.py

# Vectorize questions (compute SBERT embeddings)
python3 scripts/vectorize_exam_data.py

# Analyze patterns (clustering, keyword extraction, domain mapping)
python3 scripts/analyze_exam_patterns.py
```

### Checking script syntax (no execution)
```bash
python3 -m py_compile scripts/*.py
```

## Architecture & Key Components

### Stage 1: Extract (extract_exams.py)
- **Class:** `ExamExtractor`
- **Input:** PDF files from `exams_and_answers/questions/` and `exams_and_answers/answers/`
- **Output:** `data/extracted_json/{exam_name}.json`
- **Logic:**
  - Pairs question PDFs with answer PDFs (handles variants: `2013_april` vs `2013_april_variant`)
  - Parses answers (single-page key sheet) → maps Q# to answer letter (a/b/c/d)
  - Extracts question text from question PDFs using regex-based parsing
  - Produces raw JSON with `questions[]` (text, choices, answer, explanation)
- **Dependencies:** `pdfplumber` (PDF text extraction)

### Stage 2: Clean (clean_exam_data.py)
- **Class:** `ExamDataCleaner`
- **Input:** `data/extracted_json/` (raw PDFs can have artifacts)
- **Output:** `data/cleaned_json/`
- **Procedures:**
  - `remove_page_markers()`: Strips `– # –` patterns (page footer artifacts)
  - `normalize_whitespace()`: Collapses newlines/tabs → spaces, removes redundant whitespace
  - `detect_truncation()`: Flags suspicious endings (e.g., incomplete words, "et")
  - Validates JSON structure
- **Dependencies:** None beyond Python stdlib

### Stage 3: Refine (refine_exam_data.py)
- **Class:** `ExamDataRefiner`
- **Input:** `data/cleaned_json/`
- **Output:** `data/refined_json/`
- **Problem Solved:** Some extracted questions have option text bleeding in: `"What is X? a) foo b) bar c) baz d) qux"` instead of `"What is X?"`
- **Detection:** Regex pattern `a\)\s*.*?\s*b\)\s*.*?\s*c\)\s*.*?\s*d\)` finds all four options in sequence
- **Fix:** Truncate question text before first `"a)"`
- **Dependencies:** None beyond Python stdlib

### Stage 4: Vectorize (vectorize_exam_data.py)
- **Class:** `ExamVectorizer`
- **Input:** `data/refined_json/` (clean questions)
- **Output:**
  - `data/vectorized/vectors.npz` (NumPy sparse/dense matrix of embeddings)
  - `data/vectorized/metadata.json` (mapping: vector index → question metadata)
- **Model:** Sentence-Transformers `all-MiniLM-L6-v2` (small, fast semantic embeddings)
- **Logic:**
  - Loads all questions from refined JSON files
  - Encodes each question text via SBERT
  - Saves embeddings as NumPy arrays
  - Saves metadata (exam, q_num, question text, answer, choices)
- **Dependencies:** `sentence-transformers`, `numpy`

### Stage 5: Analyze (analyze_exam_patterns.py)
- **Class:** `ExamPatternAnalyzer`
- **Input:**
  - `data/vectorized/vectors.npz` (embeddings)
  - `data/vectorized/metadata.json` (question info)
  - `data/refined_json/*.json` (original text for keyword extraction)
- **Output:**
  - `analysis/exam_analysis_agent.json` (machine-readable JSON for LLM processing)
  - `analysis/exam_study_guide.md` (human-readable study guide)
  - `logs/analysis_YYYYMMDD_HHMMSS.log` (detailed analysis log)
- **Core Procedures:**
  - `cluster_questions()`: K-means clustering (K ∈ {6, 7, 8}), selects K by silhouette score
  - `extract_keywords()`: TF-IDF per cluster, extracts top N keywords
  - `classify_question_types()`: Regex-based classification (DEFINITION_RECALL, SCENARIO_APPLICATION, COMPUTATION, DIAGRAM_ANALYSIS, GENERAL)
  - `estimate_difficulty()`: Heuristic (text length + option count) → LOW/MEDIUM/HIGH
  - `extract_industry_vocabulary()`: Matches 42 predefined terms (20 foundational, 22 advanced) across questions
  - `map_domains()`: Assigns clusters to 7 domains (OS_KERNEL, NETWORKING, SECURITY, DATABASES, STORAGE, ALGORITHMS, MISC) via keyword overlap
- **Determinism:** `random_state=42` for K-means; all operations are reproducible
- **Dependencies:** `numpy`, `scikit-learn`, `sentence-transformers` (implicitly via vectors)

### Analysis Output Format

**`exam_analysis_agent.json`** (programmatic access):
```json
{
  "metadata": {
    "generated": "ISO timestamp",
    "total_questions": int,
    "exams_analyzed": int,
    "clusters": int,
    "model": "all-MiniLM-L6-v2"
  },
  "clusters": [
    {
      "id": int,
      "size": int,
      "primary_domain": string,
      "silhouette_score": float,
      "top_keywords": [{"word": str, "tf_idf": float, "frequency": int}, ...],
      "question_types": {"DEFINITION_RECALL": int, ...},
      "difficulty_distribution": {"LOW": int, "MEDIUM": int, "HIGH": int},
      "sample_questions": [{"q_num": int, "exam": str, "text": str}, ...]
    }
  ],
  "domain_summary": {"OS_KERNEL": {"count": int, "clusters": [int, ...]}, ...},
  "industry_vocabulary": {
    "FOUNDATIONAL": {"RAID": {"definition": str, "occurrences": int, "clusters": [int, ...]}, ...},
    "ADVANCED": {...}
  },
  "insights": [string, ...]
}
```

**`exam_study_guide.md`** (human-readable):
- Quick stats (total questions, exam count, cluster count)
- Topics to study (ranked by frequency)
- Industry vocabulary (foundational + advanced)
- Study plan by difficulty
- Question type distribution
- Exam pattern observations

## Key Design Decisions

### Two-Output Analysis
- `exam_analysis_agent.json`: Programmatic, LLM-processable
- `exam_study_guide.md`: Human-friendly for reading and planning

### K-means + Silhouette
- Deterministic with fixed seed
- Silhouette score is interpretable (−1 to +1)
- Works well with SBERT embeddings (semantic similarity)
- K range (6–8) was empirically determined

### Baked-In Industry Vocabulary
- Explicit 42-term list (not auto-discovered)
- Reproducible across runs
- Human-reviewable definitions
- Split into foundational (must-know) vs. advanced (should-know)

### Simple Difficulty Heuristics
- No labeled training data required
- Interpretable (text length + option count)
- Good enough for ranking; transparent to users

### Keyword-Based Domain Mapping
- Deterministic (no ML model)
- Fast execution
- Graceful fallback to "MISC"

## Important Constraints

- **Do NOT edit or delete `exams_and_answers/`** — this is the source of truth (PDFs)
- **Do NOT manually edit `data/extracted_json/`** — always regenerate via `extract_exams.py` if PDFs change
- **Outputs are overwritten on each run** — no incremental updates; pipeline is idempotent
- **Vectorization is time-consuming** — embedding ~1500 questions takes a few minutes; only run when needed

## Repo Structure (Read-Only Directories)

```
exams_and_answers/        [SOURCE] PDF files (questions + answers)
data/extracted_json/      [GENERATED] Raw PDFs → JSON
data/cleaned_json/        [GENERATED] After cleaning
data/refined_json/        [GENERATED] After refinement
data/vectorized/          [GENERATED] SBERT embeddings + metadata
analysis/                 [GENERATED] Clustering results + study guide
logs/                     [GENERATED] Logs from each stage
```

## Memory & Performance Notes

- Full pipeline (extract → analyze) takes ~5–10 minutes
- Vectorization (stage 4) is the slowest (~3–5 min for ~1500 questions)
- Clustering (stage 5) is fast (~< 1 sec for ~1500 questions)
- Memory: Vectorized outputs are ~10–20 MB (numpy + JSON)

## Testing & Verification

- Syntax check all scripts: `python3 -m py_compile scripts/*.py`
- Run individual stages to verify each step
- Compare outputs across runs (deterministic, so should be byte-for-byte identical except timestamps)
- Check logs for warnings/errors: `logs/extraction_*.log`, `logs/analysis_*.log`

## Future Extensions (Not Yet Implemented)

- Custom industry vocabulary (parameterizable list)
- Interactive difficulty adjustment (human labeling)
- Temporal analysis (how patterns change across exam years)
- Prerequisite graphs (which domains depend on others)
- Performance metrics (clustering quality vs. ground truth)
- Export formats (CSV, LaTeX, HTML)

## Development Notes

- All scripts use `pathlib.Path` for cross-platform compatibility
- Logging goes to both file and console (configurable verbosity)
- Classes are instantiable with custom input/output directories (useful for testing)
- No external APIs or network calls; fully reproducible locally
