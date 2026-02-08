# Prerequisite Knowledge Graph Extraction from PhilNITS Exams

## Research Question

**Can we automatically infer prerequisite relationships between concepts by analyzing semantic dependencies in exam question sequences?**

## Motivation

Students often struggle with exam prep because they lack a structured learning pathway. Textbooks provide chapter orderings, but exams test concepts in unpredictable sequences. Questions assume prior knowledge without making it explicit:

- Q47: "What is the result of a TCP retransmission timeout?"
  - Requires knowing: TCP basics, timeouts, retransmission mechanism
- Q23: "Which transport protocol guarantees delivery?"
  - Prerequisite for Q47

Current analysis doesn't capture these implicit dependencies. A **prerequisite knowledge graph** would:
1. Reveal which concepts must be learned before others
2. Generate optimal study pathways (topological sort)
3. Identify foundational vs. advanced concepts

---

## Objective

Build an automated system to extract prerequisite relationships between exam concepts, constructing a directed acyclic graph (DAG) where edges represent "must learn A before B" dependencies.

---

## Approach Overview

### Three-Phase Pipeline:

1. **Concept Extraction**: Identify atomic concepts in each question
2. **Dependency Inference**: Detect prerequisite relationships via multiple signals
3. **Graph Construction & Validation**: Build DAG, prune spurious edges, validate against expert knowledge

---

## Phase 1: Concept Extraction

**Timeline:** 1-2 weeks
**Deliverable:** Concept-to-question mapping (1,472 questions → 200-300 unique concepts)

### Method A: Entity Recognition (Domain Keywords)
Use existing `DOMAIN_KEYWORDS` dictionary (100+ technical terms):
- Extract all domain keywords present in each question
- Map: question_id → [list of concepts]

**Example:**
```
Q47: "TCP retransmission timeout causes exponential backoff"
Concepts: ['TCP', 'retransmission', 'timeout', 'exponential backoff']
```

### Method B: Keyphrase Extraction (YAKE/KeyBERT)
For concepts not in predefined dictionary:
- Run unsupervised keyphrase extraction on each question
- Filter: phrases that appear in 3+ questions (not hapax legomena)
- Cluster similar phrases (e.g., "binary search" ≈ "binary search tree")

**Tools:**
- **YAKE**: Unsupervised keyword extraction (no training needed)
- **KeyBERT**: BERT-based keyphrase extraction using embeddings

### Hybrid Approach (Recommended):
1. Extract domain keywords (guaranteed technical terms)
2. Run YAKE on remaining text to catch novel concepts
3. Manually review top 50 extracted phrases, add to concept list
4. Final output: 200-300 unique concepts, each linked to questions

---

## Phase 2: Dependency Inference

**Timeline:** 2-3 weeks
**Deliverable:** Weighted directed graph (concepts as nodes, dependencies as edges)

### Multiple Signals for Prerequisite Detection:

#### Signal 1: Co-occurrence + Temporal Ordering
**Hypothesis:** If concept A appears in earlier exams than B, and they co-occur frequently, A likely precedes B.

**Algorithm:**
```python
for concept_a in concepts:
    for concept_b in concepts:
        if concept_a == concept_b:
            continue

        # Find questions containing both
        cooccur_questions = questions_with_both(concept_a, concept_b)

        # Check temporal ordering (average exam year)
        avg_year_a = mean([exam_year(q) for q in questions_with(concept_a)])
        avg_year_b = mean([exam_year(q) for q in questions_with(concept_b)])

        if avg_year_a < avg_year_b and len(cooccur_questions) >= 3:
            add_edge(concept_a -> concept_b, weight=len(cooccur_questions))
```

#### Signal 2: Semantic Similarity (Embedding Distance)
**Hypothesis:** Prerequisite concepts are semantically related but not identical.

**Algorithm:**
```python
# For each concept, find questions that mention it
concept_embeddings = {}
for concept in concepts:
    concept_questions = questions_with(concept)
    concept_embeddings[concept] = mean([embedding(q) for q in concept_questions])

# Build similarity matrix
for concept_a in concepts:
    for concept_b in concepts:
        similarity = cosine_similarity(concept_embeddings[concept_a], concept_embeddings[concept_b])

        # Sweet spot: related but not identical (0.6 < sim < 0.85)
        if 0.6 < similarity < 0.85:
            add_edge(concept_a -> concept_b, weight=similarity)
```

#### Signal 3: Lexical Complexity
**Hypothesis:** Simple terms (short, common) precede complex terms (long, technical).

**Algorithm:**
```python
def complexity_score(concept):
    return (
        len(concept.split())  # Multi-word = complex
        + (1 if concept.isupper() else 0)  # Acronym = complex
        - (1 if concept in basic_vocab else 0)  # Common word = simple
    )

for concept_a, concept_b in concept_pairs:
    if complexity_score(concept_a) < complexity_score(concept_b):
        add_edge(concept_a -> concept_b, weight=complexity_diff)
```

#### Signal 4: Textbook Chapter Ordering (External Validation)
**Hypothesis:** If we have a reference textbook (e.g., Tanenbaum's Computer Networks), concepts that appear in earlier chapters precede later ones.

**Algorithm:**
```python
# Manual mapping: concept -> textbook chapter number
textbook_order = {
    'TCP': 6,
    'congestion control': 7,
    'routing': 5,
    ...
}

for concept_a, concept_b in concept_pairs:
    if textbook_order[concept_a] < textbook_order[concept_b]:
        add_edge(concept_a -> concept_b, weight=10)  # High confidence
```

---

### Combining Signals: Weighted Voting

Each signal produces edge weights. Combine via weighted average:
```python
edge_weight(A -> B) = (
    0.3 * cooccurrence_weight
    + 0.3 * semantic_weight
    + 0.2 * complexity_weight
    + 0.2 * textbook_weight
)
```

Threshold: Keep edges with `weight > 0.5` (tunable).

---

## Phase 3: Graph Construction & Validation

**Timeline:** 1-2 weeks
**Deliverable:** Prerequisite DAG with 200-300 nodes, validated edges

### Step 1: Build Initial Graph
- Nodes: Concepts
- Directed edges: A → B (A is prerequisite for B)
- Edge weights: Combined signal scores

### Step 2: Prune Spurious Edges
**Problem:** Raw graph may have cycles (A → B → C → A) or transitive bloat.

**Solutions:**
1. **Cycle Detection**: Use Tarjan's SCC algorithm, break cycles by removing lowest-weight edge
2. **Transitive Reduction**: If A → B and B → C and A → C, remove A → C (redundant)
3. **Threshold Pruning**: Remove edges below weight threshold (e.g., < 0.5)

### Step 3: Topological Sort (Learning Pathway)
Once DAG is cycle-free:
```python
study_order = topological_sort(prerequisite_graph)
# Returns: [foundational concepts] → [intermediate] → [advanced]
```

### Step 4: Expert Validation
**Gold Standard Comparison:**
1. Sample 50 prerequisite pairs from graph
2. Ask 3 CS instructors: "Does A need to be learned before B?" (Yes/No/Unsure)
3. Calculate precision: % of edges experts agree with
4. Calculate coverage: % of expert-identified prerequisites captured by graph

**Target:** Precision > 70%, Coverage > 60%

---

## Visualization & Insights

### Graph Visualization (NetworkX + Graphviz):
```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
for concept_a, concept_b, weight in edges:
    G.add_edge(concept_a, concept_b, weight=weight)

# Hierarchical layout (foundational → advanced)
pos = nx.spring_layout(G, k=2)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=8, arrows=True)
plt.savefig('prerequisite_graph.png', dpi=300)
```

### Key Metrics:
1. **Foundational concepts**: Nodes with in-degree = 0 (no prerequisites)
2. **Advanced concepts**: Nodes with out-degree = 0 (terminal concepts)
3. **Hub concepts**: High out-degree (many concepts depend on them)
4. **Longest path**: Critical path through curriculum (minimum study time)

### Example Insights:
- **Foundational:** `['binary arithmetic', 'logic gates', 'Boolean algebra']`
- **Hubs:** `['TCP', 'database normalization', 'process scheduling']`
- **Advanced:** `['zero-day exploits', 'consensus protocols', 'NUMA architecture']`

---

## Applications

### 1. Personalized Study Pathway
**Input:** Student's current knowledge (self-assessed or tested)
**Output:** Optimal sequence of topics to study next

```python
def generate_study_path(student_knowledge, prerequisite_graph):
    learned = set(student_knowledge)
    available = [node for node in graph.nodes if all(prereq in learned for prereq in graph.predecessors(node))]
    return available  # Topics student can learn next
```

### 2. Curriculum Gap Analysis
Compare exam prerequisites vs. university curriculum:
- Which concepts are tested but never taught?
- Which concepts are taught but never tested?

### 3. Difficulty Calibration
Concepts with many prerequisites are objectively "harder" (more background required).

```python
def concept_difficulty(concept, graph):
    return len(nx.ancestors(graph, concept))  # Number of prerequisites
```

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Concept extraction | 1-2 weeks | 200-300 concepts mapped to questions |
| Dependency inference | 2-3 weeks | Weighted directed graph |
| Graph construction & validation | 1-2 weeks | Prerequisite DAG + expert validation |
| Visualization & insights | 1 week | Graph visualizations, study pathways |
| Thesis write-up | 3-4 weeks | Final document + defense |
| **TOTAL** | **~2.5-3 months** | Complete thesis |

---

## Expected Contributions

### Academic:
1. **Novel method** for prerequisite extraction from unstructured exam data
2. **Validated prerequisite graph** for PhilNITS exam corpus (shareable dataset)
3. **Multi-signal inference framework** (generalizable to other domains)
4. **Empirical evaluation** against expert knowledge

### Practical:
1. **Automated study pathway generator** for exam prep
2. **Curriculum feedback tool** for educators
3. **Knowledge gap identification** for students

---

## Key Risks & Mitigations

### Risk 1: Concept extraction produces too many/too few concepts
**Mitigation:**
- Start with domain keywords (guaranteed quality)
- Manually review YAKE output, set frequency threshold
- Aim for 200-300 concepts (manageable for validation)

### Risk 2: Graph has too many cycles (not a DAG)
**Mitigation:**
- Cycles indicate bidirectional dependencies (A ↔ B both prerequisites)
- Break cycles by removing lowest-confidence edges
- Accept small amount of remaining cycles (real learning isn't always linear)

### Risk 3: Expert validation shows low precision (<60%)
**Mitigation:**
- Tune signal weights (maybe textbook ordering matters more)
- Add more signals (e.g., question difficulty progression)
- Fall back to conservative threshold (keep only high-confidence edges)

---

## Integration with Cognitive Taxonomy

**Synergy:** These two theses complement each other!

- **Cognitive taxonomy** answers: *What type of thinking does this question require?*
- **Prerequisite graph** answers: *What must I learn before attempting this question?*

**Combined application:**
1. Student selects target concept (e.g., "TCP congestion control")
2. System returns prerequisite path: `binary math → logic gates → networking basics → TCP → congestion control`
3. For each step, system shows questions filtered by cognitive type:
   - Start with "Definition Recall" questions (build vocabulary)
   - Then "Concept Application" (apply to scenarios)
   - Finally "System State Evolution" (trace protocol behavior)

**Thesis linkage:**
- Reference prerequisite graph in cognitive taxonomy thesis (Section 6: Future Work)
- Reference cognitive taxonomy in prerequisite graph thesis (Section 5: Applications)
- Defend both as complementary parts of holistic exam analysis system

---

## Tools & Resources

### Required:
- Existing SBERT embeddings (already generated)
- NetworkX (graph construction, topological sort, cycle detection)
- YAKE or KeyBERT (keyphrase extraction)
- Pandas/NumPy (data manipulation)
- Graphviz (visualization)

### Optional:
- Neo4j (graph database for large-scale storage)
- D3.js (interactive web-based graph visualization)
- Reference textbooks for external validation (Tanenbaum, Silberschatz, Kurose-Ross)

---

## Next Steps (Immediate)

1. **Extract domain keywords per question**: Run analysis on existing data (1 hour)
2. **Install YAKE/KeyBERT**: Test keyphrase extraction on 10 sample questions (2 hours)
3. **Build co-occurrence matrix**: Which concepts appear together frequently? (1 day)
4. **Manual concept refinement**: Review top 50 extracted phrases, merge duplicates (2 hours)
5. **Prototype dependency detection**: Implement Signal 1 (co-occurrence + temporal) on small subset (1 day)

---

**Author:** Generated during session 2026-02-08
**Status:** Planning phase
**Next review:** After concept extraction complete
