# Cognitive Taxonomy Classification for PhilNITS Exams

## Research Question

**Can we automatically classify IT certification exam questions by cognitive reasoning type, beyond surface-level topic categorization?**

## Motivation

Current exam analysis tools categorize questions by **topic** (networking, databases, OS) but ignore **how** the question tests knowledge. A question about TCP can test:
- **Definition recall**: "What does TCP stand for?"
- **System behavior**: "What happens when a TCP SYN packet is lost?"
- **Resource optimization**: "Which protocol minimizes latency: TCP or UDP?"
- **Failure analysis**: "Why would TCP perform poorly on satellite links?"

These require fundamentally different cognitive skills. Topic clustering alone doesn't help students understand *what type of thinking* each question demands.

## Objective

Build an automated classifier that maps exam questions to **cognitive reasoning categories**, validated against expert human ratings.

---

## Cognitive Taxonomy for CS/IT Exams

Proposed 7-category taxonomy (inspired by Bloom + CS-specific reasoning):

### 1. **Definition Recall**
- Tests memorization of terms, acronyms, standards
- Example: "Which of the following describes RAID 5?"
- Cognitive load: Low (recognize/remember)

### 2. **Concept Application**
- Apply known concept to new scenario
- Example: "A company needs high availability. Which RAID level is appropriate?"
- Cognitive load: Medium (understand → apply)

### 3. **System State Evolution**
- Trace how system changes over time/steps
- Example: "Process A holds lock X, requests Y. Process B holds Y, requests X. What occurs?"
- Cognitive load: Medium-High (simulate execution)

### 4. **Algorithm/Computation**
- Execute algorithm, calculate result
- Example: "Binary tree has 7 nodes. What is the minimum height?"
- Cognitive load: Medium (procedural execution)

### 5. **Resource Optimization**
- Compare tradeoffs, choose optimal solution
- Example: "Which data structure minimizes lookup time: array, linked list, hash table?"
- Cognitive load: High (evaluate constraints)

### 6. **Failure/Edge Case Analysis**
- Identify what breaks, why, or boundary conditions
- Example: "What causes TCP congestion collapse?"
- Cognitive load: High (reason about failure modes)

### 7. **Design/Architecture**
- Select appropriate system design or component
- Example: "Which architecture pattern suits microservices: layered, event-driven, or service mesh?"
- Cognitive load: Very High (synthesize requirements)

---

## Methodology

### Phase 1: Manual Labeling (Ground Truth)
**Timeline:** 2-3 weeks
**Deliverable:** Labeled dataset of 300-500 questions

1. **Select stratified sample**: 300 questions across all domains (OS, networking, DB, algorithms, security)
2. **Recruit 3 labelers**: You + 2 CS faculty/PhD students with teaching experience
3. **Labeling protocol**:
   - Each labeler independently assigns 1 cognitive category per question
   - Provide category definitions + 5 examples each
   - Use majority vote for final label
   - Calculate Cohen's kappa (inter-rater agreement)
4. **Quality control**: Discard questions with 0% agreement (ambiguous)

**Expected output:** 250-300 consistently labeled questions, κ > 0.6 (substantial agreement)

---

### Phase 2: Feature Engineering
**Timeline:** 1 week
**Deliverable:** Feature vectors for classification

Combine multiple feature types:

#### A. Semantic Features (from SBERT embeddings)
- 384-dim question embeddings (already have these)

#### B. Syntactic Features
- Question length (tokens)
- Presence of interrogatives ("which", "what", "why", "how")
- Presence of comparatives ("better", "optimal", "minimum")
- Presence of temporal markers ("after", "then", "next")
- Presence of failure terms ("incorrect", "fails", "error")

#### C. Structural Features
- Answer option variance (how different are the choices?)
- Presence of code/pseudocode (regex detection)
- Presence of numeric values (computation signal)

#### D. Domain Features
- Which domain keywords appear (from DOMAIN_KEYWORDS)
- Binary flags for domain membership

**Implementation:**
```python
def extract_features(question_text, embeddings, domain_keywords):
    features = {
        'embedding': embeddings,  # 384-dim SBERT
        'length': len(question_text.split()),
        'has_comparative': bool(re.search(r'\b(better|optimal|best|worst|minimum|maximum)\b', question_text.lower())),
        'has_temporal': bool(re.search(r'\b(after|then|next|before|when)\b', question_text.lower())),
        'has_failure': bool(re.search(r'\b(fail|error|incorrect|invalid|wrong)\b', question_text.lower())),
        'has_numeric': bool(re.search(r'\d+', question_text)),
        'domain': identify_domain(question_text, domain_keywords),
    }
    return features
```

---

### Phase 3: Model Training & Evaluation
**Timeline:** 1 week
**Deliverable:** Trained classifier with performance metrics

#### Models to Try (in order):
1. **Logistic Regression** (baseline: embedding features only)
2. **Random Forest** (all features)
3. **XGBoost** (all features)
4. **Fine-tuned SBERT** (classification head on top of embeddings)

#### Training Strategy:
- 80/20 train-test split (stratified by category)
- 5-fold cross-validation on training set
- Hyperparameter tuning via grid search

#### Evaluation Metrics:
- **Accuracy** (overall)
- **Per-class F1-score** (some categories are rare)
- **Confusion matrix** (which categories get mixed up?)
- **Feature importance** (which features matter most?)

**Expected performance:** 65-75% accuracy (7-way classification is hard, beating random baseline of 14% is success)

---

### Phase 4: Error Analysis & Refinement
**Timeline:** 1 week
**Deliverable:** Insights + improved model

1. **Inspect misclassifications**:
   - Which questions are consistently misclassified?
   - Are certain category pairs confused (e.g., Definition vs. Application)?
2. **Iterative refinement**:
   - Add more labeled examples for confused categories
   - Add new features based on error patterns
   - Consider hierarchical classification (group similar categories)

---

### Phase 5: Application to Full Corpus
**Timeline:** 1 day
**Deliverable:** All 1,472 questions classified by cognitive type

- Run trained classifier on all questions
- Generate statistics:
  - Distribution of cognitive types per domain
  - Temporal trends (are modern exams more "Design" heavy?)
  - Cluster analysis (do cognitive types group separately from topics?)

---

### Phase 6: Validation & Thesis Write-Up
**Timeline:** 3-4 weeks
**Deliverable:** Thesis document, defense slides

#### Validation:
- **Quantitative**: Report inter-rater agreement (κ), classifier accuracy, F1-scores
- **Qualitative**: Expert interviews (do CS instructors find categories useful?)
- **Comparative**: Compare against topic-only clustering (show cognitive taxonomy adds value)

#### Thesis Structure:
1. **Introduction**: Problem statement, motivation, research question
2. **Related Work**: Bloom's taxonomy, automated question classification, educational data mining
3. **Methodology**: Taxonomy design, labeling protocol, feature engineering, model training
4. **Results**: Performance metrics, confusion matrices, feature importance, full corpus analysis
5. **Discussion**: What worked, what didn't, implications for exam design
6. **Limitations**: See LIMITATIONS.md
7. **Future Work**: Multi-label classification, prerequisite integration, personalized study paths
8. **Conclusion**: Contributions, impact

---

## Expected Contributions

### Academic:
1. **Novel taxonomy** for CS/IT exam cognitive classification (7 categories)
2. **Labeled dataset** of 300+ PhilNITS questions (shareable for future research)
3. **Automated classifier** with feature ablation study
4. **Empirical insights** into PhilNITS exam design patterns

### Practical:
1. **Better study tools**: Students can filter questions by reasoning type
2. **Curriculum feedback**: Identify under-tested cognitive skills
3. **Exam design guidance**: Balance cognitive load across categories

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Manual labeling | 2-3 weeks | 300 labeled questions, κ > 0.6 |
| Feature engineering | 1 week | Feature extraction pipeline |
| Model training | 1 week | Trained classifier, metrics |
| Error analysis | 1 week | Refined model |
| Full corpus classification | 1 day | 1,472 questions classified |
| Thesis write-up | 3-4 weeks | Final document + defense |
| **TOTAL** | **~2.5-3 months** | Complete thesis |

---

## Key Risks & Mitigations

### Risk 1: Low inter-rater agreement (κ < 0.5)
**Mitigation:**
- Refine category definitions with examples
- Add labeling training session with discussion
- Consider collapsing similar categories (7 → 5)

### Risk 2: Classifier performs poorly (<50% accuracy)
**Mitigation:**
- Simplify to 3-4 categories (easier problem)
- Add more labeled data (300 → 500)
- Try ensemble methods (stacking multiple models)

### Risk 3: Not enough time before thesis deadline
**Mitigation:**
- Prioritize Phases 1-3 (labeling, training, evaluation)
- Phases 4-5 are optional (nice-to-have, not required)
- Minimum viable thesis: 200 labeled questions + baseline classifier

---

## Tools & Resources

### Required:
- Existing SBERT embeddings (already generated)
- Sklearn (classification models)
- Pandas/NumPy (data manipulation)
- Matplotlib/Seaborn (visualization)

### Optional:
- Label Studio (web UI for collaborative labeling)
- Weights & Biases (experiment tracking)
- Hugging Face Transformers (fine-tuning SBERT)

---

## Next Steps (Immediate)

1. **Define labeling protocol**: Write category definitions + examples (2-3 hours)
2. **Recruit labelers**: Find 2 CS faculty/PhD students willing to label 300 questions (1 day)
3. **Select sample**: Stratified random sample of 300 questions (30 min)
4. **Start labeling**: Each labeler does 100 questions/week (3 weeks total)

---

**Author:** Generated during session 2026-02-08
**Status:** Planning phase
**Next review:** After Phase 1 (labeling complete)
