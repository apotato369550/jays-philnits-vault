#!/usr/bin/env python3
"""
Analyze exam patterns via clustering, keyword extraction, and domain mapping.

Processes vectorized question embeddings to extract patterns, generate insights,
and produce two outputs:
1. exam_analysis_agent.json: Machine-readable analysis for LLM processing
2. exam_study_guide.md: Human-readable study guide with recommendations
"""

import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.feature_extraction.text import TfidfVectorizer


class ExamPatternAnalyzer:
    """Clusters exam questions and extracts domain/keyword/difficulty patterns."""

    # Industry vocabulary definitions
    INDUSTRY_VOCAB_FOUNDATIONAL = {
        "RAID": "Redundant Array of Independent Disks",
        "throughput": "Data transfer rate (bits/sec or packets/sec)",
        "latency": "Delay in system response time",
        "failover": "Automatic system switchover on failure",
        "cache coherency": "Multi-core memory consistency protocol",
        "context switching": "CPU switching between process/thread execution",
        "paging": "Virtual memory management via disk",
        "semaphore": "Synchronization primitive for resource access",
        "mutex": "Mutual exclusion lock for critical sections",
        "deadlock": "Circular wait condition preventing progress",
        "TCP/IP": "Internet communication protocol stack",
        "UDP": "User Datagram Protocol (connectionless)",
        "routing": "Path selection for network packets",
        "packet loss": "Dropped network packets in transmission",
        "QoS": "Quality of Service network management",
        "normalization": "Database schema design principle",
        "indexing": "Database acceleration structure",
        "transaction": "ACID database operation unit",
        "encryption": "Data transformation for confidentiality",
        "hash function": "One-way cryptographic mapping",
    }

    INDUSTRY_VOCAB_ADVANCED = {
        "side-channel attack": "Exploit physical implementation details",
        "privilege escalation": "Unauthorized elevation of access rights",
        "attestation": "Cryptographic hardware/software verification",
        "TPM": "Trusted Platform Module hardware security",
        "zero-day": "Unpatched security vulnerability",
        "sharding": "Database horizontal partitioning",
        "replication": "Data copying across systems",
        "consensus protocol": "Distributed agreement mechanism (Raft, Paxos)",
        "CAP theorem": "Consistency, Availability, Partition tolerance tradeoff",
        "ACID": "Atomicity, Consistency, Isolation, Durability",
        "BASE": "Basically Available, Soft state, Eventually consistent",
        "branch prediction": "CPU optimization guessing next instruction",
        "cache coherency": "Multi-core shared memory consistency",
        "NUMA": "Non-Uniform Memory Access architecture",
        "process affinity": "CPU binding for process execution",
        "containerization": "Lightweight virtualization (Docker, etc)",
        "orchestration": "Automated container management (Kubernetes)",
        "infrastructure-as-code": "Version-controlled infrastructure definition",
        "BGP": "Border Gateway Protocol routing",
        "OSPF": "Open Shortest Path First routing",
        "VLAN": "Virtual Local Area Network",
        "jitter": "Variation in packet arrival timing",
    }

    # Domain keyword patterns
    DOMAIN_KEYWORDS = {
        "OS_KERNEL": {
            "process",
            "thread",
            "context switching",
            "scheduling",
            "preemption",
            "interrupt",
            "system call",
            "kernel mode",
            "virtual memory",
            "paging",
            "segmentation",
            "deadlock",
            "synchronization",
            "mutex",
            "semaphore",
            "race condition",
            "critical section",
        },
        "NETWORKING": {
            "TCP",
            "UDP",
            "IP",
            "routing",
            "protocol",
            "packet",
            "socket",
            "OSI model",
            "layer",
            "handshake",
            "congestion control",
            "flow control",
            "firewall",
            "NAT",
            "VLAN",
            "gateway",
            "latency",
            "throughput",
            "jitter",
        },
        "SECURITY": {
            "encryption",
            "decryption",
            "cryptography",
            "hash",
            "authentication",
            "authorization",
            "SQL injection",
            "XSS",
            "CSRF",
            "vulnerability",
            "exploit",
            "privilege",
            "zero-day",
            "attestation",
            "TPM",
            "side-channel",
        },
        "DATABASES": {
            "database",
            "SQL",
            "normalization",
            "transaction",
            "ACID",
            "indexing",
            "query",
            "schema",
            "foreign key",
            "constraint",
            "relational",
            "replication",
            "sharding",
            "consistency",
            "BASE",
        },
        "STORAGE": {
            "disk",
            "storage",
            "RAID",
            "SSD",
            "HDD",
            "file system",
            "inode",
            "block",
            "cache",
            "buffer",
            "failover",
            "mirroring",
            "striping",
            "redundancy",
        },
        "ALGORITHMS": {
            "algorithm",
            "complexity",
            "sorting",
            "searching",
            "graph",
            "tree",
            "hash table",
            "dynamic programming",
            "recursion",
            "Big O",
            "optimization",
        },
    }

    def __init__(
        self,
        vector_file: str = "data/vectorized/vectors.npz",
        metadata_file: str = "data/vectorized/metadata.json",
        json_dir: str = "data/refined_json",
        output_dir: str = "analysis",
        log_dir: str = "logs",
        random_seed: int = 42,
    ):
        """
        Initialize the analyzer.

        Args:
            vector_file: Path to vectors.npz from vectorization
            metadata_file: Path to metadata.json from vectorization
            json_dir: Directory containing original refined JSON files
            output_dir: Directory for analysis outputs
            log_dir: Directory for logs
            random_seed: Random seed for deterministic clustering
        """
        self.vector_file = Path(vector_file)
        self.metadata_file = Path(metadata_file)
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.random_seed = random_seed

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Data containers
        self.vectors = None
        self.metadata = []
        self.questions_by_exam = defaultdict(list)
        self.clusters = None
        self.silhouette_scores = None
        self.n_clusters = None

        # Analysis results
        self.cluster_keywords = {}
        self.cluster_domains = {}
        self.cluster_question_types = {}
        self.cluster_difficulties = {}
        self.industry_terms = {"FOUNDATIONAL": {}, "ADVANCED": {}}
        self.domain_counts = defaultdict(int)

        self.logger.info("ExamPatternAnalyzer initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging to file and console."""
        logger = logging.getLogger("ExamPatternAnalyzer")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # File handler
        log_file = (
            self.log_dir
            / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def load_data(self) -> bool:
        """
        Load vectors, metadata, and original JSON files.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Loading data...")

        # Load vectors
        if not self.vector_file.exists():
            self.logger.error(f"Vector file not found: {self.vector_file}")
            return False

        try:
            data = np.load(self.vector_file)
            self.vectors = data["vectors"]
            self.logger.info(f"Loaded vectors: shape {self.vectors.shape}")
        except Exception as e:
            self.logger.error(f"Error loading vectors: {e}")
            return False

        # Load metadata
        if not self.metadata_file.exists():
            self.logger.error(f"Metadata file not found: {self.metadata_file}")
            return False

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.logger.info(f"Loaded metadata: {len(self.metadata)} items")
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            return False

        # Load original JSON files for text analysis
        self.logger.info("Loading original JSON files...")
        json_files = sorted(self.json_dir.glob("*.json"))
        total_loaded = 0

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                exam_name = data.get("metadata", {}).get("exam_name", json_file.stem)
                questions = data.get("questions", [])
                self.questions_by_exam[exam_name] = questions
                total_loaded += len(questions)
            except Exception as e:
                self.logger.warning(f"Error loading {json_file}: {e}")

        self.logger.info(f"Loaded questions from {len(self.questions_by_exam)} exams ({total_loaded} total)")
        return True

    def cluster_questions(self, k_range: Tuple[int, int] = (6, 8)) -> bool:
        """
        Cluster questions using K-means with optimal K selection via silhouette analysis.

        Args:
            k_range: Tuple of (min_k, max_k) to evaluate

        Returns:
            True if successful
        """
        self.logger.info("Clustering questions...")

        if self.vectors is None or len(self.vectors) == 0:
            self.logger.error("No vectors available for clustering")
            return False

        # Evaluate different K values
        silhouette_scores_by_k = {}
        best_k = k_range[0]
        best_score = -1

        for k in range(k_range[0], k_range[1] + 1):
            self.logger.info(f"Evaluating K={k}...")
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_seed,
                n_init=10,
                verbose=0,
            )
            clusters = kmeans.fit_predict(self.vectors)
            score = silhouette_score(self.vectors, clusters)
            silhouette_scores_by_k[k] = score
            self.logger.info(f"  K={k}: silhouette_score={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

        self.logger.info(f"Optimal K={best_k} with silhouette_score={best_score:.4f}")

        # Perform final clustering with optimal K
        kmeans = KMeans(
            n_clusters=best_k,
            random_state=self.random_seed,
            n_init=10,
            verbose=0,
        )
        self.clusters = kmeans.fit_predict(self.vectors)
        self.n_clusters = best_k

        # Calculate per-sample silhouette scores
        self.silhouette_scores = silhouette_samples(self.vectors, self.clusters)

        self.logger.info(f"Clustering complete: {best_k} clusters, {len(self.clusters)} questions")
        return True

    def extract_keywords(self, top_n: int = 10) -> None:
        """
        Extract distinctive technical keywords per cluster using hybrid approach:
        1. Domain keywords (from DOMAIN_KEYWORDS) ranked by frequency in cluster
        2. Rare n-grams (2-3 words) that appear frequently in cluster but rarely in corpus

        Args:
            top_n: Number of top keywords to extract per cluster
        """
        self.logger.info(f"Extracting keywords (top {top_n} per cluster)...")

        # Build domain keyword lookup (all technical terms across domains)
        all_domain_keywords = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                all_domain_keywords[kw.lower()] = domain

        # Build corpus-wide n-gram frequency map
        all_texts = [meta.get("question_text", "") for meta in self.metadata]
        corpus_ngram_counts = Counter()

        # Extract all n-grams from corpus for rarity filtering
        from sklearn.feature_extraction.text import CountVectorizer
        ngram_vectorizer = CountVectorizer(
            ngram_range=(2, 3),
            lowercase=True,
            max_features=5000,
        )
        try:
            ngram_vectorizer.fit(all_texts)
            corpus_ngrams = ngram_vectorizer.get_feature_names_out()

            # Count n-gram occurrences across corpus
            for text in all_texts:
                text_lower = text.lower()
                for ngram in corpus_ngrams:
                    if ngram in text_lower:
                        corpus_ngram_counts[ngram] += 1
        except:
            corpus_ngrams = []

        total_questions = len(all_texts)

        for cluster_id in range(self.n_clusters):
            # Get question indices for this cluster
            cluster_indices = np.where(self.clusters == cluster_id)[0]

            # Get texts for this cluster
            cluster_texts = [
                self.metadata[idx].get("question_text", "")
                for idx in cluster_indices
            ]

            if not cluster_texts or all(not t for t in cluster_texts):
                self.logger.warning(f"Cluster {cluster_id}: no texts available")
                self.cluster_keywords[cluster_id] = []
                continue

            cluster_size = len(cluster_texts)

            # PHASE 1: Count domain keywords in this cluster
            domain_keyword_counts = Counter()
            for text in cluster_texts:
                text_lower = text.lower()
                for keyword, domain in all_domain_keywords.items():
                    # Whole word matching
                    import re
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, text_lower):
                        domain_keyword_counts[keyword] += 1

            # PHASE 2: Extract rare n-grams (frequent in cluster, rare in corpus)
            rare_ngrams = Counter()
            for text in cluster_texts:
                text_lower = text.lower()
                for ngram in corpus_ngrams:
                    if ngram in text_lower:
                        rare_ngrams[ngram] += 1

            # Filter: must appear in >=3 questions in cluster AND <10% of corpus
            distinctive_ngrams = {}
            for ngram, cluster_freq in rare_ngrams.items():
                corpus_freq = corpus_ngram_counts[ngram]
                corpus_prevalence = corpus_freq / total_questions

                # Distinctive if: appears 3+ times in cluster, <10% corpus prevalence
                if cluster_freq >= 3 and corpus_prevalence < 0.10:
                    # Check it's not already covered by domain keywords
                    is_domain_covered = any(
                        kw in ngram for kw in domain_keyword_counts.keys()
                    )
                    if not is_domain_covered:
                        distinctive_ngrams[ngram] = cluster_freq

            # PHASE 3: Combine and rank
            keywords = []

            # Add domain keywords (prioritized)
            for keyword, freq in domain_keyword_counts.most_common():
                keywords.append({
                    "word": keyword,
                    "frequency": freq,
                    "prevalence": round(freq / cluster_size, 3),
                    "type": "domain_keyword",
                })

            # Add distinctive n-grams
            for ngram, freq in sorted(distinctive_ngrams.items(), key=lambda x: x[1], reverse=True):
                keywords.append({
                    "word": ngram,
                    "frequency": freq,
                    "prevalence": round(freq / cluster_size, 3),
                    "type": "distinctive_ngram",
                })

            # Limit to top_n
            self.cluster_keywords[cluster_id] = keywords[:top_n]

            domain_count = sum(1 for kw in self.cluster_keywords[cluster_id] if kw["type"] == "domain_keyword")
            ngram_count = sum(1 for kw in self.cluster_keywords[cluster_id] if kw["type"] == "distinctive_ngram")

            self.logger.info(
                f"Cluster {cluster_id}: extracted {len(self.cluster_keywords[cluster_id])} keywords "
                f"({domain_count} domain, {ngram_count} distinctive n-grams)"
            )

    def classify_question_types(self) -> None:
        """Classify questions by type using regex patterns."""
        self.logger.info("Classifying question types...")

        # Initialize type counters
        for cluster_id in range(self.n_clusters):
            self.cluster_question_types[cluster_id] = {
                "DEFINITION_RECALL": 0,
                "SCENARIO_APPLICATION": 0,
                "COMPUTATION": 0,
                "DIAGRAM_ANALYSIS": 0,
                "GENERAL": 0,
            }

        # Patterns
        patterns = {
            "DIAGRAM_ANALYSIS": re.compile(
                r"figure|diagram|shown|illustrated|sketch", re.IGNORECASE
            ),
            "SCENARIO_APPLICATION": re.compile(
                r"when.*is applied|scenario|situation|consider|suppose", re.IGNORECASE
            ),
            "COMPUTATION": re.compile(
                r"calculate|compute|binary|bit|operation|algorithm|\+|-|\*|/|%|&|\||>>|<<",
                re.IGNORECASE,
            ),
            "DEFINITION_RECALL": re.compile(
                r"which of the following|what is|define|meaning|refers to", re.IGNORECASE
            ),
        }

        for idx, metadata_item in enumerate(self.metadata):
            cluster_id = self.clusters[idx]
            text = metadata_item.get("question_text", "").lower()

            # Check patterns in order of specificity
            question_type = "GENERAL"
            for qtype, pattern in patterns.items():
                if pattern.search(text):
                    question_type = qtype
                    break

            self.cluster_question_types[cluster_id][question_type] += 1

        self.logger.info("Question type classification complete")

    def estimate_difficulty(self) -> None:
        """
        Estimate question difficulty based on heuristics.

        Heuristics: question length, option variance, keyword specificity
        """
        self.logger.info("Estimating question difficulty...")

        # Initialize difficulty counters
        for cluster_id in range(self.n_clusters):
            self.cluster_difficulties[cluster_id] = {
                "LOW": 0,
                "MEDIUM": 0,
                "HIGH": 0,
            }

        for idx, metadata_item in enumerate(self.metadata):
            cluster_id = self.clusters[idx]
            text = metadata_item.get("question_text", "")

            # Heuristics
            text_length = len(text.split())
            option_count = len(
                [k for k in metadata_item.keys() if k.startswith("option_")]
            )

            # Simple scoring: longer = harder, more options = harder
            score = min(100, (text_length * 2) + (option_count * 10))

            if score < 33:
                difficulty = "LOW"
            elif score < 66:
                difficulty = "MEDIUM"
            else:
                difficulty = "HIGH"

            self.cluster_difficulties[cluster_id][difficulty] += 1

        self.logger.info("Difficulty estimation complete")

    def extract_industry_vocabulary(self) -> None:
        """
        Scan all question texts for industry-specific vocabulary.

        Categorizes as FOUNDATIONAL or ADVANCED.
        Uses case-sensitive matching for acronyms (all-caps terms).
        """
        self.logger.info("Extracting industry vocabulary...")

        # Build lookup from definitions
        all_vocab = {}
        all_vocab.update(self.INDUSTRY_VOCAB_FOUNDATIONAL)
        all_vocab.update(self.INDUSTRY_VOCAB_ADVANCED)

        # Track occurrences
        term_clusters = defaultdict(set)
        term_counts = Counter()

        for idx, metadata_item in enumerate(self.metadata):
            cluster_id = self.clusters[idx]
            text = metadata_item.get("question_text", "")

            for term in all_vocab.keys():
                # Case-sensitive for acronyms (all uppercase), case-insensitive otherwise
                if term.isupper() and len(term) <= 6:
                    # Acronym: exact case match with word boundaries
                    import re
                    pattern = r'\b' + re.escape(term) + r'\b'
                    if re.search(pattern, text):
                        term_counts[term] += 1
                        term_clusters[term].add(cluster_id)
                else:
                    # Regular term: case-insensitive
                    if term.lower() in text.lower():
                        term_counts[term] += 1
                        term_clusters[term].add(cluster_id)

        # Organize by category
        for term, definition in self.INDUSTRY_VOCAB_FOUNDATIONAL.items():
            if term_counts[term] > 0:
                self.industry_terms["FOUNDATIONAL"][term] = {
                    "definition": definition,
                    "occurrences": term_counts[term],
                    "clusters": sorted(list(term_clusters[term])),
                }

        for term, definition in self.INDUSTRY_VOCAB_ADVANCED.items():
            if term_counts[term] > 0:
                self.industry_terms["ADVANCED"][term] = {
                    "definition": definition,
                    "occurrences": term_counts[term],
                    "clusters": sorted(list(term_clusters[term])),
                }

        self.logger.info(
            f"Found {len(self.industry_terms['FOUNDATIONAL'])} foundational and "
            f"{len(self.industry_terms['ADVANCED'])} advanced industry terms"
        )

    def map_domains(self) -> None:
        """
        Map clusters to domains via keyword matching and inspection.

        Assigns primary domain to each cluster.
        """
        self.logger.info("Mapping domains to clusters...")

        self.cluster_domains = {}

        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.clusters == cluster_id)[0]
            cluster_texts = [
                self.metadata[idx].get("question_text", "").lower()
                for idx in cluster_indices
            ]

            # Count keyword matches by domain
            domain_scores = defaultdict(int)

            for domain, keywords in self.DOMAIN_KEYWORDS.items():
                for keyword in keywords:
                    count = sum(
                        1
                        for text in cluster_texts
                        if keyword.lower() in text
                    )
                    domain_scores[domain] += count

            # Assign primary domain
            if domain_scores:
                primary_domain = max(domain_scores, key=domain_scores.get)
            else:
                primary_domain = "MISC"

            self.cluster_domains[cluster_id] = primary_domain
            self.domain_counts[primary_domain] += len(cluster_indices)

            self.logger.info(
                f"Cluster {cluster_id}: primary domain = {primary_domain} "
                f"(size={len(cluster_indices)})"
            )

    def generate_agent_analysis(self) -> Dict:
        """
        Generate machine-readable analysis for LLM processing.

        Returns:
            Dictionary with structured analysis data
        """
        self.logger.info("Generating agent analysis...")

        analysis = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "total_questions": len(self.metadata),
                "exams_analyzed": len(self.questions_by_exam),
                "clusters": self.n_clusters,
                "model": "all-MiniLM-L6-v2",
            },
            "clusters": [],
            "domain_summary": {},
            "industry_vocabulary": self.industry_terms,
            "insights": [],
        }

        # Build cluster data
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.clusters == cluster_id)[0]
            cluster_silhouette_score = float(self.silhouette_scores[cluster_indices].mean())

            cluster_data = {
                "id": cluster_id,
                "size": len(cluster_indices),
                "primary_domain": self.cluster_domains.get(cluster_id, "MISC"),
                "silhouette_score": round(cluster_silhouette_score, 4),
                "top_keywords": self.cluster_keywords.get(cluster_id, []),
                "question_types": self.cluster_question_types.get(cluster_id, {}),
                "difficulty_distribution": self.cluster_difficulties.get(cluster_id, {}),
                "sample_questions": self._get_sample_questions(cluster_id, sample_size=3),
            }
            analysis["clusters"].append(cluster_data)

        # Build domain summary
        domain_cluster_map = defaultdict(list)
        for cluster_id, domain in self.cluster_domains.items():
            domain_cluster_map[domain].append(cluster_id)

        for domain, clusters in sorted(domain_cluster_map.items()):
            count = self.domain_counts[domain]
            analysis["domain_summary"][domain] = {
                "count": count,
                "clusters": clusters,
            }

        # Add insights
        analysis["insights"] = self._generate_insights()

        return analysis

    def _get_sample_questions(self, cluster_id: int, sample_size: int = 3) -> List[Dict]:
        """Get sample questions from a cluster."""
        cluster_indices = np.where(self.clusters == cluster_id)[0]
        sample_indices = cluster_indices[: min(sample_size, len(cluster_indices))]

        samples = []
        for idx in sample_indices:
            meta = self.metadata[idx]
            q_num = meta.get("q_num")
            # Convert numpy types to Python native types for JSON serialization
            if hasattr(q_num, 'item'):
                q_num = q_num.item()
            samples.append(
                {
                    "q_num": q_num,
                    "exam": meta.get("exam"),
                    "text": meta.get("question_text", "")[:100] + "...",
                }
            )
        return samples

    def _generate_insights(self) -> List[str]:
        """Generate high-level insights from analysis."""
        insights = []

        # Domain distribution
        total = sum(self.domain_counts.values())
        for domain, count in sorted(
            self.domain_counts.items(), key=lambda x: x[1], reverse=True
        ):
            pct = (count / total * 100) if total > 0 else 0
            insights.append(f"{domain} concepts dominate (~{pct:.1f}% of questions)")

        # Clustering quality
        avg_silhouette = np.mean(self.silhouette_scores) if self.silhouette_scores is not None else 0
        insights.append(
            f"Average silhouette score: {avg_silhouette:.3f} "
            f"(clustering quality: {'good' if avg_silhouette > 0.5 else 'moderate' if avg_silhouette > 0.3 else 'weak'})"
        )

        # Question type patterns
        total_types = Counter()
        for type_counts in self.cluster_question_types.values():
            total_types.update(type_counts)

        if total_types:
            top_type = total_types.most_common(1)[0][0]
            top_count = total_types[top_type]
            pct = (top_count / sum(total_types.values()) * 100)
            insights.append(
                f"{top_type} questions are most common (~{pct:.0f}% of corpus)"
            )

        # Industry vocabulary
        total_industry = (
            len(self.industry_terms["FOUNDATIONAL"])
            + len(self.industry_terms["ADVANCED"])
        )
        insights.append(
            f"Found {total_industry} industry-specific terms across exams"
        )

        return insights

    def save_agent_analysis(self, data: Dict) -> None:
        """Save agent analysis to JSON."""
        output_file = self.output_dir / "exam_analysis_agent.json"

        # Custom encoder to handle NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        self.logger.info(f"Saved agent analysis to {output_file}")

    def generate_user_guide(self) -> str:
        """
        Generate human-readable markdown study guide.

        Returns:
            Markdown string
        """
        self.logger.info("Generating user study guide...")

        md = f"""# Philnits Exam Study Guide
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Quick Stats
- **Total Questions Analyzed**: {len(self.metadata)}
- **Exams Covered**: {len(self.questions_by_exam)}
- **Main Topics**: {self.n_clusters} clusters identified
- **Difficulty Range**: Low to High

## Topics to Study (By Frequency)

"""

        # Sort domains by question count
        sorted_domains = sorted(
            self.domain_counts.items(), key=lambda x: x[1], reverse=True
        )

        for rank, (domain, count) in enumerate(sorted_domains, 1):
            pct = (count / len(self.metadata)) * 100
            md += self._generate_domain_section(domain, count, pct, rank)

        md += self._generate_industry_vocab_section()
        md += self._generate_study_plan_section()
        md += self._generate_patterns_section()

        return md

    def _generate_domain_section(
        self, domain: str, count: int, pct: float, rank: int
    ) -> str:
        """Generate markdown section for a domain."""
        # Get cluster IDs for this domain
        cluster_ids = [
            cid for cid, dom in self.cluster_domains.items() if dom == domain
        ]

        # Determine difficulty and keywords
        all_difficulties = Counter()
        for cid in cluster_ids:
            all_difficulties.update(self.cluster_difficulties.get(cid, {}))

        top_difficulty = all_difficulties.most_common(1)[0][0] if all_difficulties else "MEDIUM"

        # Get unique keywords across clusters
        all_keywords = set()
        for cid in cluster_ids:
            for kw in self.cluster_keywords.get(cid, [])[:5]:
                all_keywords.add(kw["word"])

        keywords_str = ", ".join(sorted(all_keywords)[:10]) if all_keywords else "N/A"

        # Get industry terms in this domain
        industry_terms_in_domain = []
        for category in ["FOUNDATIONAL", "ADVANCED"]:
            for term, data in self.industry_terms[category].items():
                if any(cid in data["clusters"] for cid in cluster_ids):
                    industry_terms_in_domain.append((category, term, data["occurrences"]))

        industry_section = ""
        if industry_terms_in_domain:
            industry_terms_in_domain.sort(key=lambda x: x[2], reverse=True)
            industry_list = ", ".join(
                [f"**[{cat}]** {term}" for cat, term, _ in industry_terms_in_domain[:5]]
            )
            industry_section = f"\n  - **[INDUSTRY]** {industry_list}"

        md = f"""### {rank}. {domain.replace('_', ' & ').title()} ({count} questions, {pct:.1f}%)
- **Focus**: {keywords_str}
- **Difficulty**: {top_difficulty.title()}
- **Key Terms**: {keywords_str}{industry_section}

"""
        return md

    def _generate_industry_vocab_section(self) -> str:
        """Generate industry vocabulary section."""
        md = """## High-Priority Industry Vocabulary

These terms appear in exams but aren't always covered in college:

### Foundational (MUST KNOW)
"""
        foundational = self.industry_terms["FOUNDATIONAL"]
        sorted_foundational = sorted(
            foundational.items(), key=lambda x: x[1]["occurrences"], reverse=True
        )[:10]

        for term, data in sorted_foundational:
            md += f"- **{term}**: {data['definition']} ({data['occurrences']} questions)\n"

        md += """
### Advanced (SHOULD KNOW)
"""
        advanced = self.industry_terms["ADVANCED"]
        sorted_advanced = sorted(
            advanced.items(), key=lambda x: x[1]["occurrences"], reverse=True
        )[:10]

        for term, data in sorted_advanced:
            md += f"- **{term}**: {data['definition']} ({data['occurrences']} questions)\n"

        md += "\n"
        return md

    def _generate_study_plan_section(self) -> str:
        """Generate suggested study plan."""
        # Categorize by difficulty
        easy_domains = [
            d for d, c in self.domain_counts.items()
            if c < len(self.metadata) * 0.1
        ]
        medium_domains = [
            d for d, c in self.domain_counts.items()
            if len(self.metadata) * 0.1 <= c < len(self.metadata) * 0.2
        ]
        hard_domains = [
            d for d, c in self.domain_counts.items()
            if c >= len(self.metadata) * 0.2
        ]

        md = """## Study Plan by Difficulty

**Start Here (Low Difficulty, Foundation)**:
1. Basic concepts in less frequently tested domains
2. Foundational industry vocabulary
3. Common definition/recall question patterns

**Then (Medium Difficulty)**:
4. Core concepts in mid-frequency domains
5. Scenario-based and application problems
6. Medium-complexity industry terms

**Master (High Difficulty, Advanced)**:
7. Highly complex domain concepts
8. Advanced computation and diagram analysis
9. Edge cases and advanced industry terminology

## Question Distribution by Type
"""
        # Count overall question types
        total_types = Counter()
        for type_counts in self.cluster_question_types.values():
            total_types.update(type_counts)

        total_questions = sum(total_types.values())
        for qtype, count in total_types.most_common():
            pct = (count / total_questions * 100) if total_questions > 0 else 0
            md += f"- **{qtype}**: {pct:.0f}% ({count} questions)\n"

        md += "\n"
        return md

    def _generate_patterns_section(self) -> str:
        """Generate exam pattern observations."""
        md = """## Exam Pattern Observations

- **Clustering Quality**: Questions naturally group by semantic similarity, not by exam year
- **Cross-Exam Consistency**: Similar topics appear across multiple exams and years
- **Industry Vocabulary**: Modern exams incorporate increasingly technical industry terms
- **Question Complexity**: Increases with domain specialization (specialized topics > foundational)
- **Type Distribution**: Mix of definition recall, scenario application, and computation questions

## Notes on Using This Guide

1. Start with **Foundational Industry Vocabulary** - these are frequently appearing terms that aren't always taught formally
2. Focus on domains with highest question counts first (80/20 principle)
3. Practice problem types in each domain - exams mix recall, application, and computation
4. Review difficult questions from past exams in each domain
5. Cross-domain knowledge is important - concepts interconnect across OS, networking, and security

---
*Analysis generated by ExamPatternAnalyzer*
"""
        return md

    def save_user_guide(self, markdown: str) -> None:
        """Save user guide to markdown file."""
        output_file = self.output_dir / "exam_study_guide.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)
        self.logger.info(f"Saved user guide to {output_file}")

    def run(self) -> None:
        """Execute the full analysis pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("Starting exam pattern analysis pipeline")
        self.logger.info("=" * 80)

        # Load data
        if not self.load_data():
            self.logger.error("Failed to load data. Exiting.")
            return

        # Cluster
        if not self.cluster_questions():
            self.logger.error("Failed to cluster questions. Exiting.")
            return

        # Extract patterns
        self.extract_keywords()
        self.classify_question_types()
        self.estimate_difficulty()
        self.extract_industry_vocabulary()
        self.map_domains()

        # Generate outputs
        agent_analysis = self.generate_agent_analysis()
        self.save_agent_analysis(agent_analysis)

        user_guide = self.generate_user_guide()
        self.save_user_guide(user_guide)

        # Summary
        self.logger.info("=" * 80)
        self.logger.info("Analysis pipeline complete!")
        self.logger.info(
            f"Analyzed {len(self.metadata)} questions "
            f"from {len(self.questions_by_exam)} exams"
        )
        self.logger.info(f"Generated {self.n_clusters} clusters")
        self.logger.info(f"Identified {len(self.domain_counts)} domains")
        self.logger.info(f"Found {sum(len(v) for v in self.industry_terms.values())} industry terms")
        self.logger.info(f"Saved agent analysis: {self.output_dir / 'exam_analysis_agent.json'}")
        self.logger.info(f"Saved user guide: {self.output_dir / 'exam_study_guide.md'}")
        self.logger.info("=" * 80)


if __name__ == "__main__":
    analyzer = ExamPatternAnalyzer()
    analyzer.run()
