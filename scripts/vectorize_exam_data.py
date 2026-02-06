#!/usr/bin/env python3
"""
Vectorize exam questions using SBERT embeddings.

Loads questions from refined JSON files, embeds them with sentence-transformers,
and saves vectors and metadata for downstream analysis.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


class ExamVectorizer:
    """Vectorizes exam questions using SBERT embeddings."""

    def __init__(
        self,
        input_dir: str = "data/refined_json",
        output_dir: str = "data/vectorized",
        log_dir: str = "logs",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the vectorizer.

        Args:
            input_dir: Path to directory containing refined JSON exam files
            output_dir: Path to save vectorized outputs
            log_dir: Path to save logs
            model_name: HuggingFace model identifier for SBERT
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.model_name = model_name

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Data containers
        self.questions: List[Tuple[str, int, str, str]] = []
        self.vectors: np.ndarray = None
        self.metadata: List[Dict] = []
        self.model: SentenceTransformer = None

        self.logger.info("ExamVectorizer initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Model: {self.model_name}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging to file and console."""
        logger = logging.getLogger("ExamVectorizer")
        logger.setLevel(logging.INFO)

        # Remove existing handlers
        logger.handlers.clear()

        # File handler
        log_file = (
            self.log_dir
            / f"vectorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    def load_questions(self) -> int:
        """
        Load all questions from refined JSON files.

        Returns:
            Number of questions loaded
        """
        self.logger.info("Loading questions from refined JSON files...")

        json_files = sorted(self.input_dir.glob("*.json"))
        self.logger.info(f"Found {len(json_files)} JSON files")

        total_questions = 0
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                exam_name = data.get("metadata", {}).get("exam_name", json_file.stem)
                questions = data.get("questions", [])

                for q in questions:
                    q_num = q.get("q_num")
                    text = q.get("text", "")
                    correct_answer = q.get("correct_answer", "")

                    # Skip empty questions
                    if not text or not text.strip():
                        self.logger.warning(
                            f"Skipping empty question {q_num} in {exam_name}"
                        )
                        continue

                    self.questions.append((exam_name, q_num, text, correct_answer))
                    total_questions += 1

                self.logger.info(
                    f"Loaded {len(questions)} questions from {exam_name}"
                )

            except Exception as e:
                self.logger.error(f"Error loading {json_file}: {e}")
                continue

        self.logger.info(f"Total questions loaded: {total_questions}")
        return total_questions

    def _load_model(self) -> None:
        """Load SBERT model."""
        self.logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.logger.info(
            f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
        )

    def vectorize(self, batch_size: int = 32) -> int:
        """
        Embed all questions with SBERT.

        Args:
            batch_size: Number of questions to embed per batch

        Returns:
            Number of vectors created
        """
        if not self.questions:
            self.logger.warning("No questions to vectorize")
            return 0

        self.logger.info("Starting vectorization...")
        self._load_model()

        # Extract question texts
        texts = [q[2] for q in self.questions]

        # Embed in batches
        self.logger.info(f"Embedding {len(texts)} questions in batches of {batch_size}")
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )

        self.vectors = embeddings
        self.logger.info(
            f"Vectorization complete. Shape: {self.vectors.shape}"
        )

        # Build metadata
        self.logger.info("Building metadata...")
        for i, (exam_name, q_num, text, correct_answer) in enumerate(self.questions):
            self.metadata.append(
                {
                    "index": i,
                    "exam": exam_name,
                    "q_num": q_num,
                    "question_text": text,
                    "correct_answer": correct_answer,
                }
            )

        return len(embeddings)

    def save(self) -> None:
        """Save vectors, metadata, and configuration."""
        self.logger.info("Saving outputs...")

        # Save vectors
        vectors_file = self.output_dir / "vectors.npz"
        np.savez_compressed(vectors_file, vectors=self.vectors)
        self.logger.info(f"Vectors saved to {vectors_file}")

        # Save metadata
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Metadata saved to {metadata_file}")

        # Save configuration
        config = {
            "model_name": self.model_name,
            "embedding_dimension": int(self.vectors.shape[1]),
            "total_questions": len(self.metadata),
            "total_exams": len(set(m["exam"] for m in self.metadata)),
            "vectorization_timestamp": datetime.now().isoformat(),
        }
        config_file = self.output_dir / "embedding_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Configuration saved to {config_file}")

    def run(self, verbose: bool = False) -> None:
        """
        Execute the full vectorization pipeline.

        Args:
            verbose: Enable verbose output
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting exam data vectorization pipeline")
        self.logger.info("=" * 80)

        # Load questions
        num_loaded = self.load_questions()
        if num_loaded == 0:
            self.logger.error("No questions loaded. Exiting.")
            return

        # Vectorize
        num_vectorized = self.vectorize()

        # Save outputs
        self.save()

        # Summary
        self.logger.info("=" * 80)
        self.logger.info("Vectorization pipeline complete!")
        self.logger.info(f"Loaded {num_loaded} questions from {len(set(q[0] for q in self.questions))} exams")
        self.logger.info(f"Embedded with {self.model_name}")
        self.logger.info(f"Saved vectors: {self.output_dir / 'vectors.npz'}")
        self.logger.info(f"Saved metadata: {self.output_dir / 'metadata.json'}")
        self.logger.info(f"Saved config: {self.output_dir / 'embedding_config.json'}")
        self.logger.info("=" * 80)


if __name__ == "__main__":
    vectorizer = ExamVectorizer()
    vectorizer.run(verbose=False)
