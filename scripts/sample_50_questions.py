#!/usr/bin/env python3
"""
Sample 50 random questions from metadata for manual cognitive type labeling.
Outputs a CSV ready for human annotation.
"""

import json
import csv
import random
from pathlib import Path

# Seed for reproducibility
random.seed(42)

# Paths
METADATA_PATH = Path(__file__).parent.parent / "data" / "vectorized" / "metadata.json"
OUTPUT_CSV = Path(__file__).parent.parent / "cognitive_labeling_sample.csv"

# Load metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

print(f"Loaded {len(metadata)} questions from metadata")

# Sample 50 random questions
sample = random.sample(metadata, 50)
sample_sorted = sorted(sample, key=lambda x: (x["exam"], x["q_num"]))

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "exam",
            "q_num",
            "question_text",
            "cognitive_type",
            "notes",
        ],
    )
    writer.writeheader()
    for q in sample_sorted:
        writer.writerow(
            {
                "exam": q["exam"],
                "q_num": q["q_num"],
                "question_text": q["question_text"],
                "cognitive_type": "",  # For you to fill in
                "notes": "",  # For you to fill in if needed
            }
        )

print(f"✓ Exported 50 sample questions to: {OUTPUT_CSV}")
print(f"  Categories: Definition Recall | Concept Application | Computation-Trace | Analysis-Design")
