#!/usr/bin/env python3
"""
Exam PDF Extraction Script
Reads question and answer PDFs from exams_and_answers directory,
matches pairs, and outputs unified JSON files with extracted questions.
"""

import os
import re
import json
import pdfplumber
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ExamExtractor:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.questions_dir = self.base_dir / "exams_and_answers" / "questions"
        self.answers_dir = self.base_dir / "exams_and_answers" / "answers"
        self.output_dir = self.base_dir / "exams_and_answers" / "extracted_json"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_exam_pairs(self) -> Dict[str, Tuple[Optional[Path], Optional[Path]]]:
        """
        Return dict mapping exam names to (questions_pdf, answers_pdf) pairs.
        Handles variants (e.g., 2013_april_variant).
        """
        pairs = {}

        # Get all question PDFs
        question_files = list(self.questions_dir.glob("*_questions*.pdf"))
        answer_files = list(self.answers_dir.glob("*_answers*.pdf"))

        # Build dict of question files
        question_map = {}
        for qf in question_files:
            # Extract exam name (everything before _questions or _questions_variant)
            match = re.match(r"(.+?)_questions(?:_variant)?.pdf$", qf.name)
            if match:
                base_name = match.group(1)
                question_map[qf.name] = (base_name, qf)

        # Match with answers
        for af in answer_files:
            match = re.match(r"(.+?)_answers(?:_variant)?.pdf$", af.name)
            if match:
                base_name = match.group(1)
                # Find corresponding question file
                q_variant = f"{base_name}_questions_variant.pdf"
                q_standard = f"{base_name}_questions.pdf"

                if af.name.endswith("_variant.pdf"):
                    # Look for variant questions
                    if q_variant in question_map:
                        exam_name = base_name + "_variant"
                        pairs[exam_name] = (question_map[q_variant][1], af)
                else:
                    # Look for standard questions
                    if q_standard in question_map:
                        pairs[base_name] = (question_map[q_standard][1], af)
                    elif q_variant not in question_map and base_name not in pairs:
                        # Only standard exists, no variant
                        if q_standard in question_map:
                            pairs[base_name] = (question_map[q_standard][1], af)

        return pairs

    def parse_exam_name(self, exam_name: str) -> Tuple[int, str]:
        """Extract year and month from exam name like '2025_april' or '2013_april_variant'."""
        exam_name = exam_name.replace("_variant", "")
        parts = exam_name.split("_")
        year = int(parts[0])
        month = parts[1]
        return year, month

    def extract_answers(self, pdf_path: Path) -> Dict[int, str]:
        """
        Extract answers from single-page answer PDF.
        Returns dict mapping question number to answer letter (a/b/c/d).
        """
        answers = {}
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                # Answers are typically in first page
                text = pdf.pages[0].extract_text()

                # Find all lines with "Q.No" or just question numbers with answers
                # Pattern: number(s) followed by letter (a-d), possibly with Q.No or Q. prefix
                pattern = r"(\d+)\s+([a-d])"
                matches = re.findall(pattern, text)

                for q_num_str, answer in matches:
                    q_num = int(q_num_str)
                    if 1 <= q_num <= 100:
                        answers[q_num] = answer.lower()

        except Exception as e:
            print(f"  ERROR: Failed to extract answers from {pdf_path.name}: {e}")
            return {}

        return answers

    def extract_questions(self, pdf_path: Path) -> Dict[int, Dict]:
        """
        Extract questions from PDF.
        Returns dict mapping question number to question data.
        """
        questions = {}
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                # Combine all text from all pages
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

                # Find all Q markers
                q_pattern = r"Q(\d+)\."
                matches = list(re.finditer(q_pattern, full_text))

                if not matches:
                    return {}

                # Skip instruction/preamble Q1 if it exists (find first real question)
                real_start_idx = 0
                for i, m in enumerate(matches):
                    if m.group(1) == '1':
                        start_pos = m.start()
                        context = full_text[start_pos:start_pos+300]
                        # Check if this looks like a real question (typical question starters)
                        if any(phrase in context for phrase in 
                               ["is always true", "binary number", "combination", 
                                "appropriate description", "desk that", "employees get"]):
                            real_start_idx = i
                            break

                # Process questions from real start
                for i in range(real_start_idx, len(matches)):
                    m = matches[i]
                    q_num = int(m.group(1))
                    if q_num > 100:
                        continue

                    # Extract text from this question to next question
                    start_pos = m.start()
                    if i + 1 < len(matches):
                        end_pos = matches[i + 1].start()
                    else:
                        end_pos = len(full_text)

                    q_text = full_text[start_pos:end_pos].strip()

                    # Parse question text and options
                    q_data = self._parse_question_text(q_text, q_num)
                    if q_data:
                        questions[q_num] = q_data

        except Exception as e:
            print(f"  ERROR: Failed to extract questions from {pdf_path.name}: {e}")
            return {}

        return questions

    def _parse_question_text(self, q_text: str, q_num: int) -> Optional[Dict]:
        """
        Parse question text to extract question body and options (a, b, c, d).
        Handles both multi-line and inline option formats.
        """
        lines = q_text.split("\n")

        # Find where options start
        options_start = -1
        for idx, line in enumerate(lines):
            if re.match(r"^[a-d]\)", line.strip()):
                options_start = idx
                break

        # Extract options
        options = {}

        if options_start >= 0:
            # Multi-line format: options on separate lines
            q_body_lines = lines[:options_start]
            q_body = " ".join([l.strip() for l in q_body_lines if l.strip()])

            # Parse options from separate lines
            current_option = None
            current_text = []

            for line in lines[options_start:]:
                line = line.strip()
                if not line:
                    continue

                m = re.match(r"^([a-d])\)\s*(.*)", line)
                if m:
                    # Save previous option
                    if current_option:
                        options[current_option] = " ".join(current_text).strip()

                    current_option = m.group(1)
                    current_text = [m.group(2)] if m.group(2) else []
                else:
                    # Continue current option text
                    if current_option:
                        current_text.append(line)

            # Save last option
            if current_option:
                options[current_option] = " ".join(current_text).strip()

        else:
            # Inline format: options on same line (e.g., "a) text b) text c) text d) text")
            # Look for options anywhere in the text
            full_line = " ".join([l.strip() for l in lines])

            # Try to split by option pattern
            parts = re.split(r"\s+([a-d])\)", full_line)
            # parts will be like: [question_text, 'a', 'option_a_text', 'b', 'option_b_text', ...]

            if len(parts) >= 9:  # Need at least: text, a, text, b, text, c, text, d, text
                q_body = parts[0].strip()

                # Extract options from alternating pattern
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        opt_letter = parts[i]
                        opt_text = parts[i + 1].strip()

                        # Remove trailing content that might belong to next section
                        # Truncate at common delimiters
                        for delim in [" Q", "\nQ"]:
                            if delim in opt_text:
                                opt_text = opt_text[:opt_text.index(delim)].strip()

                        options[opt_letter] = opt_text

        # Remove "Qn." prefix from body
        q_body = re.sub(r"^Q\d+\.\s*", "", q_body)

        # Validate we have all 4 options with non-empty text
        if not all(opt in options for opt in ['a', 'b', 'c', 'd']):
            return None

        # Check that all options have text
        if any(len(options[opt].strip()) == 0 for opt in ['a', 'b', 'c', 'd']):
            return None

        return {
            "q_num": q_num,
            "text": q_body,
            "options": options
        }

    def process_exam(self, exam_name: str, questions_pdf: Path, answers_pdf: Path) -> bool:
        """
        Process a single exam: extract questions and answers, merge, and output JSON.
        Returns True if successful, False otherwise.
        """
        print(f"Processing {exam_name}... ", end="", flush=True)

        try:
            # Extract answers first (usually simpler)
            answers = self.extract_answers(answers_pdf)
            if not answers:
                print(f"FAILED (no answers extracted)")
                return False

            # Extract questions
            questions = self.extract_questions(questions_pdf)
            if not questions:
                print(f"FAILED (no questions extracted)")
                return False

            # Check we got all 100 questions
            if len(questions) < 100:
                print(f"FAILED (only {len(questions)}/100 questions extracted)")
                return False

            # Build output structure
            year, month = self.parse_exam_name(exam_name)

            output = {
                "metadata": {
                    "exam_name": exam_name,
                    "year": year,
                    "month": month,
                    "total_questions": 100,
                    "extraction_date": datetime.now().strftime("%Y-%m-%d")
                },
                "questions": []
            }

            # Merge questions with answers
            for q_num in range(1, 101):
                if q_num not in questions:
                    print(f"FAILED (missing Q{q_num})")
                    return False

                q_data = questions[q_num].copy()
                q_data["correct_answer"] = answers.get(q_num, "")

                # Validate answer is set
                if not q_data["correct_answer"]:
                    print(f"FAILED (missing answer for Q{q_num})")
                    return False

                output["questions"].append(q_data)

            # Write JSON
            output_path = self.output_dir / f"{exam_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            print("OK")
            return True

        except Exception as e:
            print(f"FAILED ({e})")
            return False

    def run(self):
        """Run extraction for all exam pairs."""
        pairs = self.get_exam_pairs()

        if not pairs:
            print("ERROR: No exam pairs found in directories.")
            return

        print(f"Found {len(pairs)} exam(s) to process.\n")

        success_count = 0
        failure_count = 0

        # Sort by exam name for consistent processing
        for exam_name in sorted(pairs.keys()):
            questions_pdf, answers_pdf = pairs[exam_name]

            if questions_pdf is None or answers_pdf is None:
                print(f"Processing {exam_name}... FAILED (missing PDF files)")
                failure_count += 1
                continue

            if self.process_exam(exam_name, questions_pdf, answers_pdf):
                success_count += 1
            else:
                failure_count += 1

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total exams: {success_count + failure_count}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed: {failure_count}")

        # Count output files
        output_files = list(self.output_dir.glob("*.json"))
        print(f"JSON files created: {len(output_files)}")

        if output_files:
            print(f"\nOutput directory: {self.output_dir}")
            print("Files created:")
            for f in sorted(output_files):
                print(f"  - {f.name}")


if __name__ == "__main__":
    # Get base directory (script location's parent's parent)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    extractor = ExamExtractor(str(base_dir))
    extractor.run()
