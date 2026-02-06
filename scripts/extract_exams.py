#!/usr/bin/env python3
"""
Exam PDF Extraction Script
Reads question and answer PDFs from exams_and_answers directory,
matches pairs, and outputs unified JSON files with extracted questions.
Uses dual-pass extraction (raw PDF + intermediary) for robustness.
"""

import os
import re
import json
import sys
import pdfplumber
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ExamExtractor:
    def __init__(self, base_dir: str, verbose: bool = False):
        self.base_dir = Path(base_dir)
        self.questions_dir = self.base_dir / "exams_and_answers" / "questions"
        self.answers_dir = self.base_dir / "exams_and_answers" / "answers"
        self.output_dir = self.base_dir / "data" / "extracted_json"
        self.logs_dir = self.base_dir / "logs"
        self.verbose = verbose

        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.log_file = self.logs_dir / f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_entries = []

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

    def extract_questions_raw(self, pdf_path: Path) -> Dict[int, Dict]:
        """
        Pass 1: Extract questions directly from PDF text using Q# boundaries.
        Splits by Q\d+\. pattern (Q1., Q2., etc.) to isolate each question block.
        Then extracts options within each block.
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

                # Pre-filter: Remove page markers (– \d+ –) that interfere with boundary detection
                full_text, marker_count = self._remove_page_markers(full_text)
                if marker_count > 0:
                    self.log(f"  RAW: Removed {marker_count} page markers")

                # Split by Q# boundary: Q\d+\. matches "Q1. ", "Q2. ", etc.
                q_blocks = self._split_by_question_boundaries(full_text)

                if not q_blocks:
                    return {}

                skipped = []
                for q_num, q_block in q_blocks.items():
                    # Extract options within this question block
                    options = self._extract_options_from_block(q_block)

                    # Validate: must have all 4 options
                    if len(options) == 4:
                        q_body = self._extract_question_body(q_block)
                        q_data = {
                            "q_num": q_num,
                            "text": q_body,
                            "options": {
                                "a": options[0],
                                "b": options[1],
                                "c": options[2],
                                "d": options[3]
                            }
                        }
                        questions[q_num] = q_data
                    else:
                        skipped.append(f"Q{q_num} (found {len(options)}/4 options)")

                if self.verbose and skipped:
                    self.log(f"  RAW: Skipped {len(skipped)} questions: {', '.join(skipped[:5])}")

        except Exception as e:
            self.log(f"ERROR: Failed to extract questions (raw) from {pdf_path.name}: {e}")
            return {}

        return questions

    def extract_questions_intermediary(self, pdf_path: Path) -> Dict[int, Dict]:
        """
        Pass 2: Convert PDF to intermediary text format using Q# boundaries, then extract.
        Normalizes whitespace first, then splits by Q# boundaries.
        Returns dict mapping question number to question data.
        """
        questions = {}
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                # Extract text page by page, handling layout
                full_text = ""
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

                # Pre-filter: Remove page markers (– \d+ –) that interfere with boundary detection
                full_text, marker_count = self._remove_page_markers(full_text)
                if marker_count > 0:
                    self.log(f"  INTERMEDIARY: Removed {marker_count} page markers")

                # Normalize whitespace and line endings
                full_text = re.sub(r'\r\n', '\n', full_text)
                full_text = re.sub(r'\n{3,}', '\n\n', full_text)

                # Split by Q# boundary: Q\d+\. matches "Q1. ", "Q2. ", etc.
                q_blocks = self._split_by_question_boundaries(full_text)

                if not q_blocks:
                    return {}

                skipped = []
                for q_num, q_block in q_blocks.items():
                    # Extract options within this question block
                    options = self._extract_options_from_block(q_block)

                    # Validate: must have all 4 options
                    if len(options) == 4:
                        q_body = self._extract_question_body(q_block)
                        q_data = {
                            "q_num": q_num,
                            "text": q_body,
                            "options": {
                                "a": options[0],
                                "b": options[1],
                                "c": options[2],
                                "d": options[3]
                            }
                        }
                        questions[q_num] = q_data
                    else:
                        skipped.append(f"Q{q_num} (found {len(options)}/4 options)")

                if self.verbose and skipped:
                    self.log(f"  INTERMEDIARY: Skipped {len(skipped)} questions: {', '.join(skipped[:5])}")

        except Exception as e:
            self.log(f"ERROR: Failed to extract questions (intermediary) from {pdf_path.name}: {e}")
            return {}

        return questions

    def _remove_page_markers(self, text: str) -> Tuple[str, int]:
        """
        Remove page markers (– \d+ –) from text before processing.
        These markers interfere with Q# and option boundary detection.
        Returns tuple of (cleaned_text, marker_count).
        """
        pattern = r'–\s*\d+\s*–'
        cleaned = re.sub(pattern, '', text)
        marker_count = len(re.findall(pattern, text))
        return cleaned, marker_count

    def _split_by_question_boundaries(self, text: str) -> Dict[int, str]:
        """
        Split text by Q# boundaries.
        Returns dict mapping question number to the text block for that question.

        Uses multiple patterns for robustness:
        - Primary: Q\d+\. (Q followed by digits and literal period)
        - Fallback: Q\d+\) (Q followed by digits and closing paren)
        - Fallback: Q\d+\: (Q followed by digits and colon)

        Splits on these patterns and isolates text from Q[n] to Q[n+1].
        Skips everything before Q1 (sample questions, instructions).
        """
        q_blocks = {}

        # Find all Q# markers with improved regex patterns
        # Try primary pattern first (Q\d+\.), then fallback to other formats
        primary_pattern = r"Q\s*(\d+)\."
        fallback_patterns = [
            r"Q\s*(\d+)\)",
            r"Q\s*(\d+)\:"
        ]

        # Collect all matches from all patterns
        all_matches = list(re.finditer(primary_pattern, text))

        # Add fallback matches only if primary pattern didn't find enough
        if len(all_matches) < 50:  # If primary pattern finds fewer than 50 questions
            for pattern in fallback_patterns:
                fallback_matches = list(re.finditer(pattern, text))
                # Merge, avoiding duplicates (by position)
                match_positions = {m.start() for m in all_matches}
                for m in fallback_matches:
                    if m.start() not in match_positions:
                        all_matches.append(m)
            # Sort by position after merging
            all_matches.sort(key=lambda m: m.start())

        matches = all_matches

        if not matches:
            return {}

        # Process each match
        for i, match in enumerate(matches):
            q_num_str = match.group(1)
            try:
                q_num = int(q_num_str)
                if not (1 <= q_num <= 100):
                    continue
            except ValueError:
                continue

            # Get start of this question block (after the Q# marker)
            block_start = match.end()

            # Get end of this question block (start of next Q# marker, or end of text)
            if i + 1 < len(matches):
                block_end = matches[i + 1].start()
            else:
                block_end = len(text)

            # Extract block text
            block_text = text[block_start:block_end].strip()
            q_blocks[q_num] = block_text

        return q_blocks

    def _extract_options_from_block(self, block_text: str) -> List[str]:
        """
        Extract options a, b, c, d from a question block.
        Finds [a-d]) or [a-d]. patterns and captures text until next option.
        Validates options are non-empty and handles edge cases near page breaks.
        Returns list of 4 option texts in order [a_text, b_text, c_text, d_text].
        Returns shorter list if fewer than 4 options found.
        """
        options = []

        # Find all option markers: a) or a. or b) or b. etc.
        pattern = r"[a-d][).]"
        matches = list(re.finditer(pattern, block_text))

        if not matches:
            return []

        # Process each option marker
        for i, match in enumerate(matches):
            # Skip if this option letter is out of sequence
            option_letter = block_text[match.start()].lower()
            if option_letter != chr(ord('a') + len(options)):
                break  # Stop if options are not sequential

            # Get start of option text (after the marker)
            opt_start = match.end()

            # Get end of option text (start of next option marker, or end of block)
            if i + 1 < len(matches):
                opt_end = matches[i + 1].start()
            else:
                opt_end = len(block_text)

            # Extract and clean option text
            opt_text = block_text[opt_start:opt_end].strip()

            # Validate: option text should not be empty
            if not opt_text:
                # Log potential malformed option and break
                if self.verbose:
                    self.log(f"    WARNING: Option {option_letter} is empty (potential truncation)")
                break

            # Check if option text is suspiciously short (< 5 chars) - likely truncated
            if len(opt_text) < 5 and i < 3:  # Allow short last option
                if self.verbose:
                    self.log(f"    WARNING: Option {option_letter} is very short ({len(opt_text)} chars), may be truncated")

            options.append(opt_text)

        return options

    def _extract_question_number(self, text: str) -> Optional[int]:
        """
        Extract question number from text using Q# pattern.
        Returns the question number as int, or None if not found.
        """
        match = re.search(r"Q\s*(\d+)", text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _extract_question_body(self, text: str) -> str:
        """
        Extract question body by removing the Q# prefix.
        Returns the question text (everything after Q# marker).
        """
        # Remove Q# and any following whitespace/punctuation
        body = re.sub(r"^.*?Q\s*\d+[.):\s]*", "", text, flags=re.DOTALL)
        return body.strip()

    def log(self, message: str):
        """Log message to file and optionally to stdout if verbose."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_entries.append(log_message)
        if self.verbose:
            print(message)

    def write_logs(self):
        """Write all accumulated log entries to log file."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                for entry in self.log_entries:
                    f.write(entry + '\n')
        except Exception as e:
            print(f"WARNING: Failed to write logs to {self.log_file}: {e}")

    def process_exam(self, exam_name: str, questions_pdf: Path, answers_pdf: Path) -> bool:
        """
        Process a single exam: dual-pass extraction, answers, and output JSON.
        Returns True if successful, False otherwise.
        """
        if not self.verbose:
            print(f"Processing {exam_name}... ", end="", flush=True)

        self.log(f"\n{'='*60}")
        self.log(f"Processing exam: {exam_name}")
        self.log(f"{'='*60}")

        try:
            # Extract answers first (usually simpler)
            answers = self.extract_answers(answers_pdf)
            if not answers:
                msg = f"FAILED: No answers extracted"
                if not self.verbose:
                    print(msg)
                self.log(msg)
                return False

            self.log(f"Answers extracted: {len(answers)} found")

            # PASS 1: Raw PDF extraction
            questions_raw = self.extract_questions_raw(questions_pdf)
            self.log(f"PASS 1 (RAW PDF): {len(questions_raw)} questions extracted")

            # PASS 2: Intermediary extraction
            questions_intermediary = self.extract_questions_intermediary(questions_pdf)
            self.log(f"PASS 2 (INTERMEDIARY): {len(questions_intermediary)} questions extracted")

            # Check discrepancies
            raw_set = set(questions_raw.keys())
            inter_set = set(questions_intermediary.keys())
            discrepancies = raw_set.symmetric_difference(inter_set)
            if discrepancies:
                self.log(f"Discrepancies between passes: {sorted(discrepancies)}")

            # Use the pass with more questions, or raw if equal
            if len(questions_intermediary) > len(questions_raw):
                best_questions = questions_intermediary
                best_pass = "INTERMEDIARY"
            else:
                best_questions = questions_raw
                best_pass = "RAW"

            self.log(f"Selected {best_pass} pass ({len(best_questions)} questions)")

            # If we have fewer than 100, try to fill from the other pass
            if len(best_questions) < 100:
                other_pass = "RAW" if best_pass == "INTERMEDIARY" else "INTERMEDIARY"
                other_questions = questions_raw if best_pass == "INTERMEDIARY" else questions_intermediary
                missing = set(range(1, 101)) - set(best_questions.keys())
                filled = 0
                for q_num in missing:
                    if q_num in other_questions:
                        best_questions[q_num] = other_questions[q_num]
                        filled += 1
                if filled > 0:
                    self.log(f"Filled {filled} missing questions from {other_pass} pass")

            # Validate: Check for gaps in Q# sequence and log diagnostics
            extracted_q_nums = sorted(best_questions.keys())
            if extracted_q_nums:
                gaps = []
                for i in range(1, 101):
                    if i not in best_questions:
                        gaps.append(i)

                if gaps:
                    # Find contiguous ranges of missing questions
                    gap_ranges = []
                    start = gaps[0]
                    end = gaps[0]
                    for q in gaps[1:]:
                        if q == end + 1:
                            end = q
                        else:
                            gap_ranges.append((start, end))
                            start = q
                            end = q
                    gap_ranges.append((start, end))

                    gap_summary = ", ".join([f"Q{s}-Q{e}" if s != e else f"Q{s}" for s, e in gap_ranges])
                    self.log(f"Missing questions (extraction boundary issue): {gap_summary}")
                    self.log(f"Extraction rate: {len(best_questions)}/100 ({len(best_questions)}%)")

            # Build output structure
            year, month = self.parse_exam_name(exam_name)

            # Process PASS 1: Raw PDF
            output_raw = {
                "metadata": {
                    "exam_name": exam_name,
                    "year": year,
                    "month": month,
                    "extraction_method": "raw_pdf",
                    "extraction_date": datetime.now().strftime("%Y-%m-%d"),
                    "total_questions_extracted": len(questions_raw)
                },
                "questions": []
            }

            for q_num in range(1, 101):
                if q_num in questions_raw:
                    q_data = questions_raw[q_num].copy()
                    q_data["correct_answer"] = answers.get(q_num, "")
                    output_raw["questions"].append(q_data)

            # Process PASS 2: Intermediary
            output_intermediary = {
                "metadata": {
                    "exam_name": exam_name,
                    "year": year,
                    "month": month,
                    "extraction_method": "intermediary",
                    "extraction_date": datetime.now().strftime("%Y-%m-%d"),
                    "total_questions_extracted": len(questions_intermediary)
                },
                "questions": []
            }

            for q_num in range(1, 101):
                if q_num in questions_intermediary:
                    q_data = questions_intermediary[q_num].copy()
                    q_data["correct_answer"] = answers.get(q_num, "")
                    output_intermediary["questions"].append(q_data)

            # Write JSON for PASS 1
            output_path_raw = self.output_dir / f"{exam_name}_raw_pdf.json"
            try:
                with open(output_path_raw, 'w', encoding='utf-8') as f:
                    json.dump(output_raw, f, indent=2, ensure_ascii=False)
                self.log(f"Written: {output_path_raw.name} ({len(questions_raw)} questions)")
            except Exception as e:
                self.log(f"ERROR: Failed to write {output_path_raw.name}: {e}")
                return False

            # Write JSON for PASS 2
            output_path_intermediary = self.output_dir / f"{exam_name}_intermediary.json"
            try:
                with open(output_path_intermediary, 'w', encoding='utf-8') as f:
                    json.dump(output_intermediary, f, indent=2, ensure_ascii=False)
                self.log(f"Written: {output_path_intermediary.name} ({len(questions_intermediary)} questions)")
            except Exception as e:
                self.log(f"ERROR: Failed to write {output_path_intermediary.name}: {e}")
                return False

            # Verify files were actually created
            if not output_path_raw.exists() or not output_path_intermediary.exists():
                self.log(f"ERROR: Output files not verified on disk")
                if not self.verbose:
                    print("FAILED (files not written)")
                return False

            msg = f"OK ({len(questions_raw)} raw, {len(questions_intermediary)} intermediary)"
            if not self.verbose:
                print(msg)
            self.log(msg)
            return True

        except Exception as e:
            msg = f"FAILED ({e})"
            if not self.verbose:
                print(msg)
            self.log(msg)
            return False

    def run(self):
        """Run extraction for all exam pairs."""
        pairs = self.get_exam_pairs()

        if not pairs:
            print("ERROR: No exam pairs found in directories.")
            self.log("ERROR: No exam pairs found in directories.")
            return

        if not self.verbose:
            print(f"Found {len(pairs)} exam(s) to process.\n")
        self.log(f"Found {len(pairs)} exam(s) to process.")
        self.log(f"Verbose mode: {self.verbose}")
        self.log(f"Output directory: {self.output_dir}")

        success_count = 0
        failure_count = 0

        # Sort by exam name for consistent processing
        for exam_name in sorted(pairs.keys()):
            questions_pdf, answers_pdf = pairs[exam_name]

            if questions_pdf is None or answers_pdf is None:
                msg = f"Processing {exam_name}... FAILED (missing PDF files)"
                print(msg)
                self.log(msg)
                failure_count += 1
                continue

            if self.process_exam(exam_name, questions_pdf, answers_pdf):
                success_count += 1
            else:
                failure_count += 1

        # Print summary
        summary_lines = [
            "",
            f"{'='*60}",
            "SUMMARY",
            f"{'='*60}",
            f"Total exams: {success_count + failure_count}",
            f"Successfully processed: {success_count}",
            f"Failed: {failure_count}",
        ]

        # Count output files
        output_files = list(self.output_dir.glob("*.json"))
        summary_lines.extend([
            f"JSON files created: {len(output_files)}",
        ])

        if output_files:
            summary_lines.extend([
                f"",
                f"Output directory: {self.output_dir}",
                "Files created:",
            ])
            for f in sorted(output_files):
                summary_lines.append(f"  - {f.name}")

        summary_lines.extend([
            f"",
            f"Logs written to: {self.log_file}",
        ])

        for line in summary_lines:
            print(line)
            self.log(line)

        # Write all logs to file
        self.write_logs()


if __name__ == "__main__":
    # Get base directory (script location's parent's parent)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    # Parse CLI arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    extractor = ExamExtractor(str(base_dir), verbose=verbose)
    extractor.run()
