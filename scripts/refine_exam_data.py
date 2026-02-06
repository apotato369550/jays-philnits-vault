#!/usr/bin/env python3
"""
Exam Data Refinement Script

Detects and removes option text that has leaked into the question field.

Detection Pattern:
- A question is contaminated if it contains all four option markers (a), b), c), d) in sequence
- Pattern: a\)\s*.*?\s*b\)\s*.*?\s*c\)\s*.*?\s*d\)

Refinement Logic:
1. Load cleaned JSON from data/cleaned_json/
2. For each question, check if it contains all four option patterns
3. If found, extract text before first "a)" as clean question
4. Log the refinement (exam, Q#, text removed)
5. Save refined JSON to data/refined_json/
6. Generate refinement report
"""

import json
import re
import sys
from pathlib import Path


class ExamDataRefiner:
    def __init__(self, input_dir='data/cleaned_json',
                 output_dir='data/refined_json'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.stats = {
            'files_processed': 0,
            'total_refinements': 0,
            'questions_examined': 0,
        }
        self.refinement_log = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_contamination(self, question_text):
        """Check if question text contains all four options a) b) c) d) in sequence"""
        pattern = r'a\)\s*.*?\s*b\)\s*.*?\s*c\)\s*.*?\s*d\)'
        return bool(re.search(pattern, question_text, re.DOTALL))

    def extract_clean_question(self, question_text):
        """
        Extract clean question by removing options.

        Returns:
            tuple: (clean_text, removed_text)
        """
        # Find first "a)" and extract text before it
        match = re.search(r'a\)', question_text)
        if match:
            clean_text = question_text[:match.start()].strip()
            removed_text = question_text[match.start():].strip()
            return clean_text, removed_text
        return question_text, ""

    def refine_exam_file(self, input_path):
        """
        Process single exam file.

        Args:
            input_path: Path to cleaned JSON file

        Returns:
            dict: Refined exam data
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            exam_data = json.load(f)

        exam_name = exam_data['metadata']['exam_name']

        # Process each question
        for question in exam_data['questions']:
            self.stats['questions_examined'] += 1
            q_num = question['q_num']
            original_text = question['text']

            # Check for contamination
            if self.detect_contamination(original_text):
                clean_text, removed_text = self.extract_clean_question(original_text)

                # Log refinement
                self.stats['total_refinements'] += 1
                log_entry = {
                    'exam': exam_name,
                    'q_num': q_num,
                    'text_removed_chars': len(removed_text),
                    'text_removed_preview': removed_text[:100] + ('...' if len(removed_text) > 100 else '')
                }
                self.refinement_log.append(log_entry)

                # Update question text
                question['text'] = clean_text

        return exam_data

    def run(self, verbose=False):
        """
        Process all cleaned exam files.

        Args:
            verbose: If True, print detailed refinement log
        """
        # Find all .json files in input directory
        json_files = sorted(self.input_dir.glob('*.json'))

        if not json_files:
            print(f"No JSON files found in {self.input_dir}")
            return

        # Process each file
        for input_path in json_files:
            refined_data = self.refine_exam_file(input_path)
            self.stats['files_processed'] += 1

            # Save refined JSON
            output_path = self.output_dir / input_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(refined_data, f, indent=2, ensure_ascii=False)

        # Print report
        self._print_report(verbose)

    def _print_report(self, verbose=False):
        """Print refinement report"""
        print("\n" + "="*70)
        print("EXAM DATA REFINEMENT REPORT")
        print("="*70)

        print(f"\nFiles processed:        {self.stats['files_processed']}")
        print(f"Questions examined:     {self.stats['questions_examined']}")
        print(f"Total refinements:      {self.stats['total_refinements']}")

        if self.stats['total_refinements'] > 0:
            print("\nRefinements by exam:")
            print("-" * 70)

            # Group refinements by exam
            exam_counts = {}
            for entry in self.refinement_log:
                exam = entry['exam']
                exam_counts[exam] = exam_counts.get(exam, 0) + 1

            for exam in sorted(exam_counts.keys()):
                count = exam_counts[exam]
                print(f"  {exam}: {count} refinement(s)")

            if verbose:
                print("\nDetailed refinements:")
                print("-" * 70)
                for entry in self.refinement_log:
                    print(f"  {entry['exam']} Q#{entry['q_num']}: "
                          f"removed {entry['text_removed_chars']} chars")
                    print(f"    Preview: {entry['text_removed_preview']}")
        else:
            print("\nNo contaminated questions found.")

        print(f"\nRefined JSON files saved to: {self.output_dir}")
        print("="*70 + "\n")


if __name__ == '__main__':
    verbose = '--verbose' in sys.argv
    refiner = ExamDataRefiner()
    refiner.run(verbose=verbose)
