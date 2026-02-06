#!/usr/bin/env python3
"""
Exam Data Cleaning Script

Cleans extracted JSON exam files by:
- Removing page markers (– # –)
- Normalizing whitespace
- Validating question structure
- Detecting truncation issues
"""

import json
import re
from pathlib import Path


class ExamDataCleaner:
    def __init__(self, input_dir='data/extracted_json',
                 output_dir='data/cleaned_json'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.stats = {
            'files_processed': 0,
            'page_markers_removed': 0,
            'whitespace_normalizations': 0,
            'questions_validated': 0,
            'truncation_suspects': 0,
            'invalid_structures': 0,
        }
        self.truncation_suspects = []
        self.invalid_structures = []

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def remove_page_markers(self, text):
        """Remove '– # –' patterns (em-dash, number, em-dash)."""
        if not text:
            return text

        original = text
        # Pattern: em-dash (–), optional whitespace, one or more digits, optional whitespace, em-dash (–)
        text = re.sub(r'–\s*\d+\s*–', ' ', text)

        if text != original:
            self.stats['page_markers_removed'] += 1

        return text

    def normalize_whitespace(self, text):
        """Collapse multiple newlines/spaces to single space, strip leading/trailing."""
        if not text:
            return text

        original = text
        # Replace newlines and tabs with spaces
        text = re.sub(r'[\n\t]+', ' ', text)
        # Collapse multiple spaces to single space
        text = re.sub(r' +', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()

        if text != original:
            self.stats['whitespace_normalizations'] += 1

        return text

    def detect_truncation(self, text):
        """Detect suspicious endings that suggest truncation."""
        if not text:
            return False

        # Look for incomplete words or suspicious endings
        suspicious_patterns = [
            r'[a-z](\s|$)',  # Single letter followed by space or end
            r'(\s|^)et(\s|$)',  # Isolated "et"
            r'(\s|^)etc\.?(\s|$)',  # "etc" or "etc."
            r'[a-z]{1,2}(\s|,)$',  # Short word at end
            r'[a-zA-Z]*[aeiou]{2,}(\s|$)',  # Incomplete-sounding ending
        ]

        # More aggressive: check if text ends with incomplete phrase patterns
        if re.search(r'\s[a-z]{1,3}$', text):  # Short word at very end
            return True

        if re.search(r',\s*$', text):  # Trailing comma
            return True

        return False

    def validate_question(self, question):
        """Validate question structure. Return (is_valid, issues_list)."""
        issues = []

        # Check q_num
        if 'q_num' not in question:
            issues.append("Missing q_num")
        elif not isinstance(question['q_num'], int):
            issues.append(f"q_num is not int: {type(question['q_num'])}")
        elif not (1 <= question['q_num'] <= 100):
            issues.append(f"q_num out of range: {question['q_num']}")

        # Check options
        if 'options' not in question:
            issues.append("Missing options")
        elif not isinstance(question['options'], dict):
            issues.append("options is not dict")
        elif set(question['options'].keys()) != {'a', 'b', 'c', 'd'}:
            issues.append(f"options don't have exactly a,b,c,d: {set(question['options'].keys())}")

        # Check correct_answer
        if 'correct_answer' not in question:
            issues.append("Missing correct_answer")
        elif not isinstance(question['correct_answer'], str):
            issues.append(f"correct_answer is not string: {type(question['correct_answer'])}")
        elif question['correct_answer'] not in ['a', 'b', 'c', 'd']:
            issues.append(f"correct_answer not a/b/c/d: {question['correct_answer']}")

        # Check text
        if 'text' not in question:
            issues.append("Missing text")

        return (len(issues) == 0, issues)

    def clean_question(self, question):
        """Apply all cleaning operations to one question."""
        self.stats['questions_validated'] += 1

        # Deep copy to avoid modifying original
        cleaned = {
            'q_num': question.get('q_num'),
            'text': question.get('text', ''),
            'options': {},
            'correct_answer': question.get('correct_answer')
        }

        # Clean text
        cleaned['text'] = self.remove_page_markers(cleaned['text'])
        cleaned['text'] = self.normalize_whitespace(cleaned['text'])

        # Check for truncation in text
        if self.detect_truncation(cleaned['text']):
            self.stats['truncation_suspects'] += 1
            self.truncation_suspects.append({
                'exam': '?',  # Will be set by caller
                'q_num': cleaned['q_num'],
                'text': cleaned['text'][:100] + '...' if len(cleaned['text']) > 100 else cleaned['text']
            })

        # Clean options
        if 'options' in question and isinstance(question['options'], dict):
            for key, value in question['options'].items():
                if isinstance(value, str):
                    cleaned_opt = self.remove_page_markers(value)
                    cleaned_opt = self.normalize_whitespace(cleaned_opt)
                    cleaned['options'][key] = cleaned_opt

                    # Check for truncation in option
                    if self.detect_truncation(cleaned_opt):
                        self.stats['truncation_suspects'] += 1
                        if not any(s['q_num'] == cleaned['q_num'] for s in self.truncation_suspects):
                            self.truncation_suspects.append({
                                'exam': '?',
                                'q_num': cleaned['q_num'],
                                'text': f"Option {key}: {cleaned_opt[:80]}..."
                            })

        # Validate cleaned structure
        is_valid, issues = self.validate_question(cleaned)
        if not is_valid:
            self.stats['invalid_structures'] += 1
            self.invalid_structures.append({
                'exam': '?',
                'q_num': cleaned['q_num'],
                'issues': issues
            })

        return cleaned

    def clean_exam_file(self, input_path):
        """Load JSON, clean all questions, save to output. Return summary."""
        input_path = Path(input_path)
        exam_name = input_path.stem.replace('_raw_pdf', '')

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'status': 'error', 'exam': exam_name, 'error': str(e)}

        # Extract metadata and questions
        metadata = data.get('metadata', {})
        questions = data.get('questions', [])

        # Clean all questions
        cleaned_questions = []
        for question in questions:
            cleaned_q = self.clean_question(question)
            cleaned_questions.append(cleaned_q)

        # Update truncation and invalid suspects with exam name
        for suspect in self.truncation_suspects[-len(questions):]:
            suspect['exam'] = exam_name
        for invalid in self.invalid_structures[-len(questions):]:
            invalid['exam'] = exam_name

        # Build output JSON
        output_data = {
            'metadata': metadata,
            'questions': cleaned_questions
        }

        # Write to output
        output_path = self.output_dir / input_path.name.replace('_raw_pdf.json', '.json')
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            return {'status': 'error', 'exam': exam_name, 'error': str(e)}

        self.stats['files_processed'] += 1
        return {
            'status': 'success',
            'exam': exam_name,
            'questions_cleaned': len(cleaned_questions)
        }

    def run(self, verbose=False):
        """Process all *_raw_pdf.json files and generate report."""
        # Find all raw PDF JSON files
        raw_files = sorted(self.input_dir.glob('*_raw_pdf.json'))

        if not raw_files:
            print(f"No files found in {self.input_dir}")
            return

        print(f"Found {len(raw_files)} files to process")
        print()

        # Process each file
        for input_path in raw_files:
            result = self.clean_exam_file(input_path)
            if verbose:
                print(f"  {result['exam']}: {result.get('questions_cleaned', 0)} questions cleaned")

        # Generate report
        print()
        print("=" * 70)
        print("CLEANING REPORT")
        print("=" * 70)
        print(f"Files processed:              {self.stats['files_processed']}")
        print(f"Total questions validated:   {self.stats['questions_validated']}")
        print(f"Page markers removed:        {self.stats['page_markers_removed']}")
        print(f"Whitespace normalizations:   {self.stats['whitespace_normalizations']}")
        print(f"Truncation suspects flagged: {self.stats['truncation_suspects']}")
        print(f"Invalid structures found:    {self.stats['invalid_structures']}")
        print()

        if self.truncation_suspects:
            print("-" * 70)
            print("TRUNCATION SUSPECTS (manual review recommended):")
            print("-" * 70)
            for suspect in self.truncation_suspects[:20]:  # Show first 20
                print(f"  {suspect['exam']} Q{suspect['q_num']}: {suspect['text']}")
            if len(self.truncation_suspects) > 20:
                print(f"  ... and {len(self.truncation_suspects) - 20} more")
            print()

        if self.invalid_structures:
            print("-" * 70)
            print("INVALID STRUCTURES (needs fixing):")
            print("-" * 70)
            for invalid in self.invalid_structures[:20]:  # Show first 20
                print(f"  {invalid['exam']} Q{invalid['q_num']}:")
                for issue in invalid['issues']:
                    print(f"    - {issue}")
            if len(self.invalid_structures) > 20:
                print(f"  ... and {len(self.invalid_structures) - 20} more")
            print()

        print("=" * 70)
        print(f"Cleaned JSON files saved to: {self.output_dir.absolute()}")
        print("=" * 70)


if __name__ == '__main__':
    import sys
    verbose = '--verbose' in sys.argv
    cleaner = ExamDataCleaner()
    cleaner.run(verbose=verbose)
