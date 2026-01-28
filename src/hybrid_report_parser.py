# HYBRID MEDICAL REPORT PARSER v5.2
# Combines ML sentence classification + rule-based extraction
# Handles: X-ray, MRI, CT, Ultrasound, ECG, Lab/Biochemistry, Serology, Urinalysis
# Version: v5.2_hybrid_ml_table_aware

import re
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from lab_text_row_parser import build_lab_rows_from_text

def extract_patient_info(raw_text: str) -> Dict:
    """
    Deterministic patient info extractor (form-based, not NLP).
    Works across lab, imaging, ECG reports.
    """
    info = {}

    # Name (Mr / Mrs / Ms patterns common in Indian lab reports)
    name_match = re.search(r"\b(Mr|Mrs|Ms)\.?\s+([A-Z ]+)", raw_text, re.IGNORECASE)
    if name_match:
        info["name"] = name_match.group(2).title().strip()

    # Age / Sex (e.g. 24 Yrs / Male)
    age_sex_match = re.search(
        r"(\d+)\s*Yrs?\s*/\s*(Male|Female)",
        raw_text,
        re.IGNORECASE
    )
    if age_sex_match:
        info["age"] = int(age_sex_match.group(1))
        info["sex"] = age_sex_match.group(2).capitalize()

    # Report Date
    date_match = re.search(
        r"Date\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
        raw_text
    )
    if date_match:
        info["report_date"] = date_match.group(1)

    # Lab Number / Report Number
    lab_match = re.search(
        r"Lab\s*No\.?\s*[:\-]?\s*(\d+)",
        raw_text,
        re.IGNORECASE
    )
    if lab_match:
        info["lab_number"] = lab_match.group(1)

    return info

class MedicalSentenceClassifier:
    """
    Sentence-level classifier using rule-based + simple ML heuristics
    Assigns intent labels to each sentence in medical reports

    Labels:
    - patient_info: Demographics, IDs, dates
    - clinical_history: Chief complaint, indication, reason for study
    - findings: Observations, measurements, test results
    - diagnosis: Impression, interpretation, conclusion
    - recommendation: Follow-up, treatment, actions
    - administrative: Provider info, signatures, authorization
    - metadata: Noise/non-clinical
    """

    def __init__(self):
        # Keywords for each intent (comprehensive + synonyms)
        self.intent_keywords = {
            'patient_info': [
                'patient', 'name', 'age', 'sex', 'dob', 'mrn', 'id', 'uhid', 'hospital',
                'gender', 'date of birth', 'admission', 'discharge'
            ],
            'clinical_history': [
                'clinical history', 'indication', 'chief complaint', 'presentation',
                'reason for visit', 'reason for study', 'why here', 'presents with',
                'presenting with', 'history of', 'patient presents', 'referred for'
            ],
            'findings': [
                'findings', 'results', 'observation', 'noted', 'shows', 'demonstrates',
                'revealed', 'appears', 'no evidence', 'normal', 'clear', 'patent',
                'what we found', 'mmhg', 'bpm', 'unremarkable', 'intact',
                'cumm', 'fl', 'pg'
            ],
            'diagnosis': [
                'impression', 'interpretation', 'conclusion', 'diagnosis', 'assessment',
                'opinion', 'summary', 'doctor says', 'final diagnosis', 'consistent with',
                'further confirm', 'anemia'
            ],
            'recommendation': [
                'recommendation', 'follow-up', 'treatment', 'management', 'suggest',
                'advise', 'consider', 'monitor', 'next steps', 'do this', 'action'
            ],
            'administrative': [
                'reported by', 'authorized by', 'signed', 'license', 'registration',
                'dr.', 'doctor', 'physician', 'clinician', 'electronically'
            ]
        }

    def classify_sentence(self, sentence: str) -> Tuple[str, float]:
        """
        Classify a single sentence into intent label
        Returns: (label, confidence_score)
        """
        sentence_lower = sentence.lower().strip()

        # Skip empty sentences
        if not sentence_lower or len(sentence_lower) < 5:
            return 'metadata', 0.0

        # Score each intent
        scores = defaultdict(float)

        # Keyword matching
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    scores[intent] += 1.0

        # Structure-based heuristics
        # If sentence has "Name: Value" format → patient_info
        if re.match(r'^[A-Za-z\s]+:\s+[A-Za-z0-9\s\-\/]+$', sentence):
            scores['patient_info'] += 2.0

        # If sentence has measurements → findings
        if re.search(r'\d+\.?\d*\s*(?:g/dL|mmHg|%|bpm|°|cm|mm|cumm|fL|pg)', sentence, re.IGNORECASE):
            scores['findings'] += 2.0

        # Normalize scores
        if not scores:
            return 'metadata', 0.0

        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent] / max(sum(scores.values()), 1)

        return best_intent, confidence

    def classify_text(self, raw_text: str) -> List[Tuple[str, str, float]]:
        """
        Classify all sentences in text
        Returns: List of (sentence, intent_label, confidence)
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)(?=[A-Z])', raw_text)

        classified = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                label, confidence = self.classify_sentence(sentence)
                classified.append((sentence, label, confidence))

        return classified


class HybridMedicalReportParser:
    """
    Hybrid parser: ML sentence classifier + rule-based extractor
    Works with ANY report structure by classifying sentence intent first
    """

    def __init__(self):
        self.classifier = MedicalSentenceClassifier()

    def extract_sections_by_intent(self, classified_sentences: List[Tuple[str, str, float]]) -> Dict[str, List[str]]:
        """Group sentences by intent label"""
        sections = defaultdict(list)

        for sentence, intent, confidence in classified_sentences:
            if confidence > 0.3:
                sections[intent].append(sentence)

        return dict(sections)

    def extract_patient_info_ml(self, patient_sentences: List[str]) -> Dict:
        """Extract patient info from labeled sentences"""
        patient_info = {}
        combined_text = ' '.join(patient_sentences)

        # Name - multiple patterns
        for pattern in [
            r'^([A-Za-z\s]+)\s+(?:Sample|Age|Sex)',  # Name at start before Sample/Age/Sex
            r'(?:Patient\s+)?Name\s*:\s*([A-Za-z\s]+?)(?:\n|,|ID|Age)',
            r'^([A-Za-z\s]+?)(?:,|\s+\d+)',
            r'^Pt:\s*([A-Za-z\s]+?),',
        ]:
            match = re.search(pattern, combined_text, re.IGNORECASE | re.MULTILINE)
            if match:
                candidate = match.group(1).strip()
                # Avoid junk like "Sample", "Age"
                if candidate.lower() not in ['sample', 'age', 'sex']:
                    patient_info['name'] = candidate
                    break

        # ID - multiple patterns (including UHID)
        id_patterns = [
            r'(?:UHID|ID|MRN|Patient\s+ID)\s*:\s*([A-Za-z0-9\-]+)',
            r'ID\s*[:=]\s*([A-Za-z0-9\-]+)',
            r'ID:([A-Za-z0-9\-]+)'
        ]
        for pattern in id_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                patient_info['patient_id'] = match.group(1).strip()
                break

        # Age
        age_match = re.search(r'Age\s*[:=]\s*(\d+)', combined_text, re.IGNORECASE)
        if age_match:
            patient_info['age'] = int(age_match.group(1))

        # Sex/Gender
        sex_match = re.search(r'(?:Sex|Gender)\s*[:=]\s*(Male|Female|M|F)', combined_text, re.IGNORECASE)
        if sex_match:
            patient_info['sex'] = 'Male' if sex_match.group(1).upper() in ['M', 'MALE'] else 'Female'

        return patient_info

    def extract_clinical_history_ml(self, history_sentences: List[str]) -> str:
        """Extract clinical history"""
        if not history_sentences:
            return ""
        history = ' '.join(history_sentences)
        history = re.sub(
            r'(?:clinical history|indication|reason for|chief complaint|why here)[:\s]*',
            '',
            history,
            flags=re.IGNORECASE
        )
        return history.strip()

    def extract_findings_ml(self, finding_sentences: List[str]) -> str:
        """Extract findings"""
        if not finding_sentences:
            return ""
        findings = ' '.join(finding_sentences)
        findings = re.sub(
            r'(?:findings|results|what we found|observations)[:\s]*',
            '',
            findings,
            flags=re.IGNORECASE
        )
        return findings.strip()

    def extract_diagnosis_ml(self, diagnosis_sentences: List[str]) -> str:
        """Extract diagnosis"""
        if not diagnosis_sentences:
            return ""
        diagnosis = ' '.join(diagnosis_sentences)
        diagnosis = re.sub(
            r'(?:impression|conclusion|interpretation|doctor says|summary|diagnosis)[:\s]*',
            '',
            diagnosis,
            flags=re.IGNORECASE
        )
        return diagnosis.strip()

    def extract_recommendations_ml(self, rec_sentences: List[str]) -> str:
        """Extract recommendations"""
        if not rec_sentences:
            return ""
        recommendations = ' '.join(rec_sentences)
        recommendations = re.sub(
            r'(?:recommendation|follow-up|next steps|do this|treatment)[:\s]*',
            '',
            recommendations,
            flags=re.IGNORECASE
        )
        return recommendations.strip()

    def parse(self, raw_text: str) -> Dict:
        """Master parsing function using hybrid ML + rules"""
        # STEP 1: Classify sentences
        classified_sentences = self.classifier.classify_text(raw_text)

        # STEP 2: Group by intent
        sections = self.extract_sections_by_intent(classified_sentences)

        # STEP 3: Extract structured data
        structured_report = {
            'patient_info': self.extract_patient_info_ml(sections.get('patient_info', [])),
            'examination': self._detect_exam_type(raw_text),
            'clinical_history': self.extract_clinical_history_ml(sections.get('clinical_history', [])),
            'findings': self.extract_findings_ml(sections.get('findings', [])),
            'diagnosis': self.extract_diagnosis_ml(sections.get('diagnosis', [])),
            'recommendations': self.extract_recommendations_ml(sections.get('recommendation', [])),
            # Always provide test_results key (empty by default here)
            'test_results': {},
            'metadata': {
                'parser_version': 'v5.2_hybrid_ml_table_aware',
                'total_sentences_classified': len(classified_sentences),
                'intents_detected': list(sections.keys())
            }
        }

        return structured_report

    def _detect_exam_type(self, raw_text: str) -> Dict:
        """Detect exam type"""
        exam_info = {}
        exam_keywords = {
            'xray': ['chest x-ray', 'x-ray', 'radiograph', 'cxr', 'pa view', 'lateral view'],
            'mri': ['mri', 'magnetic resonance', 'mri brain', 'mri spine'],
            'ct': ['ct scan', 'computed tomography', 'mdct', 'helical ct', 'ct head'],
            'ultrasound': ['ultrasound', 'usg', 'echo', 'b-mode'],
            'ecg': ['electrocardiogram', 'ecg', 'ekg'],
            'lab': [
                'biochemistry', 'blood test', 'panel', 'serology', 'urinalysis',
                'cbc', 'complete blood count', 'pathology', 'pathology lab'
            ],
        }

        text_lower = raw_text.lower()
        for exam_type, keywords in exam_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    exam_info['exam_type'] = exam_type
                    break
            if 'exam_type' in exam_info:
                break

        # Try to extract exam name
        exam_pattern = r'Examination\s*:\s*([^\n]+)'
        match = re.search(exam_pattern, raw_text, re.IGNORECASE)
        if match:
            exam_info['exam_name'] = match.group(1).strip()

        return exam_info


class LabReportParser(HybridMedicalReportParser):
    """
    Specialized parser for lab/biochemistry/serology/pathology reports
    Inherits from HybridMedicalReportParser and adds test result extraction
    """

    def extract_test_results(self) -> Dict[str, List[Dict]]:
        """
        Extract structured test results from lab reports (from text lines)
        Handles: value, unit, normal range, abnormality flags
        """
        results_by_category = {}
        current_category = None

        lines = self.raw_text.split('\n')

        # Categories to skip (junk/metadata)
        junk_categories = ['TEST RESULTS', 'IMPRESSION', 'RECOMMENDATION', 'REPORT', 'FINDINGS', 'RESULTS']

        # Lines to skip completely (metadata)
        skip_patterns = ['Generated on', 'Reported by', 'Authorized by', 'Signed', 'Date:', 'Email:']

        for line in lines:
            if not line.strip():
                continue

            # Skip metadata lines
            if any(skip_keyword in line for skip_keyword in skip_patterns):
                continue

            # Detect category headers (ALL CAPS line ending with colon)
            if re.match(r'^[A-Z][A-Z\s]+:\s*$', line.strip()):
                potential_category = line.strip().rstrip(':').strip()

                # Only add if not in junk list
                if potential_category not in junk_categories:
                    current_category = potential_category
                    results_by_category[current_category] = []
                continue

            # Parse test lines only in valid categories
            if current_category:
                # Test line: "Test Name: value unit (Normal: range) [Flag]"
                test_pattern = r'^([A-Za-z\s\(\)]+?):\s*(.+?)(?=\s+\(Normal|$)'
                match = re.match(test_pattern, line.strip())

                if match:
                    test_name = match.group(1).strip()
                    test_value_str = match.group(2).strip()

                    # Parse the value details
                    test_entry = self._parse_test_value(test_value_str, line)

                    if test_entry:
                        results_by_category[current_category].append({
                            'test_name': test_name,
                            **test_entry
                        })

        return results_by_category

    def _parse_test_value(self, value_str: str, full_line: str) -> Optional[Dict]:
        """
        Parse test value string into structured components
        Handles compound units like g/dL, x 10^9/L, %
        """
        test_data = {}

        # Extract numeric VALUE (handles decimals and ratios like 120/80)
        value_pattern = r'(\d+\.?\d*(?:/\d+\.?\d*)?)'
        value_match = re.search(value_pattern, value_str)

        if value_match:
            test_data['value'] = value_match.group(1)
        else:
            return None  # No numeric value found

        # Extract UNIT (between value and opening paren or end of string)
        # Handles: g/dL, x 10^9/L, %, U/L, mEq/L, mg/dL, etc.
        unit_pattern = r'(\d+\.?\d*(?:/\d+\.?\d*)?)\s*([\w\s\^\/\-]+?)(?:\s*\(|$)'
        unit_match = re.search(unit_pattern, value_str)

        if unit_match:
            unit = unit_match.group(2).strip()
            # Clean up and normalize whitespace
            unit = ' '.join(unit.split()) if unit else ''
            test_data['unit'] = unit
        else:
            test_data['unit'] = ''

        # Extract NORMAL RANGE from parentheses
        range_pattern = r'\(Normal:\s*([\d\.\-\>\<\ ]+)\)'
        range_match = re.search(range_pattern, full_line, re.IGNORECASE)

        if range_match:
            test_data['normal_range'] = range_match.group(1).strip()
        else:
            test_data['normal_range'] = ''

        # Extract ABNORMALITY FLAG (L = Low, H = High) at end of line
        flag_pattern = r'\s([LH])\s*$'
        flag_match = re.search(flag_pattern, full_line.strip())

        if flag_match:
            flag = flag_match.group(1)
            test_data['abnormality_flag'] = 'Low' if flag == 'L' else 'High'
        else:
            test_data['abnormality_flag'] = 'Normal'

        return test_data

    def parse(self, raw_text: str, table_results: Optional[List[Dict]] = None) -> Dict:
        """
        Master parsing for lab reports.

        table_results: optional list of dicts from cleaned table, e.g.
        [
          {"test_name": "Hemoglobin", "value": 11.8, "unit": "g/dL", "reference_range": "12-16"},
          ...
        ]
        """
        self.raw_text = raw_text

        # Get base report structure from hybrid parser
        base_report = super().parse(raw_text)

        # Add lab-specific test results from text
        text_results = self.extract_test_results()

        # -------------------------------------------------------
        # LAB TABLE FALLBACK (TEXT → ROW RECONSTRUCTION)
        # -------------------------------------------------------

        final_table_rows = table_results if table_results else []

        exam_type = base_report.get("examination", {}).get("exam_type", "").lower()

        if exam_type == "lab" and not final_table_rows:
            reconstructed_rows = build_lab_rows_from_text(raw_text)

            if reconstructed_rows:
                final_table_rows = reconstructed_rows

        base_report["test_results"] = {
            "from_table": final_table_rows,
            "from_text": {}  # prevent numeric leakage into findings
        }
        return base_report


def parse_universal_report(
    raw_text: str,
    table_results: Optional[List[Dict]] = None
) -> Dict:
    """
    Convenience wrapper for UI / API.
    Automatically chooses LabReportParser for lab-type text,
    otherwise uses HybridMedicalReportParser.

    Also performs:
    - deterministic patient info extraction
    - lab table fallback reconstruction when needed
    - cleanup of numeric leakage from findings for lab reports
    """

    # -------------------------------
    # 1️⃣ Detect modality
    # -------------------------------
    text_lower = raw_text.lower()
    is_lab = any(k in text_lower for k in [
        "cbc", "complete blood count", "biochemistry",
        "serology", "urinalysis", "pathology", "blood test"
    ])

    # -------------------------------
    # 2️⃣ Run main parser
    # -------------------------------
    if is_lab:
        parser = LabReportParser()
        report = parser.parse(raw_text, table_results=table_results)
    else:
        parser = HybridMedicalReportParser()
        report = parser.parse(raw_text)
        report.setdefault("test_results", {})

    # -------------------------------
    # 3️⃣ ALWAYS extract patient info
    # -------------------------------
    patient_info = extract_patient_info(raw_text)

    report.setdefault("patient_info", {})
    report["patient_info"].update({
        k: v for k, v in patient_info.items()
        if k not in report["patient_info"]
    })

    # -------------------------------
    # 4️⃣ LAB TABLE FALLBACK (TEXT → ROWS)
    # -------------------------------
    if is_lab:
        test_results = report.setdefault("test_results", {})
        from_table = test_results.get("from_table", [])

        if not from_table:
            reconstructed_rows = build_lab_rows_from_text(raw_text)
            if reconstructed_rows:
                test_results["from_table"] = reconstructed_rows
                test_results["from_text"] = {}

        # -------------------------------
        # 5️⃣ CLEAN FINDINGS (IMPORTANT)
        # -------------------------------
        # Once structured lab rows exist, findings should be narrative only.
        # Prevent numeric duplication & LLM confusion.
        if test_results.get("from_table"):
            report["findings"] = ""

    return report