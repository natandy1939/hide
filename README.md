# Multilingual CV & Dossier De-identification Pipeline

An automated, high-precision de-identification pipeline designed to sanitize resumes (CVs), diplomas, attestations, and candidate dossiers across **English**, **French**, and **Arabic**.

The pipeline sanitizes both text and visuals with zero over-redaction, completely excising:
1. **Candidate Names** (via Multilingual Transformer NER `Davlan/bert-base-multilingual-cased-ner-hrl`)
2. **Profile Photos / Headshots** (via Computer Vision & MediaPipe Face Detection)
3. **Phone Numbers** (International, French `06/07`, and Moroccan `05/06/07` formats)
4. **Email Addresses** (Standard and multiline layouts)

---

## Key Features

- **Multilingual Transformer NER:** Integrates `Davlan/bert-base-multilingual-cased-ner-hrl` to discriminate person names (`PER`) while strictly preserving locations (`LOC`), organizations (`ORG`), degrees, and job titles.
- **Domain Stopword Guard:** Over 120+ professional terms (job titles, degrees, disciplines, institutions) are safeguarded against accidental redaction.
- **Contextual Recipient Matching:** Distinguishes candidates from signatories on attestations and certificates (`certifie que M. [Name]`, `décernée à [Name]`, `يشهد أن السيد [Name]`), preserving organization headers and director signatures.
- **Dossier Identity Propagation:** Extracts candidate identity from the primary CV and propagates it across supplementary files (certificates, attestations, images) within the candidate's folder.
- **Native PyMuPDF Redactions:** Employs `page.add_redact_annot()` and `page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)` to permanently wipe text streams, font encodings, and image pixels.
- **Full Metadata Erasure:** Strips PDF metadata (Author, Title, Producer, CreationDate).

---

## Installation

```bash
# Clone repository
git clone git@github.com:natandy1939/hide.git
cd hide

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Download spaCy language models if using fallback rules
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

---

## Usage

### 1. Batch Dossier Processing
To process candidate dossiers organized in folders (e.g. `test/<candidate_id>/...`):

```bash
python batch_process.py
```
Output files will be saved in `result/<candidate_id>/...` along with a comprehensive audit report in `result/redaction_report.json`.

### 2. Single Document Redaction
```python
from redactor import CVRedactor

redactor = CVRedactor()

# Redact a PDF document
summary = redactor.redact_file("input_cv.pdf", "output_cv.pdf")
print("Redaction Summary:", summary)
```

### 3. Run Pipeline Tests
```bash
python test_pipeline.py
```

---

## Project Structure

```
├── redactor.py         # Core redaction engine (Transformer NER + PyMuPDF native redaction)
├── batch_process.py    # Batch dossier processor & identity propagation
├── face_detector.py    # MediaPipe face & profile photo locator
├── config.py           # NLP analyzer configuration and language detection
├── test_pipeline.py    # Synthetic CV generator & pipeline verification suite
├── requirements.txt    # Project dependencies
└── .gitignore          # Git exclusion rules
```

---

## License
MIT
