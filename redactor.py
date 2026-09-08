import os
import io
import re
import unicodedata
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw

from config import create_analyzer_engine, detect_language
from face_detector import ProfilePhotoDetector

logger = logging.getLogger(__name__)

# Comprehensive domain stopwords: Terms in resumes/certificates that must NEVER be redacted as names
RESUME_STOPWORDS: Set[str] = {
    # Job Titles & Professional Designations
    'professeur', 'enseignant', 'enseignante', 'formateur', 'formatrice',
    'ingenieur', 'ingénieur', 'architecte', 'technicien', 'technicienne',
    'developpeur', 'développeur', 'directeur', 'directrice', 'consultant',
    'consultante', 'cadre', 'athlete', 'athlète', 'entraineur', 'entraîneur',
    'manager', 'leader', 'docteur', 'doctor', 'assistant', 'assistante',
    'stagiaire', 'stagiere', 'responsable', 'chef', 'coordinateur', 'expert', 'analyste',
    'chercheur', 'chercheuse', 'specialiste', 'spécialiste', 'administrateur',
    'technique', 'commercial', 'commerciale', 'ouvrier', 'agent', 'collaborateur',
    'inspecteur', 'educateur', 'éducateur', 'instituteur', 'institutrice',
    
    # Degrees, Diplomas, Certifications
    'doctorat', 'master', 'licence', 'baccalaureat', 'baccalauréat', 'bac',
    'deug', 'deup', 'these', 'thèse', 'certificat', 'diplome', 'diplôme', 'attestation',
    'formation', 'dut', 'bts', 'ingenierie', 'ingénierie', 'qualification',
    'scolarite', 'scolarité', 'etudes', 'études', 'mention', 'semestre',
    'bulletin', 'releve', 'relevé', 'notes', 'accreditation',
    
    # Academic Disciplines, Skills & Subject Matter
    'bioinformatique', 'biologie', 'informatique', 'mathematiques', 'mathématiques',
    'physique', 'chimie', 'droit', 'economie', 'économie', 'gestion', 'commerce',
    'marketing', 'algorithmics', 'genomique', 'génomique', 'genetique', 'génétique',
    'sport', 'eps', 'sante', 'santé', 'science', 'sciences', 'telecom', 'réseaux',
    'robotique', 'automatisme', 'mecanique', 'mécanique', 'electronique', 'électronique',
    'genie', 'génie', 'civil', 'electrique', 'électrique', 'industriel',
    
    # Institutions, Companies & Organizations
    'universite', 'université', 'faculte', 'faculté', 'institut', 'ecole', 'école',
    'ministere', 'ministère', 'academie', 'académie', 'centre', 'club', 'societe',
    'société', 'entreprise', 'groupe', 'ocp', 'ofppt', 'fus', 'lycee', 'lycée',
    'chambre', 'association', 'fondation', 'agence', 'banque', 'office',
    
    # Geographic Locations (Cities, Regions, Countries)
    'rabat', 'casablanca', 'marrakech', 'fes', 'fès', 'tanger', 'agadir',
    'kenitra', 'kénitra', 'temara', 'témara', 'sale', 'salé', 'maroc',
    'france', 'paris', 'tantan', 'agdal', 'souissi', 'birami', 'youssoufia',
    'oujda', 'meknes', 'meknès', 'tetouan', 'tétouan', 'nador', 'el jadida',
    'bouskoura', 'mohammedia', 'safi', 'settat', 'khouribga', 'beni mellal',
    
    # Resume Section Headers & Layout Keywords
    'profil', 'profile', 'contact', 'donnees', 'personnels', 'personnelles',
    'professional', 'summary', 'details', 'sommaire', 'resume', 'cv',
    'curriculum', 'vitae', 'experience', 'experiences',
    'competence', 'competences', 'compétence', 'compétences', 'langues',
    'centres', 'interet', 'intérêt', 'interets', 'intérêts', 'education',
    'parcours', 'professionnel', 'professionnelle', 'situation', 'marie',
    'marié', 'mariee', 'mariée', 'celibataire', 'célibataire', 'naissance', 'adresse', 'tel',
    'email', 'mail', 'linkedin', 'voyage', 'lecture', 'benevolat', 'bénévolat',
    'leadership', 'polyvalence', 'creativite', 'créativité', 'actif',
    'communication', 'francais', 'français', 'arabe', 'anglais', 'courant',
    'maternelle', 'intermediaire', 'intermédiaire', 'training', 'intermediate',
    'projets', 'realisations', 'réalisations', 'references', 'références',
    'stage', 'stages', 'emploi', 'missions', 'taches', 'tâches', 'loisirs',
    'qualites', 'qualités', 'esprit', 'equipe', 'équipe',
    
    # Common French/English Functional & Sentence Words
    'de', 'du', 'des', 'en', 'la', 'le', 'les', 'et', 'ma', 'mon', 'mes',
    'sur', 'pour', 'dans', 'avec', 'par', 'aux', 'au', 'un', 'une', 'd’un',
    'the', 'and', 'for', 'with', 'from', 'in', 'at', 'on', 'to', 'of',
    'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
    'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'votre', 'leur', 'leurs',
    'bon', 'bonne', 'bons', 'bonnes', 'suis', 'est', 'sont', 'avoir', 'etre', 'être',
    'faire', 'souhaite', 'doté', 'dotée', 'dynamique', 'sérieux', 'serieux', 'sérieuse',
    'rigoureux', 'rigoureuse', 'motivé', 'motivée', 'cherche', 'actuellement',
    'ans', 'domaine', 'secteur', 'sien', 'sein', 'relever', 'défi', 'defi',
    
    # Arabic Non-Name Resume Terms & Education Vocabulary
    'التدريس', 'التعليم', 'التربوي', 'المستوى', 'الباكالوريا', 'شهادة',
    'األهلية', 'الثانوي', 'التأهيلي', 'الممتازة', 'مادة', 'اللغة', 'العربية',
    'الفرنسية', 'اإلنجليزية', 'الخبرات', 'المهنية', 'المؤهلات', 'المهارات',
    'التقنية', 'إدارة', 'الخوادم', 'الحوسبة', 'السحابية', 'البرمجيات',
    'تطوير', 'قواعد', 'البيانات', 'مراكز', 'االهتمام', 'الدار', 'البيضاء',
    'المغرب', 'أستاذ', 'متوسط', 'جيد', 'األم', 'شعر', 'الجهوي', 'التخرج',
    'المركز', 'األدب', 'العصرية', 'الحديثة', 'العامة', 'الخاصة', 'ثانوية',
    'جامعة', 'كلية', 'معهد', 'مدرسة', 'أكاديمية', 'وزارة', 'مؤسسة',
    'دبلوم', 'إجازة', 'ماستر', 'دكتوراه', 'تدريب', 'سيرة', 'ذاتية',
    'معلومات', 'شخصية', 'الهاتف', 'البريد', 'العنوان', 'تاريخ', 'الميلاد',
    'الرباط', 'سلا', 'تمارة', 'القنيطرة', 'فاس', 'مراكش', 'طنجة', 'أكادير',
    'وجذة', 'مكناس', 'تطوان', 'الناظور', 'المحمدية', 'الجديدة', 'آسفي',
    'مغربية', 'مغربي', 'السن', 'سنة', 'الجنسية', 'الحالة', 'العائلية',
    'متزوج', 'عازب', 'أعزب', 'نبذة', 'عني', 'الهوايات', 'التخصص',
    'دكتور', 'مهندس', 'تقني', 'مدير', 'رئيس', 'مسؤول', 'مستشار'
}


def normalize_arabic(text: str) -> str:
    """
    Normalizes Arabic text:
    - Normalizes presentation forms to standard Unicode (NFKC)
    - Strips tatweel / kashida
    - Strips tashkeel (diacritics)
    - Normalizes alef variants (أ, إ, آ, ٱ -> ا)
    """
    if not text:
        return ""
    t = unicodedata.normalize('NFKC', text)
    t = re.sub(r'[\u0640]', '', t)
    t = re.sub(r'[\u064B-\u065F\u0670]', '', t)
    t = re.sub(r'[إأآٱ]', 'ا', t)
    return t


def merge_consecutive_entities(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reassembles WordPiece tokens and merges consecutive same-type entities.
    E.g.: 'H' (PER) + '##AMZA AMRANI' (PER) -> 'HAMZA AMRANI' (PER)
    """
    merged = []
    for ent in ents:
        if not merged:
            merged.append(dict(ent))
            continue
        prev = merged[-1]
        word = ent.get('word', '')
        is_subword = word.startswith('##')
        is_adjacent = ent.get('start', 0) <= prev.get('end', 0) + 1
        if prev['entity_group'] == ent['entity_group'] and (is_subword or is_adjacent):
            if is_subword:
                prev['word'] = prev['word'] + word[2:]
            else:
                prev['word'] = prev['word'] + ' ' + word
            prev['end'] = ent.get('end', prev['end'])
            prev['score'] = min(prev['score'], ent.get('score', 1.0))
        else:
            merged.append(dict(ent))
    return merged


class CVRedactor:
    """
    Precision PDF & Document De-identification Pipeline for CVs and certificates.
    Focuses specifically on:
    1. Candidate Names (Multilingual transformer NER + multi-stage header discovery)
    2. Profile Photos / Headshots (Exact face box masking, preserving layout banners)
    3. Phone Numbers (Local and international regex)
    4. Email Addresses (Including multiline/spaced formats)
    """

    TARGET_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"]

    def __init__(
        self,
        enable_arabic_hf: bool = False,
        redact_fill_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        padding_pt: float = 2.0,
        scanned_char_threshold: int = 50,
    ):
        self.redact_fill_color = redact_fill_color
        self.padding_pt = padding_pt
        self.scanned_char_threshold = scanned_char_threshold

        logger.info("Initializing HuggingFace Multilingual Transformer NER (Davlan/bert-base-multilingual-cased-ner-hrl)...")
        try:
            from transformers import pipeline
            self.hf_ner = pipeline(
                "ner",
                model="Davlan/bert-base-multilingual-cased-ner-hrl",
                aggregation_strategy="simple",
                device=-1,  # CPU
            )
            logger.info("HuggingFace Multilingual Transformer NER successfully initialized.")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace NER pipeline: {e}")
            self.hf_ner = None

        logger.info("Initializing Presidio AnalyzerEngine fallback...")
        try:
            self.analyzer = create_analyzer_engine(enable_arabic_hf=enable_arabic_hf)
        except Exception as e:
            logger.warning(f"Presidio AnalyzerEngine initialization warning: {e}")
            self.analyzer = None

        logger.info("Initializing ProfilePhotoDetector...")
        self.face_detector = ProfilePhotoDetector()

        self._ocr_engine = None

    def _get_ocr_engine(self):
        """Lazy loader for RapidOCR."""
        if self._ocr_engine is None:
            try:
                from rapidocr import RapidOCR
                self._ocr_engine = RapidOCR()
                logger.info("RapidOCR initialized for scanned PDF and image fallback.")
            except Exception as e:
                logger.warning(f"RapidOCR initialization failed: {e}.")
                self._ocr_engine = "pytesseract"
        return self._ocr_engine

    def extract_names_with_transformer(self, text: str) -> Set[str]:
        """
        Extracts high-confidence Person entities using HuggingFace Multilingual Transformer NER.
        Reassembles subwords, filters PER score >= 0.80, and checks candidate capitalization rules.
        """
        names: Set[str] = set()
        if not self.hf_ner or not text.strip():
            return names

        try:
            lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= 3]
            for line in lines[:30]:
                if '@' in line or re.match(r'^[0-9+\s\-–/.]+$', line):
                    continue
                first_word = line.split()[0].lower()
                if first_word in {'formation', 'experiences', 'expérience', 'competences', 'compétence', 'langues'}:
                    continue

                raw_ents = self.hf_ner(line[:500])
                merged_ents = merge_consecutive_entities(raw_ents)

                for ent in merged_ents:
                    if ent.get('entity_group') == 'PER' and ent.get('score', 0) >= 0.80:
                        word = ent['word'].strip()
                        word = re.sub(r'##', '', word).strip()
                        word = re.sub(r'\s+', ' ', word)
                        clean_word = re.sub(r'[^\w\s\u0600-\u06FF\uFB50-\uFEFF\-\.]', '', word).strip()
                        tokens = clean_word.split()

                        # Must have 2 to 4 words or be a distinct Arabic multi-token name
                        is_arabic = any('\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFEFF' for c in clean_word)
                        min_tokens = 1 if is_arabic else 2

                        if min_tokens <= len(tokens) <= 4 and len(clean_word) >= 5:
                            if not any(t.lower() in RESUME_STOPWORDS for t in tokens):
                                if is_arabic or clean_word.isupper() or all(t[0].isupper() for t in tokens if t):
                                    names.add(clean_word)
        except Exception as e:
            logger.debug(f"Transformer NER error in extract_names_with_transformer: {e}")

        return names

    def _extract_attestation_names(self, doc: fitz.Document) -> Set[str]:
        """
        Extracts candidate names from attestations, certificates, and diplomas by matching
        standard recipient phrases (certifie que M. X, décernée à Y, awarded to Z, etc.)
        while strictly excluding signatories and organizations.
        """
        names: Set[str] = set()
        if len(doc) == 0:
            return names

        attestation_patterns = [
            r'(?:atteste|certifie)\s+que\s+(?:M\.|Mme|Mlle|Monsieur|Madame)?\s*([A-ZÀ-Ÿ][a-zà-ÿA-ZÀ-Ÿ\s\-]+?)(?:\s*,|\s+a\s+|\s+est\s+|\s+n[eé]\s+|\s+titulaire|\s+matricule|\n|\r)',
            r'(?:d[eé]cern[eé]e?|accord[eé]e?|attribu[eé]e?|d[eé]livr[eé]e?|remis)\s+[aà]\s+(?:M\.|Mme|Mlle|Monsieur|Madame)?\s*([A-ZÀ-Ÿ][a-zà-ÿA-ZÀ-Ÿ\s\-]+?)(?:\s*,|\s+pour\s+|\s+a\s+|\s+le\s+|\n|\r)',
            r'(?:au\s+profit\s+de)\s+(?:M\.|Mme|Mlle|Monsieur|Madame)?\s*([A-ZÀ-Ÿ][a-zà-ÿA-ZÀ-Ÿ\s\-]+?)(?:\s*,|\s+pour\s+|\n|\r)',
            r'(?:certify\s+that|awarded\s+to|presented\s+to|conferred\s+upon)\s+(?:Mr\.|Ms\.|Mrs\.)?\s*([A-Z][a-zA-Z\s\-]+?)(?:\s*,|\s+for\s+|\s+has\s+|\n|\r)',
            r'(?:نشهد\s+أن|يشهد\s+أن)\s+(?:السيد|السيدة|الآنسة|الطالب|التلميذ)?\s*([^\n\r,]+?)(?:\s*,|\s+قد|\s+أنه|\n|\r)'
        ]

        page = doc[0]
        raw_text = page.get_text()
        norm_text = unicodedata.normalize('NFKC', raw_text).replace('\xa0', ' ')
        for p in attestation_patterns:
            m = re.search(p, norm_text, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                cand = re.sub(r'\s+', ' ', cand)
                cand = re.split(r'\s*[:\-–•|]\s*', cand)[0].strip()
                words = [w for w in cand.split() if w]
                if 2 <= len(words) <= 4 and not any(w.lower() in RESUME_STOPWORDS for w in words):
                    if not any(w.lower() in ['société', 'centre', 'ministère', 'université', 'école', 'institut'] for w in words):
                        names.add(' '.join(words))
                        for w in words:
                            if len(w) >= 4 and (w.isupper() or w[0].isupper()):
                                names.add(w)
        return names

    def _identify_candidate_names(self, doc: fitz.Document) -> Set[str]:
        """
        Extracts the candidate's actual name from the document using:
        1. Labeled patterns (Nom:, Name:, الاسم الكامل:, etc.)
        2. Attestation / Certificate recipient patterns (certifie que M. X, décernée à Y)
        3. Top 35% Page 1 header blocks analyzed with Davlan Transformer NER
        4. Prominent capital header strings
        5. RapidOCR analysis on embedded header image cards (for Canva/Word exported headers)
        6. Arabic text normalization and Arabic NER
        """
        names: Set[str] = set()
        if len(doc) == 0:
            return names

        page = doc[0]
        full_text = page.get_text()

        # 1. Check for labeled name field (Latin & Arabic, normalized)
        norm_label_text = re.sub(r'االسم', 'الاسم', full_text)
        label_match = re.search(
            r'(?:Nom(?:\s+et\s+pr[eé]nom)?|Name|الاسم(?:\s+الكامل)?|المرشح|صاحب\s+السيرة)\s*[:\-]\s*([^\n\r]+(?:\n[^\n\r:]+)?)',
            norm_label_text,
            re.IGNORECASE
        )
        if label_match:
            raw_cand = label_match.group(1).strip()
            # Replace internal newline with space
            candidate_str = re.sub(r'\s+', ' ', raw_cand)
            candidate_str = re.split(r'\s+[\-–•]\s+|\s*\|\s*|[:]', candidate_str)[0].strip()
            candidate_str = re.sub(r'[^\w\s\u0600-\u06FF\uFB50-\uFEFF\-\.]', '', candidate_str).strip()
            words = [w for w in re.split(r'[\s\-_]+', candidate_str) if w]
            if 1 <= len(words) <= 4:
                clean = ' '.join(words)
                if not any(w.lower() in RESUME_STOPWORDS for w in words):
                    names.add(clean)

        # 1b. Check for attestation recipient patterns
        att_names = self._extract_attestation_names(doc)
        names.update(att_names)

        # 2. Check header text blocks (top 35% of Page 1)
        header_blocks = [
            b for b in page.get_text('blocks')
            if b[1] < page.rect.height * 0.35 and b[4].strip()
        ]
        header_blocks.sort(key=lambda b: (b[1], b[0]))

        header_text = "\n".join(b[4] for b in header_blocks)

        # Run Transformer NER on header text
        ner_header_names = self.extract_names_with_transformer(header_text)
        for n in ner_header_names:
            names.add(n)

        # Also inspect header lines directly for prominent 2-4 word capitalized names
        for b in header_blocks:
            lines = [l.strip() for l in b[4].split('\n') if l.strip()]
            for line in lines:
                line_first = re.split(r'\s+[\-–•]\s+|\s*\|\s*', line)[0].strip()
                clean = re.sub(r'[^\w\s\u0600-\u06FF\uFB50-\uFEFF\-\.]', '', line_first).strip()
                words = clean.split()
                if 2 <= len(words) <= 4 and len(clean) >= 6:
                    lower_words = [w.lower() for w in words]
                    if not any(w in RESUME_STOPWORDS for w in lower_words):
                        is_case_valid = clean.isupper() or all(w[0].isupper() for w in words if w)
                        if is_case_valid and re.match(r'^[A-Za-z\s\-\.\'\u0600-\u06FF\uFB50-\uFEFF]+$', clean):
                            names.add(clean)
                            break

        # 3. Check for Arabic candidate names across Page 1
        if any('\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFEFF' for c in full_text):
            norm_full = normalize_arabic(full_text)
            ar_ner_names = self.extract_names_with_transformer(norm_full[:1200])
            for n in ar_ner_names:
                norm_n = normalize_arabic(n)
                words = [w for w in norm_n.split() if len(w) >= 3 and w not in RESUME_STOPWORDS]
                if 1 <= len(words) <= 4:
                    names.add(n)
                    names.add(norm_n)

        # 4. Fallback for embedded header images (e.g. Canva or Word CVs where header is an image)
        if not names and len(full_text.strip()) < 80:
            try:
                for img_info in page.get_images()[:4]:
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    for r in rects:
                        if r.y0 < page.rect.height * 0.40 and r.width > 80 and r.height > 30:
                            extracted = doc.extract_image(xref)
                            pil_img = Image.open(io.BytesIO(extracted["image"]))
                            ocr_engine = self._get_ocr_engine()
                            if ocr_engine != "pytesseract":
                                res = ocr_engine(pil_img)
                                txt_list = []
                                if hasattr(res, 'txts') and res.txts:
                                    txt_list = list(res.txts)
                                elif isinstance(res, (list, tuple)) and res and res[0]:
                                    txt_list = [item[1] for item in res[0]]
                                
                                ocr_text = "\n".join(txt_list)
                                img_names = self.extract_names_with_transformer(ocr_text)
                                for n in img_names:
                                    names.add(n)
            except Exception as e:
                logger.debug(f"Error checking embedded header images in candidate identification: {e}")

        # Expand candidate names with constituent tokens
        expanded_names: Set[str] = set()
        for name in names:
            expanded_names.add(name)
            # Add Latin constituent parts (len >= 4, uppercase or title case)
            parts = [p.strip() for p in name.split() if len(p.strip()) >= 4]
            for p in parts:
                if p.lower() not in RESUME_STOPWORDS and not re.match(r'^\d+$', p):
                    if p.isupper() or p[0].isupper():
                        expanded_names.add(p)
            # Add Arabic constituent parts and composite variations
            if any('\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFEFF' for c in name):
                norm_name = normalize_arabic(name)
                expanded_names.add(norm_name)
                if 'عبد ' in norm_name:
                    expanded_names.add(norm_name.replace('عبد ', 'عبد'))
                elif 'عبد' in norm_name:
                    expanded_names.add(re.sub(r'عبد(\w+)', r'عبد \1', norm_name))
                for w in norm_name.split():
                    if len(w) >= 3 and w not in RESUME_STOPWORDS:
                        expanded_names.add(w)

        return expanded_names

    def _redact_searchable_page(
        self,
        page: fitz.Page,
        page_idx: int,
        candidate_names: Set[str],
        summary: Dict[str, Any],
    ) -> None:
        """
        Applies targeted redactions to candidate names, phones, emails, and faces on a page.
        """
        raw_text = page.get_text()
        if not raw_text.strip():
            # Check for faces even on pages with little or no text
            face_rects = self.face_detector.detect_faces_on_page(page, dpi=150)
            for face_rect in face_rects:
                page.add_redact_annot(face_rect, fill=self.redact_fill_color)
                summary["entities_found"]["PROFILE_PHOTO"] += 1
                summary["total_redactions"] += 1
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
            return

        detected_lang = detect_language(raw_text)
        if detected_lang not in summary["languages_detected"]:
            summary["languages_detected"].append(detected_lang)

        redaction_rects: List[Tuple[fitz.Rect, str]] = []

        # 1. Redact Candidate Names (Exact match and constituent parts)
        for name in candidate_names:
            if not name.strip() or len(name.strip()) < 3:
                continue
            for r in page.search_for(name):
                pad_r = fitz.Rect(
                    r.x0 - self.padding_pt,
                    r.y0 - self.padding_pt,
                    r.x1 + self.padding_pt,
                    r.y1 + self.padding_pt
                )
                redaction_rects.append((pad_r, "PERSON"))
                summary["entities_found"]["PERSON"] += 1

        # Arabic presentation form matching via words table
        if any(any('\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFEFF' for c in n) for n in candidate_names):
            norm_target_names = {normalize_arabic(n) for n in candidate_names if any('\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFEFF' for c in n)}
            page_words = page.get_text('words')
            for w in page_words:
                norm_w = normalize_arabic(w[4])
                if norm_w and len(norm_w) >= 3 and norm_w not in RESUME_STOPWORDS:
                    for target in norm_target_names:
                        if norm_w == target or (len(norm_w) >= 4 and norm_w in target):
                            pad_r = fitz.Rect(
                                w[0] - self.padding_pt,
                                w[1] - self.padding_pt,
                                w[2] + self.padding_pt,
                                w[3] + self.padding_pt
                            )
                            redaction_rects.append((pad_r, "PERSON"))
                            summary["entities_found"]["PERSON"] += 1
                            break

        # 2. Redact Phone Numbers via Regex
        # Matches international (+XX...), French (06..., 07...), and Moroccan (05..., 06..., 07...)
        phone_pattern = re.compile(
            r'(?<!\d)(?:(?:\+|00)(?:33|212)[\s.-]?(?:\(0\)[\s.-]?)?|0)[1-7](?:[\s.-]?\d{2}){4}(?!\d)|'
            r'(?<!\d)(?:\+|00)[1-9]\d{0,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?)?\d{2,4}[\s.-]?\d{2,4}(?!\d)'
        )
        for m in phone_pattern.finditer(raw_text):
            phone_str = m.group()
            if len(re.findall(r'\d', phone_str)) >= 8:
                for r in page.search_for(phone_str):
                    pad_r = fitz.Rect(
                        r.x0 - self.padding_pt,
                        r.y0 - self.padding_pt,
                        r.x1 + self.padding_pt,
                        r.y1 + self.padding_pt
                    )
                    redaction_rects.append((pad_r, "PHONE_NUMBER"))
                    summary["entities_found"]["PHONE_NUMBER"] += 1

        # 3. Redact Email Addresses (Handles standard and multiline/spaced email layouts)
        email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+(?:\s*@\s*|\s*\[at\]\s*)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        )
        for m in email_pattern.finditer(raw_text):
            email_str = m.group()
            parts = re.split(r'\s+', email_str)
            for part in parts:
                if len(part) >= 3:
                    for r in page.search_for(part):
                        pad_r = fitz.Rect(
                            r.x0 - self.padding_pt,
                            r.y0 - self.padding_pt,
                            r.x1 + self.padding_pt,
                            r.y1 + self.padding_pt
                        )
                        redaction_rects.append((pad_r, "EMAIL_ADDRESS"))
                        summary["entities_found"]["EMAIL_ADDRESS"] += 1

        # 4. Profile Photo & Face Detection
        face_rects = self.face_detector.detect_faces_on_page(page, dpi=150)
        for face_rect in face_rects:
            redaction_rects.append((face_rect, "PROFILE_PHOTO"))
            summary["entities_found"]["PROFILE_PHOTO"] += 1

        # 5. Apply native redaction annotations
        applied_rects: List[fitz.Rect] = []
        for rect, etype in redaction_rects:
            if any(rect in app for app in applied_rects):
                continue
            applied_rects.append(rect)
            page.add_redact_annot(rect, fill=self.redact_fill_color)
            summary["total_redactions"] += 1

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)

    def _process_scanned_doc(
        self,
        doc: fitz.Document,
        summary: Dict[str, Any],
        candidate_names: Optional[Set[str]] = None,
    ) -> fitz.Document:
        """
        High-resolution 300 DPI OCR pipeline for scanned CVs.
        """
        sanitized_doc = fitz.open()
        ocr_engine = self._get_ocr_engine()

        for page_idx, page in enumerate(doc):
            dpi = 300
            pix = page.get_pixmap(dpi=dpi)
            img_pil = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
            img_np = np.array(img_pil)

            draw = ImageDraw.Draw(img_pil)

            # 1. Face detection
            face_boxes = self.face_detector.detect_faces_in_numpy(img_np, is_rgb=True)
            for (fx, fy, fw, fh) in face_boxes:
                pad_w = int(fw * 0.20)
                pad_h = int(fh * 0.25)
                box = [
                    max(0, fx - pad_w),
                    max(0, fy - pad_h),
                    min(pix.w, fx + fw + pad_w),
                    min(pix.h, fy + fh + pad_h),
                ]
                draw.rectangle(box, fill=(0, 0, 0))
                summary["entities_found"]["PROFILE_PHOTO"] += 1
                summary["total_redactions"] += 1

            # 2. OCR text extraction
            ocr_boxes: List[Dict[str, Any]] = []
            full_ocr_text = ""

            if ocr_engine != "pytesseract":
                try:
                    results = ocr_engine(img_np)
                    raw_items = []
                    if hasattr(results, 'boxes') and hasattr(results, 'txts') and results.boxes is not None:
                        for b_pts, txt in zip(results.boxes, results.txts):
                            raw_items.append((b_pts, txt))
                    elif isinstance(results, (list, tuple)) and results and results[0]:
                        for item in results[0]:
                            raw_items.append((item[0], item[1]))

                    for box_pts, word_text in raw_items:
                        x0 = int(min(p[0] for p in box_pts))
                        y0 = int(min(p[1] for p in box_pts))
                        x1 = int(max(p[0] for p in box_pts))
                        y1 = int(max(p[1] for p in box_pts))

                        start_char = len(full_ocr_text)
                        full_ocr_text += word_text + "\n"
                        end_char = len(full_ocr_text) - 1

                        ocr_boxes.append({
                            "start": start_char,
                            "end": end_char,
                            "bbox": (x0, y0, x1, y1),
                            "text": word_text,
                        })
                except Exception as e:
                    logger.warning(f"RapidOCR error on scanned page: {e}")

            # 3. Identify Candidate Name, Phones, Emails from OCR
            active_names = set(candidate_names) if candidate_names else set()
            if full_ocr_text.strip():
                if not active_names:
                    lines = [l.strip() for l in full_ocr_text.split('\n') if l.strip()]
                    for line in lines[:5]:
                        words = line.split()
                        if 2 <= len(words) <= 4:
                            if not any(w.lower() in RESUME_STOPWORDS for w in words):
                                active_names.add(line)
                                for w in words:
                                    if len(w) >= 4 and w.lower() not in RESUME_STOPWORDS:
                                        active_names.add(w)
                                break

                for ob in ocr_boxes:
                    txt = ob["text"].strip()
                    x0, y0, x1, y1 = ob["bbox"]

                    # Mask candidate name
                    if active_names and any(part.lower() in txt.lower() for part in active_names if len(part) >= 4):
                        if txt.lower() not in RESUME_STOPWORDS:
                            draw.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], fill=(0, 0, 0))
                            summary["entities_found"]["PERSON"] += 1
                            summary["total_redactions"] += 1
                            continue

                    # Mask Phone
                    if re.search(r'(?:\+|00|0)[5-7]\d{8}', re.sub(r'[\s.-]', '', txt)):
                        draw.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], fill=(0, 0, 0))
                        summary["entities_found"]["PHONE_NUMBER"] += 1
                        summary["total_redactions"] += 1
                        continue

                    # Mask Email
                    if '@' in txt and '.' in txt:
                        draw.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], fill=(0, 0, 0))
                        summary["entities_found"]["EMAIL_ADDRESS"] += 1
                        summary["total_redactions"] += 1

            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            new_page = sanitized_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=img_bytes.getvalue())

        return sanitized_doc

    def redact_pdf(
        self,
        input_path: str,
        output_path: str,
        candidate_names: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Redacts PII (candidate names, phones, emails, headshots) from a PDF.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        doc = fitz.open(input_path)
        summary: Dict[str, Any] = {
            "input_path": input_path,
            "output_path": output_path,
            "pages_processed": len(doc),
            "total_redactions": 0,
            "entities_found": {
                "PERSON": 0,
                "PHONE_NUMBER": 0,
                "EMAIL_ADDRESS": 0,
                "PROFILE_PHOTO": 0,
            },
            "languages_detected": [],
            "is_scanned": False,
        }

        total_text_chars = sum(len(page.get_text().strip()) for page in doc)
        doc_is_scanned = total_text_chars < (self.scanned_char_threshold * len(doc))
        summary["is_scanned"] = doc_is_scanned

        if candidate_names:
            names_to_use = set(candidate_names)
        else:
            names_to_use = self._identify_candidate_names(doc)

        if doc_is_scanned:
            logger.info(f"Scanned document detected for {os.path.basename(input_path)}. Routing through 300 DPI OCR pipeline.")
            doc = self._process_scanned_doc(doc, summary, candidate_names=names_to_use)
        else:
            logger.info(f"Search-based PDF detected for {os.path.basename(input_path)}. Target names: {names_to_use}")
            for page_idx, page in enumerate(doc):
                self._redact_searchable_page(page, page_idx, names_to_use, summary)

        # Sanitize document metadata permanently
        doc.set_metadata({})

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        doc.save(
            output_path,
            garbage=4,
            deflate=True,
            clean=True,
        )
        doc.close()

        logger.info(f"Sanitized PDF saved to {output_path}. Summary: {summary}")
        return summary

    def redact_image(
        self,
        input_path: str,
        output_path: str,
        candidate_names: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Redacts standalone images (profile photos, scanned certificates).
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Image not found: {input_path}")

        summary: Dict[str, Any] = {
            "input_path": input_path,
            "output_path": output_path,
            "total_redactions": 0,
            "entities_found": {
                "PERSON": 0,
                "PHONE_NUMBER": 0,
                "EMAIL_ADDRESS": 0,
                "PROFILE_PHOTO": 0,
            },
            "is_image": True,
        }

        pil_img = Image.open(input_path).convert("RGB")
        img_np = np.array(pil_img)
        w_img, h_img = pil_img.size
        draw = ImageDraw.Draw(pil_img)

        # 1. Face detection
        face_boxes = self.face_detector.detect_faces_in_numpy(img_np, is_rgb=True)
        for (fx, fy, fw, fh) in face_boxes:
            pad_w = int(fw * 0.20)
            pad_h = int(fh * 0.25)
            box = [
                max(0, fx - pad_w),
                max(0, fy - pad_h),
                min(w_img, fx + fw + pad_w),
                min(h_img, fy + fh + pad_h),
            ]
            draw.rectangle(box, fill=(0, 0, 0))
            summary["entities_found"]["PROFILE_PHOTO"] += 1
            summary["total_redactions"] += 1

        # 2. OCR for text in certificates / documents
        ocr_engine = self._get_ocr_engine()
        if ocr_engine != "pytesseract":
            try:
                results = ocr_engine(img_np)
                raw_items = []
                if hasattr(results, 'boxes') and hasattr(results, 'txts') and results.boxes is not None:
                    for b_pts, txt in zip(results.boxes, results.txts):
                        raw_items.append((b_pts, txt))
                elif isinstance(results, (list, tuple)) and results and results[0]:
                    for item in results[0]:
                        raw_items.append((item[0], item[1]))

                active_names = set(candidate_names) if candidate_names else set()
                for box_pts, word_text in raw_items:
                    txt = word_text.strip()
                    x0 = int(min(p[0] for p in box_pts))
                    y0 = int(min(p[1] for p in box_pts))
                    x1 = int(max(p[0] for p in box_pts))
                    y1 = int(max(p[1] for p in box_pts))

                    # Check for candidate name
                    if active_names and any(part.lower() in txt.lower() for part in active_names if len(part) >= 4):
                        if txt.lower() not in RESUME_STOPWORDS:
                            draw.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], fill=(0, 0, 0))
                            summary["entities_found"]["PERSON"] += 1
                            summary["total_redactions"] += 1
                            continue

                    # Check for phone
                    if re.search(r'(?:\+|00|0)[5-7]\d{8}', re.sub(r'[\s.-]', '', txt)):
                        draw.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], fill=(0, 0, 0))
                        summary["entities_found"]["PHONE_NUMBER"] += 1
                        summary["total_redactions"] += 1
                        continue

                    # Check for email
                    if '@' in txt and '.' in txt:
                        draw.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], fill=(0, 0, 0))
                        summary["entities_found"]["EMAIL_ADDRESS"] += 1
                        summary["total_redactions"] += 1
            except Exception as e:
                logger.debug(f"OCR error on image: {e}")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        ext = os.path.splitext(output_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            pil_img.save(output_path, format="JPEG", quality=95)
        else:
            pil_img.save(output_path, format="PNG")

        return summary

    def redact_file(
        self,
        input_path: str,
        output_path: str,
        candidate_names: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Dispatches to redact_pdf or redact_image."""
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".pdf":
            return self.redact_pdf(input_path, output_path, candidate_names=candidate_names)
        elif ext in [".jpg", ".jpeg", ".png"]:
            return self.redact_image(input_path, output_path, candidate_names=candidate_names)
        else:
            import shutil
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return {"input_path": input_path, "output_path": output_path, "total_redactions": 0, "entities_found": {}}
