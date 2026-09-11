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

logger = logging.getLogger(__name__)

# Common Moroccan, French, and International given names that must NEVER be redacted as a single word alone
# (Prevents over-redacting "Université Mohammed V", "Lycée Hassan II", "Avenue Allal Ben Abdellah", etc.)
COMMON_GIVEN_NAMES: Set[str] = {
    # French / Moroccan / International given names (Latin)
    'mohammed', 'mohamed', 'mohammad', 'ahmed', 'youssef', 'yousef', 'ali', 'hassan', 'houssam',
    'amine', 'mehdi', 'hamza', 'omar', 'othman', 'khalid', 'tarik', 'tariq', 'anas',
    'karim', 'morad', 'mourad', 'mustapha', 'mostafa', 'adil', 'said', 'rachid', 'rachida',
    'fatima', 'khadija', 'salma', 'sara', 'sarah', 'iman', 'imane', 'meriem', 'meryem',
    'soukaina', 'asmaa', 'asma', 'chaimae', 'chaima', 'nouhaila', 'zineb', 'nassima',
    'laila', 'doha', 'soufyane', 'soufiane', 'rachad', 'madiha', 'yassine', 'samir',
    'abdellah', 'abdallah', 'abdelaziz', 'abdelkader', 'abderrahim', 'abdelali',
    'hicham', 'nabil', 'jamal', 'driss', 'idriss', 'faycal', 'faysal', 'anwar', 'anouar',
    'safaa', 'hanaa', 'wafa', 'wafaa', 'kaoutar', 'kawtar', 'ghita', 'loubna', 'bouchra',
    'mona', 'mouna', 'najlae', 'najwa', 'hanane', 'ihsane', 'mariam', 'marwa', 'hajar',
    'jean', 'pierre', 'michel', 'paul', 'marie', 'alain', 'philippe', 'nicolas',
    'alexandre', 'thomas', 'laurent', 'julien', 'david', 'antoine', 'stephane',
    'john', 'michael', 'david', 'james', 'robert', 'william', 'mary', 'sarah',
    # Arabic given names
    'محمد', 'أحمد', 'يوسف', 'علي', 'حسن', 'أمين', 'مهدي', 'حمزة', 'عمر', 'عثمان',
    'خالد', 'طارق', 'أنس', 'كريم', 'مراد', 'مصطفى', 'عادل', 'سعيد', 'رشيد', 'رشيدة',
    'فاطمة', 'خديجة', 'سلمى', 'سارة', 'إيمان', 'مريم', 'سكينة', 'أسماء', 'شيماء',
    'نهيلة', 'زينب', 'نسيمة', 'ليلى', 'ضحى', 'سفيان', 'رشاد', 'مديحة', 'ياسين', 'سمير',
    'عبد', 'عبدالله', 'عبد الله', 'عبدالعزيز', 'عبد العزيز', 'عبدالرحمن', 'عبد الرحمن',
    'عبدالرحيم', 'عبد الكريم', 'عبدالكريم', 'هشام', 'نبيل', 'جمال', 'إدريس', 'فيصل',
    'أنور', 'صفاء', 'هناء', 'وفاء', 'كوثر', 'غيثة', 'لبنى', 'بشرى', 'منى', 'نجلاء',
    'نجوى', 'حنان', 'إحسان', 'مروة', 'هاجر', 'الخامس', 'الأول', 'الثاني'
}

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
    'دكتور', 'مهندس', 'تقني', 'مدير', 'رئيس', 'مسؤول', 'مستشار',
    'الشواهد', 'شواهد', 'أكاديمية', 'األكاديمية', 'الأكاديمية', 'االكاديمية',
    'أكاديمي', 'األكاديمي', 'الأكاديمي', 'تجارب', 'تكوينات', 'تكوين', 'الشخصية'
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
                        clean_word = re.sub(r'^(?:المغرب|المملكة المغربية|الدار البيضاء|الرباط|سلا|فاس|طنجة|مكناس|أكادير|تطوان|وجدة|القنيطرة)\s*', '', clean_word)
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

    def _extract_single_candidate_name(self, doc: fitz.Document) -> Optional[str]:
        """
        Pinpoints the SINGLE primary candidate name on the CV using:
        1. Explicit labeled fields (Nom:, Name:, الاسم الكامل:)
        2. Visual font-size layout hierarchy (top 38% of page 1)
        3. Embedded graphic header card OCR (Canva/Word header cards)
        4. Scanned page OCR on the first non-blank page
        5. Davlan Multilingual Transformer NER
        """
        if len(doc) == 0:
            return None

        # Find first non-blank page
        target_page_idx = 0
        for i in range(min(3, len(doc))):
            p = doc[i]
            if len(p.get_text().strip()) > 30 or len(p.get_images()) > 0:
                target_page_idx = i
                break

        page = doc[target_page_idx]
        full_text = page.get_text()

        # 1. Explicit labeled patterns
        norm_label_text = unicodedata.normalize('NFKC', full_text).replace('\xa0', ' ')
        norm_label_text = re.sub(r'االسم', 'الاسم', norm_label_text)
        label_match = re.search(
            r'(?:Nom(?:\s+et\s+pr[eé]nom)?|Name|الاسم(?:\s+الكامل)?|المرشح|صاحب\s+السيرة)\s*[:\-]\s*([^\n\r]+(?:\n[^\n\r:]+)?)',
            norm_label_text,
            re.IGNORECASE
        )
        if label_match:
            raw_cand = label_match.group(1).strip()
            candidate_str = re.sub(r'\s+', ' ', raw_cand)
            candidate_str = re.split(r'\s+[\-–•]\s+|\s*\|\s*|[:]', candidate_str)[0].strip()
            candidate_str = re.sub(r'[^\w\s\u0600-\u06FF\uFB50-\uFEFF\-\.]', '', candidate_str).strip()
            words = [w for w in re.split(r'[\s\-_]+', candidate_str) if w]
            if 1 <= len(words) <= 4:
                clean = ' '.join(words)
                if not any(w.lower() in RESUME_STOPWORDS for w in words):
                    return clean

        # 2. Visual layout hierarchy: font size and top placement
        try:
            page_dict = page.get_text('dict')
            blocks = page_dict.get('blocks', [])
            spans_by_line = []
            for b in blocks:
                if 'lines' in b:
                    for l in b['lines']:
                        line_spans = []
                        for s in l['spans']:
                            st = s['text'].strip()
                            if len(st) >= 2 and not re.match(r'^[\d\s\W]+$', st):
                                if '@' not in st and not re.search(r'(?:\+?212|0[5-7])\d{8}', st):
                                    line_spans.append(s)
                        if line_spans:
                            combined = ' '.join(s['text'].strip() for s in line_spans)
                            max_sz = max(s['size'] for s in line_spans)
                            min_y0 = min(s['bbox'][1] for s in line_spans)
                            if min_y0 < page.rect.height * 0.38:
                                spans_by_line.append((max_sz, min_y0, combined))

            # Prioritize top 22% of page first (true header), then fallback to top 38%
            top_header_spans = [s for s in spans_by_line if s[1] < page.rect.height * 0.22]
            lower_header_spans = [s for s in spans_by_line if s[1] >= page.rect.height * 0.22]

            for group in [top_header_spans, lower_header_spans]:
                group.sort(key=lambda x: (-x[0], x[1]))
                for sz, y0, line_text in group:
                    clean = re.sub(r'\s+', ' ', line_text).strip()
                    clean = re.sub(r'^(?:M\.|Mme|Mlle|Monsieur|Madame|Dr\.?|Mr\.?)\s+', '', clean, flags=re.IGNORECASE)
                    clean = re.sub(r'^(?:المغرب|المملكة المغربية|الدار البيضاء|الرباط|سلا|فاس|طنجة|مكناس|أكادير|تطوان|وجدة|القنيطرة)\s*', '', clean)
                    words = [w for w in re.split(r'[\s\-]+', clean) if w]
                    if any(w.lower() in RESUME_STOPWORDS for w in words):
                        continue
                    if any(h in clean.upper() for h in ['CURRICULUM', 'VITAE', 'DONNEES', 'PERSONNELS', 'PROFIL', 'CONTACT', 'RESUME']):
                        continue
                    if 1 <= len(words) <= 4 and 4 <= len(clean) <= 40:
                        return clean
        except Exception as e:
            logger.debug(f"Visual layout hierarchy error: {e}")

        # 3. Embedded Graphic Header Card OCR (Word/Canva headers)
        try:
            for img_info in page.get_images():
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if rects and rects[0].y0 < page.rect.height * 0.38:
                    if rects[0].width > 100 and rects[0].height > 25:
                        pix = fitz.Pixmap(doc, xref)
                        img_pil = Image.open(io.BytesIO(pix.tobytes()))
                        ocr_engine = self._get_ocr_engine()
                        if ocr_engine != "pytesseract":
                            res = ocr_engine(np.array(img_pil))
                            txts = []
                            if hasattr(res, 'txts') and res.txts: txts = res.txts
                            elif isinstance(res, (list, tuple)) and res and res[0]: txts = [it[1] for it in res[0]]
                            for line in txts[:6]:
                                line_clean = line.strip()
                                line_clean = re.sub(r'^(?:M\.|Mme|Mlle|Monsieur|Madame|Dr\.?|Mr\.?)\s+', '', line_clean, flags=re.IGNORECASE)
                                words = [w for w in line_clean.split() if w]
                                if 1 <= len(words) <= 4 and 4 <= len(line_clean) <= 40:
                                    if not any(w.lower() in RESUME_STOPWORDS for w in words):
                                        if not any(c in line_clean for c in ['@', 'http', '.com']) and not re.search(r'\d{5,}', line_clean):
                                            return line_clean
        except Exception as e:
            logger.debug(f"Header card OCR error: {e}")

        # 4. Pure Scanned Page OCR Fallback
        if len(full_text.strip()) < 80:
            try:
                pix = page.get_pixmap(dpi=200)
                img_pil = Image.open(io.BytesIO(pix.tobytes()))
                ocr_engine = self._get_ocr_engine()
                if ocr_engine != "pytesseract":
                    res = ocr_engine(np.array(img_pil))
                    txts = []
                    if hasattr(res, 'txts') and res.txts: txts = res.txts
                    elif isinstance(res, (list, tuple)) and res and res[0]: txts = [it[1] for it in res[0]]
                    for line in txts[:6]:
                        line_clean = line.strip()
                        line_clean = re.sub(r'^(?:M\.|Mme|Mlle|Monsieur|Madame|Dr\.?|Mr\.?)\s+', '', line_clean, flags=re.IGNORECASE)
                        words = [w for w in line_clean.split() if w]
                        if 1 <= len(words) <= 4 and 4 <= len(line_clean) <= 40:
                            if not any(w.lower() in RESUME_STOPWORDS for w in words):
                                if not any(c in line_clean for c in ['@', 'http', '.com']) and not re.search(r'\d{5,}', line_clean):
                                    return line_clean
            except Exception as e:
                logger.debug(f"Scanned page OCR error: {e}")

        # 5. Fallback: Davlan Transformer NER on Header Text
        header_blocks = [
            b for b in page.get_text('blocks')
            if b[1] < page.rect.height * 0.38 and b[4].strip()
        ]
        header_blocks.sort(key=lambda b: (b[1], b[0]))
        header_text = "\n".join(b[4] for b in header_blocks)
        ner_names = self.extract_names_with_transformer(header_text)
        if ner_names:
            return sorted(list(ner_names), key=lambda x: -len(x))[0]

        return None

    def _generate_candidate_search_strings(self, candidate_name: str) -> Set[str]:
        """
        Generates targeted search strings for the single candidate name.
        Specifically ensures:
        - Full name is redacted in all case variations.
        - Inverted First/Last name is redacted.
        - Distinctive family name is redacted.
        - COMMON GIVEN NAMES (Mohamed, Ahmed, Ali, Hassan, etc.) are NEVER redacted alone.
        """
        targets: Set[str] = set()
        if not candidate_name:
            return targets

        clean = re.sub(r'\s+', ' ', candidate_name).strip()
        clean = re.sub(r'^(?:M\.|Mme|Mlle|Monsieur|Madame|Dr\.?|Mr\.?)\s+', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'^(?:المغرب|المملكة المغربية|الدار البيضاء|الرباط|سلا|فاس|طنجة|مكناس|أكادير|تطوان|وجدة|القنيطرة)\s*', '', clean)
        if len(clean) < 3:
            return targets

        # Full name variations
        targets.add(clean)
        targets.add(clean.upper())
        targets.add(clean.title())
        targets.add(clean.lower())

        words = [w for w in re.split(r'[\s\-]+', clean) if w]
        if len(words) == 2:
            targets.add(f"{words[1]} {words[0]}")
            targets.add(f"{words[1].upper()} {words[0].upper()}")
            targets.add(f"{words[1].title()} {words[0].title()}")

        # Add family name ONLY if length >= 5 and NOT a common given name and NOT a stopword
        for w in words:
            w_low = w.lower()
            if len(w) >= 5 and w_low not in COMMON_GIVEN_NAMES and w_low not in RESUME_STOPWORDS:
                targets.add(w)
                targets.add(w.upper())
                targets.add(w.title())

        # Arabic normalization forms
        if any('\u0600' <= c <= '\u06FF' for c in clean):
            norm = normalize_arabic(clean)
            targets.add(norm)
            if 'عبد ' in norm:
                targets.add(norm.replace('عبد ', 'عبد'))
            elif 'عبد' in norm:
                targets.add(re.sub(r'عبد(\w+)', r'عبد \1', norm))
            for w in norm.split():
                if len(w) >= 4 and w not in COMMON_GIVEN_NAMES and w not in RESUME_STOPWORDS:
                    targets.add(w)

        return targets

    def _identify_candidate_names(self, doc: fitz.Document) -> Set[str]:
        """Identifies the single candidate name and generates its search strings."""
        cand = self._extract_single_candidate_name(doc)
        targets: Set[str] = set()
        if cand:
            targets.update(self._generate_candidate_search_strings(cand))

        # Also include any attestation recipient name
        att_names = self._extract_attestation_names(doc)
        for aname in att_names:
            targets.update(self._generate_candidate_search_strings(aname))

        return targets

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
                if norm_w and len(norm_w) >= 4 and norm_w not in RESUME_STOPWORDS and norm_w not in COMMON_GIVEN_NAMES:
                    for target in norm_target_names:
                        if norm_w == target or (len(norm_w) >= 5 and norm_w in target):
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
            r'(?<!\d)(?:\+|00)[1-9]\d{0,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?)?\d{2,4}[\s.-]?\d{2,4}(?!\d)|'
            r'(?<!\d)[567]\d{8}(?!\d)'
        )
        for m in phone_pattern.finditer(raw_text):
            phone_str = m.group()
            if len(re.findall(r'\d', phone_str)) >= 8:
                for r in page.search_for(phone_str):
                    pad_x0 = r.x0 - self.padding_pt
                    # Check if there is an adjacent prefix like +212 or 212 or + immediately before r
                    prefix_rect = fitz.Rect(max(0, r.x0 - 45), r.y0 - 2, r.x0, r.y1 + 2)
                    prefix_text = page.get_text("text", clip=prefix_rect).strip()
                    if any(p in prefix_text for p in ["212", "+"]):
                        for match_pref in ["+212", "212", "+"]:
                            pref_rects = page.search_for(match_pref, clip=prefix_rect)
                            if pref_rects:
                                pad_x0 = min(pad_x0, min(pr.x0 for pr in pref_rects) - self.padding_pt)
                    pad_r = fitz.Rect(
                        pad_x0,
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

        # 3b. Redact embedded graphic header cards on Page 0 (e.g., Canva/Word headers)
        if page_idx == 0:
            for img_info in page.get_images():
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if rects and rects[0].y0 < page.rect.height * 0.40 and rects[0].width > 80 and rects[0].height > 25:
                    img_rect = rects[0]
                    try:
                        pix = fitz.Pixmap(page.parent, xref)
                        pil_img = Image.open(io.BytesIO(pix.tobytes()))
                        ocr_engine = self._get_ocr_engine()
                        if ocr_engine != "pytesseract":
                            res = ocr_engine(np.array(pil_img))
                            raw_items = []
                            if hasattr(res, 'boxes') and hasattr(res, 'txts') and res.boxes is not None:
                                for b_pts, txt in zip(res.boxes, res.txts):
                                    raw_items.append((b_pts, txt))
                            elif isinstance(res, (list, tuple)) and res and res[0]:
                                for item in res[0]:
                                    raw_items.append((item[0], item[1]))

                            scale_x = img_rect.width / pix.w
                            scale_y = img_rect.height / pix.h
                            for b_pts, txt in raw_items:
                                is_match = False
                                etype = ""
                                if candidate_names and any(part.lower() in txt.lower() for part in candidate_names if len(part) >= 4):
                                    if txt.lower() not in RESUME_STOPWORDS and txt.lower() not in COMMON_GIVEN_NAMES:
                                        is_match = True
                                        etype = "PERSON"
                                elif re.search(r'(?:\+|00|0)?[5-7]\d{8}', re.sub(r'[\s.-]', '', txt)):
                                    is_match = True
                                    etype = "PHONE_NUMBER"
                                elif '@' in txt and '.' in txt:
                                    is_match = True
                                    etype = "EMAIL_ADDRESS"

                                if is_match:
                                    bx0 = min(pt[0] for pt in b_pts)
                                    by0 = min(pt[1] for pt in b_pts)
                                    bx1 = max(pt[0] for pt in b_pts)
                                    by1 = max(pt[1] for pt in b_pts)
                                    px0 = img_rect.x0 + bx0 * scale_x
                                    py0 = img_rect.y0 + by0 * scale_y
                                    px1 = img_rect.x0 + bx1 * scale_x
                                    py1 = img_rect.y0 + by1 * scale_y
                                    r = fitz.Rect(px0 - 2, py0 - 2, px1 + 2, py1 + 2)
                                    redaction_rects.append((r, etype))
                                    summary["entities_found"][etype] += 1
                    except Exception as e:
                        logger.debug(f"Embedded header image redaction error: {e}")

        # 4. Apply native redaction annotations (NO FACE REDACTIONS)
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

            # 1. OCR text extraction
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
                    if re.search(r'(?:\+|00|0)?[5-7]\d{8}', re.sub(r'[\s.-]', '', txt)):
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
        # 1. OCR for text in certificates / documents (NO FACE REDACTIONS)
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
                        if txt.lower() not in RESUME_STOPWORDS and txt.lower() not in COMMON_GIVEN_NAMES:
                            draw.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], fill=(0, 0, 0))
                            summary["entities_found"]["PERSON"] += 1
                            summary["total_redactions"] += 1
                            continue

                    # Check for phone
                    if re.search(r'(?:\+|00|0)?[5-7]\d{8}', re.sub(r'[\s.-]', '', txt)):
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
