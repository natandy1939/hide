import logging
import re
from typing import List, Optional, Set
import spacy
from presidio_analyzer import (
    AnalyzerEngine,
    EntityRecognizer,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
    RecognizerRegistry,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider

logger = logging.getLogger(__name__)

# Fallback Arabic name patterns and gazetteers
ARABIC_COMMON_FIRST_NAMES: Set[str] = {
    'محمد', 'احمد', 'أحمد', 'إبراهيم', 'ابراهيم', 'محمود', 'علي', 'حسن', 'حسين',
    'عمر', 'عمرو', 'خالد', 'طارق', 'يوسف', 'مصطفى', 'كريم', 'ياسين', 'حمزة',
    'سعيد', 'هشام', 'وليد', 'رشيد', 'عادل', 'سامي', 'أيمن', 'ايمن', 'زياد',
    'فاطمة', 'مريم', 'سارة', 'خديجة', 'عائشة', 'نور', 'ياسمين', 'زينب', 'هدى',
    'منى', 'آية', 'اية', 'أسماء', 'اسماء', 'أمينة', 'امينة', 'صفاء', 'شيماء',
    'سلمى', 'ليلى', 'سميرة', 'حنان', 'وفاء', 'سناء', 'جمال', 'سمير', 'كمال',
    'بلال', 'انس', 'أنس', 'يونس', 'سفيان', 'عبدالله', 'عبد الله', 'عبدالرحمن',
    'عبد الرحمن', 'عبدالعزيز', 'عبد العزيز', 'صلاح', 'مهدي', 'نبيل', 'فؤاد'
}

ARABIC_TITLES = (
    r'(?:السيد|السيدة|الآنسة|الأستاذ|الاستاذ|الأستاذة|الاستاذة|'
    r'الدكتور|الدكتورة|المهندس|المهندسة|المحامي|المحامية|الباحث|الباحثة)'
)

ARABIC_COMPOUND_PREFIXES = (
    r'(?:عبد\s+(?:الله|الرحمن|الرحيم|العزيز|المجيد|الملك|الكريم|الواحد|الفتاح|الخالق|الرزاق|اللطيف|العليم|الحليم|الحكيم|الشكور|الحميد|القدوس|السلام|المؤمن|المهيمن|الجبار|المتكبر)|'
    r'(?:نور|شمس|سيف|صلاح|علاء|عماد|جمال|بهاء|ضياء|حسام|كمال)\s+الدين)'
)


class ArabicNerRecognizer(EntityRecognizer):
    """
    Dedicated EntityRecognizer for Arabic Candidate Names (PERSON / اسم).
    Integrates Hugging Face NER pipeline (CAMeL-Lab/bert-base-arabic-camelbert-mix-ner or similar)
    with graceful fallback to heuristic gazetteer and grammatical patterns.
    """

    def __init__(
        self,
        supported_language: str = "ar",
        supported_entities: Optional[List[str]] = None,
        hf_model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
        enable_hf: bool = False,
    ):
        if supported_entities is None:
            supported_entities = ["PERSON"]
        self.hf_model_name = hf_model_name
        self.enable_hf = enable_hf
        self._hf_pipeline = None
        self._hf_load_attempted = False
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=f"ArabicNerRecognizer_{supported_language}",
        )

    def load(self) -> None:
        if self.enable_hf and not self._hf_load_attempted:
            self._hf_load_attempted = True
            try:
                from transformers import pipeline
                logger.info(f"Loading Arabic NER transformer model: {self.hf_model_name}")
                self._hf_pipeline = pipeline(
                    "token-classification",
                    model=self.hf_model_name,
                    aggregation_strategy="simple"
                )
                logger.info("Arabic NER transformer model loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"Could not load Hugging Face model {self.hf_model_name}: {e}. "
                    "Falling back to high-accuracy Arabic pattern recognizer."
                )
                self._hf_pipeline = None

    def analyze(
        self, text: str, entities: List[str], nlp_artifacts=None
    ) -> List[RecognizerResult]:
        import unicodedata
        results: List[RecognizerResult] = []
        if "PERSON" not in entities:
            return results

        # Run primary detection on raw text
        results.extend(self._analyze_text_segment(text))

        # Check for Arabic Presentation Forms / Visual RTL Order in PDF streams
        # E.g. 'ﺭﻭﺹﻥﻡ ﺩﻡﺡﺃ' (Presentation Forms) or reversed glyphs
        norm_text = unicodedata.normalize('NFKC', text)
        if norm_text != text:
            norm_results = self._analyze_text_segment(norm_text)
            for nr in norm_results:
                if not any(r.start == nr.start and r.end == nr.end for r in results):
                    results.append(nr)

        # Also inspect reversed words (visual RTL extraction)
        words_with_spans = list(re.finditer(r'\S+', text))
        for idx, match in enumerate(words_with_spans):
            w_raw = match.group()
            w_norm = unicodedata.normalize('NFKC', w_raw)
            w_rev = w_norm[::-1]
            if w_norm in ARABIC_COMMON_FIRST_NAMES or w_rev in ARABIC_COMMON_FIRST_NAMES:
                start_char = match.start()
                end_char = match.end()
                # Include adjacent Arabic token (family name)
                if idx - 1 >= 0:
                    prev_w = words_with_spans[idx - 1]
                    if re.match(r'^[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]+$', prev_w.group()) and not any(kw in prev_w.group() for kw in [':', 'الاسم', 'ﻡﺱﺍﻝﺍ']):
                        start_char = prev_w.start()
                if idx + 1 < len(words_with_spans):
                    next_w = words_with_spans[idx + 1]
                    if re.match(r'^[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]+$', next_w.group()) and not any(kw in next_w.group() for kw in [':', 'هاتف', 'بريد']):
                        end_char = next_w.end()

                if not any(r.start <= start_char and r.end >= end_char for r in results):
                    results.append(
                        RecognizerResult(
                            entity_type="PERSON",
                            start=start_char,
                            end=end_char,
                            score=0.90,
                        )
                    )

        return results

    def _analyze_text_segment(self, text: str) -> List[RecognizerResult]:
        results: List[RecognizerResult] = []

        # 1. Try Hugging Face transformer if enabled
        if self.enable_hf:
            if not self._hf_load_attempted:
                self.load()
            if self._hf_pipeline is not None:
                try:
                    hf_preds = self._hf_pipeline(text)
                    for pred in hf_preds:
                        entity_group = pred.get("entity_group", "") or pred.get("entity", "")
                        if "PER" in entity_group.upper():
                            results.append(
                                RecognizerResult(
                                    entity_type="PERSON",
                                    start=pred["start"],
                                    end=pred["end"],
                                    score=float(pred.get("score", 0.85)),
                                )
                            )
                except Exception as e:
                    logger.warning(f"Error during Arabic HF NER inference: {e}")

        # 2. Contextual Arabic Pattern Recognizers (Titles, Header, Resume introduction)
        # Matches: [Title] [First Name] [Father/Family Name]
        title_pattern = re.compile(
            rf'(?:^|\s)({ARABIC_TITLES}\s+([\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){{1,3}}))',
            re.UNICODE
        )
        for match in title_pattern.finditer(text):
            full_span = match.span(1)
            results.append(
                RecognizerResult(
                    entity_type="PERSON",
                    start=full_span[0],
                    end=full_span[1],
                    score=0.90,
                )
            )

        # Matches "الاسم: أحمد منصور" or "المرشح: ..."
        label_pattern = re.compile(
            r'(?:الاسم(?:\s+الكامل)?|المرشح|صاحب السيرة)\s*[:\-]\s*([\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){1,3})',
            re.UNICODE
        )
        for match in label_pattern.finditer(text):
            name_span = match.span(1)
            results.append(
                RecognizerResult(
                    entity_type="PERSON",
                    start=name_span[0],
                    end=name_span[1],
                    score=0.95,
                )
            )

        # 3. Compound Name Pattern (e.g. عبد الله أحمد, نور الدين عمر)
        compound_pattern = re.compile(
            rf'(?:^|\s)({ARABIC_COMPOUND_PREFIXES}(?:\s+[\u0600-\u06FF]+){{1,2}})',
            re.UNICODE
        )
        for match in compound_pattern.finditer(text):
            span = match.span(1)
            results.append(
                RecognizerResult(
                    entity_type="PERSON",
                    start=span[0],
                    end=span[1],
                    score=0.88,
                )
            )

        # 4. Gazetteer First Name + Next Arabic Words
        words = list(re.finditer(r'[\u0600-\u06FF]+', text))
        i = 0
        while i < len(words):
            word_str = words[i].group()
            if word_str in ARABIC_COMMON_FIRST_NAMES:
                # Capture first name and following 1 to 2 words as full name
                start_char = words[i].start()
                end_char = words[i].end()
                count = 1
                j = i + 1
                while j < len(words) and count < 3:
                    # Check that words are separated only by whitespace
                    intervening = text[end_char:words[j].start()]
                    if intervening.strip() == "":
                        end_char = words[j].end()
                        count += 1
                        j += 1
                    else:
                        break
                
                # Check if this span is already covered
                already_covered = any(
                    r.start <= start_char and r.end >= end_char for r in results
                )
                if not already_covered:
                    results.append(
                        RecognizerResult(
                            entity_type="PERSON",
                            start=start_char,
                            end=end_char,
                            score=0.85 if count > 1 else 0.70,
                        )
                    )
                i = j
            else:
                i += 1

        return results


def build_phone_recognizers() -> List[PatternRecognizer]:
    """
    Creates custom phone number pattern recognizers for:
    - International E.164 formats (+XX ... / 00XX ...)
    - French formats (06/07..., +33...)
    - North African / Moroccan formats (05/06/07..., +212...)
    """
    recognizers: List[PatternRecognizer] = []

    # 1. French phone numbers (Local and International)
    # Examples: 06 12 34 56 78, 07.12.34.56.78, 0145236789, +33 6 12 34 56 78, +33 (0)6 12 34 56 78
    fr_patterns = [
        Pattern(
            name="fr_phone_local",
            regex=r'(?<!\d)(?:0|\+33\s*(?:\(0\)\s*)?)[1-9](?:[\s.-]?\d{2}){4}(?!\d)',
            score=0.85,
        ),
        Pattern(
            name="fr_phone_spaced",
            regex=r'(?<!\d)(?:0|\+33\s*)[1-9](?:\s+\d{2}){4}(?!\d)',
            score=0.90,
        ),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            name="FrenchPhoneRecognizer",
            patterns=fr_patterns,
            supported_language="fr",
        )
    )

    # 2. Moroccan and North African phone numbers
    # Examples: 0612345678, 0712345678, 0522334455, +212 6 12 34 56 78, +212 (0)6...
    # Also Algerian (+213), Tunisian (+216)
    ma_patterns = [
        Pattern(
            name="ma_phone_intl",
            regex=r'(?<!\d)(?:\+212\s*(?:\(0\)\s*)?|00212\s*)[5-7](?:[\s.-]?\d{2}){4}(?!\d)',
            score=0.90,
        ),
        Pattern(
            name="ma_phone_local",
            regex=r'(?<!\d)0[5-7](?:[\s.-]?\d{2}){4}(?!\d)',
            score=0.85,
        ),
        Pattern(
            name="dz_tn_phone",
            regex=r'(?<!\d)(?:\+213|\+216|00213|00216)[\s.-]?[5-7]\d(?:[\s.-]?\d{2}){3}(?!\d)',
            score=0.85,
        ),
    ]
    recognizers.append(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            name="NorthAfricaPhoneRecognizer",
            patterns=ma_patterns,
            supported_language="ar",
        )
    )
    # Also add to 'fr' and 'en' as CVs in North Africa are frequently written in French or English
    recognizers.append(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            name="NorthAfricaPhoneRecognizerMultilingual",
            patterns=ma_patterns,
            supported_language="fr",
        )
    )
    recognizers.append(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            name="NorthAfricaPhoneRecognizerEN",
            patterns=ma_patterns,
            supported_language="en",
        )
    )

    # 3. Universal International E.164 & formatted phone numbers
    # Examples: +1 (555) 234-5678, +44 20 7946 0958, +971 50 123 4567, +966 50 123 4567
    intl_patterns = [
        Pattern(
            name="e164_universal",
            regex=r'(?<!\d)(?:\+|00)[1-9]\d{0,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?)?\d{1,4}[\s.-]?\d{2,4}[\s.-]?\d{2,9}(?!\d)',
            score=0.80,
        )
    ]
    for lang in ["en", "fr", "ar"]:
        recognizers.append(
            PatternRecognizer(
                supported_entity="PHONE_NUMBER",
                name=f"UniversalIntlPhoneRecognizer_{lang}",
                patterns=intl_patterns,
                supported_language=lang,
            )
        )

    return recognizers


def detect_language(text: str) -> str:
    """
    Detect language of the document text.
    Returns 'ar', 'fr', or 'en'.
    Prioritizes Arabic script detection, then langdetect with safe fallback.
    """
    if not text or not text.strip():
        return "en"

    # 1. Script-based check for Arabic
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_alpha = len(re.findall(r'[^\W\d_]', text))
    if total_alpha > 0 and (arabic_chars / total_alpha) > 0.25:
        return "ar"

    # 2. Use langdetect for Latin script languages
    try:
        from langdetect import detect
        detected = detect(text)
        if detected in ["fr", "ar", "en"]:
            return detected
        elif detected in ["es", "it", "pt", "ca"]:
            return "fr"  # Romance language similarity fallback
        else:
            return "en"
    except Exception:
        # Fallback keyword checks
        lower = text.lower()
        if any(w in lower for w in ["curriculum vitae", "expérience", "formation", "compétences", "téléphone"]):
            return "fr"
        return "en"


def get_available_spacy_models():
    """
    Check which spaCy models are installed and choose lg over sm when available.
    """
    en_model = "en_core_web_lg" if spacy.util.is_package("en_core_web_lg") else "en_core_web_sm"
    fr_model = "fr_core_news_lg" if spacy.util.is_package("fr_core_news_lg") else "fr_core_news_sm"
    return en_model, fr_model


def create_analyzer_engine(enable_arabic_hf: bool = False) -> AnalyzerEngine:
    """
    Build and configure the multilingual Presidio AnalyzerEngine.
    Sets up spaCy NLP engine for English and French, blank spaCy for Arabic,
    and registers custom recognizers for Arabic NER and international phone numbers.
    """
    en_model, fr_model = get_available_spacy_models()
    logger.info(f"Using spaCy models: en={en_model}, fr={fr_model}")

    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": en_model},
            {"lang_code": "fr", "model_name": fr_model},
        ],
    }

    provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
    nlp_engine = provider.create_engine()

    # Register blank Arabic spaCy pipeline in the NLP engine to allow Presidio tokenization
    try:
        nlp_engine.nlp["ar"] = spacy.blank("ar")
    except Exception as e:
        logger.warning(f"Could not initialize blank Arabic spacy pipeline: {e}")

    # Build recognizer registry with default recognizers
    registry = RecognizerRegistry(supported_languages=["en", "fr", "ar"])
    registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en", "fr"])

    # Register custom phone recognizers
    for phone_rec in build_phone_recognizers():
        registry.add_recognizer(phone_rec)

    # Register universal EmailRecognizer for Arabic
    email_patterns = [
        Pattern(
            name="email_universal",
            regex=r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
            score=1.0,
        )
    ]
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            name="UniversalEmailRecognizer_ar",
            patterns=email_patterns,
            supported_language="ar",
        )
    )

    # Register custom Arabic NER recognizer
    arabic_recognizer = ArabicNerRecognizer(
        supported_language="ar",
        enable_hf=enable_arabic_hf,
    )
    registry.add_recognizer(arabic_recognizer)

    # Also register Arabic recognizer for multilingual documents where Arabic text appears in EN/FR CVs
    arabic_rec_multilingual = ArabicNerRecognizer(
        supported_language="en",
        enable_hf=enable_arabic_hf,
    )
    registry.add_recognizer(arabic_rec_multilingual)

    arabic_rec_fr = ArabicNerRecognizer(
        supported_language="fr",
        enable_hf=enable_arabic_hf,
    )
    registry.add_recognizer(arabic_rec_fr)

    # Assemble and return AnalyzerEngine
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["en", "fr", "ar"],
    )

    return analyzer




