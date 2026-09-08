import os
import io
import urllib.request
import logging
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw

from redactor import CVRedactor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_AVATAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
SYNTHETIC_PDF_PATH = "synthetic_multilingual_cv.pdf"
SANITIZED_PDF_PATH = "sanitized_cv.pdf"
SCANNED_PDF_PATH = "scanned_cv.pdf"
SANITIZED_SCANNED_PDF_PATH = "sanitized_scanned_cv.pdf"


def get_or_create_avatar_image(path: str = "avatar.jpg") -> str:
    """
    Downloads a sample portrait or generates a clean avatar image with facial features.
    """
    if os.path.exists(path):
        return path

    try:
        urllib.request.urlretrieve(SAMPLE_AVATAR_URL, path)
        logger.info(f"Downloaded sample face photo to {path}")
        return path
    except Exception as e:
        logger.warning(f"Could not download sample avatar: {e}. Generating synthetic face image.")
        # Create a fallback synthetic image with clear facial structure
        img = Image.new("RGB", (200, 200), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)
        # Face oval
        draw.ellipse([50, 40, 150, 160], fill=(235, 195, 165), outline=(180, 140, 110))
        # Eyes
        draw.ellipse([70, 75, 90, 90], fill=(50, 50, 60))
        draw.ellipse([110, 75, 130, 90], fill=(50, 50, 60))
        # Nose
        draw.polygon([(100, 95), (95, 115), (105, 115)], fill=(210, 160, 130))
        # Mouth
        draw.arc([80, 120, 120, 140], start=0, end=180, fill=(180, 60, 60), width=3)
        img.save(path)
        return path


def create_decorative_icon(color: tuple, size: int = 24) -> bytes:
    """Creates a tiny decorative icon (e.g. phone or email badge)."""
    img = Image.new("RGBA", (size, size), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([2, 2, size - 2, size - 2], radius=4, fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_synthetic_multilingual_cv(output_path: str) -> None:
    """
    Generates a realistic 1-page multilingual CV in English, French, and Arabic.
    Includes:
    - Candidate names (English/French: 'Jean-Luc Picard', Arabic: 'أحمد منصور')
    - Emails ('jeanluc.picard@starfleet.fr', 'ahmed.mansour@tech.ma')
    - Phone numbers ('06 12 34 56 78', '+212 6 12 34 56 78', '+1 555-0199')
    - Profile avatar photo
    - Decorative icons (phone badge, email badge)
    - Legitimate non-PII resume content
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size in points

    # Embed Arial or system font for Arabic unicode
    font_path = "C:/Windows/Fonts/arial.ttf"
    if os.path.exists(font_path):
        font = fitz.Font(fontfile=font_path)
        font_name = "arial_custom"
        page.insert_font(fontname=font_name, fontbuffer=font.buffer)
    else:
        font_name = "helv"

    # 1. Header Bar (Decorative header background)
    page.draw_rect(fitz.Rect(0, 0, 595, 130), color=None, fill=(0.12, 0.25, 0.45))

    # Candidate Name (English / French)
    page.insert_text(
        (40, 48),
        "Jean-Luc Picard",
        fontname="helv",
        fontsize=22,
        color=(1, 1, 1),
    )

    # Job Title
    page.insert_text(
        (40, 72),
        "Senior Cloud Architect & Systems Engineer",
        fontname="helv",
        fontsize=13,
        color=(0.85, 0.90, 1.0),
    )

    # Arabic Candidate Name & Title
    if font_name == "arial_custom":
        page.insert_textbox(
            fitz.Rect(40, 85, 450, 120),
            "المرشح: أحمد منصور - مهندس الأنظمة السحابية",
            fontname=font_name,
            fontsize=12,
            color=(0.95, 0.95, 0.95),
        )

    # 2. Profile Photo / Avatar (Top-Right)
    avatar_path = get_or_create_avatar_image("test_avatar.jpg")
    avatar_rect = fitz.Rect(470, 15, 565, 110)
    page.insert_image(avatar_rect, filename=avatar_path)

    # 3. Contact Details & Decorative Icons
    y_contact = 155
    page.insert_text((40, y_contact), "CONTACT DETAILS", fontname="helv", fontsize=11, color=(0.12, 0.25, 0.45))
    page.draw_line((40, y_contact + 5), (555, y_contact + 5), color=(0.7, 0.7, 0.7), width=1)

    # Decorative icon 1 (Email badge - tiny 18x18 pt)
    email_icon_bytes = create_decorative_icon((40, 120, 200), size=18)
    page.insert_image(fitz.Rect(40, y_contact + 15, 58, y_contact + 33), stream=email_icon_bytes)
    page.insert_text((65, y_contact + 28), "Email (FR): jeanluc.picard@starfleet.fr", fontname="helv", fontsize=10)
    page.insert_text((310, y_contact + 28), "Email (MA): ahmed.mansour@tech.ma", fontname="helv", fontsize=10)

    # Decorative icon 2 (Phone badge - tiny 18x18 pt)
    phone_icon_bytes = create_decorative_icon((40, 180, 100), size=18)
    page.insert_image(fitz.Rect(40, y_contact + 40, 58, y_contact + 58), stream=phone_icon_bytes)
    page.insert_text((65, y_contact + 53), "Phone (FR): 06 12 34 56 78", fontname="helv", fontsize=10)
    page.insert_text((220, y_contact + 53), "Phone (MA): +212 6 12 34 56 78", fontname="helv", fontsize=10)
    page.insert_text((410, y_contact + 53), "Intl: +1 555-0199", fontname="helv", fontsize=10)

    # 4. Professional Summary (Non-PII)
    y_sum = 240
    page.insert_text((40, y_sum), "PROFESSIONAL SUMMARY", fontname="helv", fontsize=11, color=(0.12, 0.25, 0.45))
    page.draw_line((40, y_sum + 5), (555, y_sum + 5), color=(0.7, 0.7, 0.7), width=1)

    summary_text = (
        "Seasoned systems architect with extensive expertise in distributed infrastructures, "
        "Kubernetes orchestration, high-availability clusters, and machine learning pipelines. "
        "Proven track record of delivering resilient cloud-native architectures."
    )
    page.insert_textbox(fitz.Rect(40, y_sum + 12, 555, y_sum + 60), summary_text, fontname="helv", fontsize=10)

    # 5. Technical Skills (Non-PII)
    y_skills = 320
    page.insert_text((40, y_skills), "TECHNICAL SKILLS", fontname="helv", fontsize=11, color=(0.12, 0.25, 0.45))
    page.draw_line((40, y_skills + 5), (555, y_skills + 5), color=(0.7, 0.7, 0.7), width=1)

    skills_text = (
        "• Core Languages: Python, Go, Rust, C++, SQL, Bash\n"
        "• Cloud & DevOps: Docker, Kubernetes, Terraform, AWS, Azure, CI/CD pipelines\n"
        "• Data & Vision: PyTorch, OpenCV, Document Processing, RapidOCR, Natural Language Processing"
    )
    page.insert_textbox(fitz.Rect(40, y_skills + 12, 555, y_skills + 80), skills_text, fontname="helv", fontsize=10)

    # 6. Professional Experience (Non-PII)
    y_exp = 420
    page.insert_text((40, y_exp), "WORK EXPERIENCE", fontname="helv", fontsize=11, color=(0.12, 0.25, 0.45))
    page.draw_line((40, y_exp + 5), (555, y_exp + 5), color=(0.7, 0.7, 0.7), width=1)

    exp_text = (
        "Lead Infrastructure Engineer | Enterprise Cloud Solutions (2020 - Present)\n"
        "- Architected multi-region Kubernetes deployments scaling to millions of daily requests.\n"
        "- Spearheaded automated data privacy pipelines ensuring full GDPR compliance.\n\n"
        "Senior Software Engineer | Data Systems Corp (2017 - 2020)\n"
        "- Engineered microservices processing financial telemetry with sub-millisecond latencies."
    )
    page.insert_textbox(fitz.Rect(40, y_exp + 12, 555, y_exp + 140), exp_text, fontname="helv", fontsize=10)

    # 7. Arabic Resume Section (Non-PII summary & skills)
    y_ar = 580
    if font_name == "arial_custom":
        page.insert_textbox(
            fitz.Rect(40, y_ar, 555, y_ar + 30),
            "الخبرات المهنية والمؤهلات",
            fontname=font_name,
            fontsize=12,
            color=(0.12, 0.25, 0.45),
        )
        page.draw_line((40, y_ar + 25), (555, y_ar + 25), color=(0.7, 0.7, 0.7), width=1)

        ar_body = (
            "المهارات التقنية: إدارة الخوادم، الحوسبة السحابية، تطوير نماذج الرؤية الحاسوبية، "
            "وحماية البيانات والخصوصية الرقمية."
        )
        page.insert_textbox(
            fitz.Rect(40, y_ar + 32, 555, y_ar + 80),
            ar_body,
            fontname=font_name,
            fontsize=10,
        )

    # 8. Set Author metadata (to verify metadata sanitization)
    doc.set_metadata({
        "author": "Jean-Luc Picard",
        "creator": "LibreOffice CV Generator",
        "title": "Jean-Luc Picard - Curriculum Vitae",
    })

    doc.save(output_path)
    doc.close()
    logger.info(f"Synthetic multilingual CV created at {output_path}")


def create_scanned_cv_from_pdf(src_pdf_path: str, scanned_pdf_path: str) -> None:
    """
    Simulates a scanned resume by rendering a PDF page to a 300 DPI image
    and embedding it into an image-only PDF with zero text streams.
    """
    src_doc = fitz.open(src_pdf_path)
    page = src_doc[0]
    w, h = page.rect.width, page.rect.height
    pix = page.get_pixmap(dpi=300)
    pix_bytes = pix.tobytes("png")
    src_doc.close()

    scanned_doc = fitz.open()
    new_page = scanned_doc.new_page(width=w, height=h)
    new_page.insert_image(new_page.rect, stream=pix_bytes)
    scanned_doc.save(scanned_pdf_path)
    scanned_doc.close()
    logger.info(f"Scanned CV PDF created at {scanned_pdf_path}")


def run_tests():
    logger.info("=== STEP 1: Generating Synthetic Multilingual CV ===")
    generate_synthetic_multilingual_cv(SYNTHETIC_PDF_PATH)

    logger.info("=== STEP 2: Running Automated Redaction Pipeline ===")
    redactor = CVRedactor(enable_arabic_hf=False)
    summary = redactor.redact_pdf(SYNTHETIC_PDF_PATH, SANITIZED_PDF_PATH)

    logger.info(f"Redaction Summary: {summary}")

    logger.info("=== STEP 3: Verifying Unrecoverability of PII ===")
    assert os.path.exists(SANITIZED_PDF_PATH), "Sanitized PDF file does not exist!"

    sanitized_doc = fitz.open(SANITIZED_PDF_PATH)
    all_text = " ".join([page.get_text() for page in sanitized_doc])

    logger.info(f"Total pages in sanitized PDF: {len(sanitized_doc)}")
    logger.info("Extracted residual text length: %d chars", len(all_text))

    # 1. Assert Names are completely excised
    pii_names = ["Jean-Luc Picard", "Picard", "Jean-Luc", "Sarah Connor"]
    for name in pii_names:
        assert name.lower() not in all_text.lower(), f"PII LEAK: Candidate name '{name}' still present in text stream!"

    # 2. Assert Arabic Name is completely excised
    import unicodedata
    norm_all_text = unicodedata.normalize('NFKC', all_text)
    rev_all_words = [w[::-1] for w in norm_all_text.split()]
    assert "أحمد منصور" not in all_text, "PII LEAK: Arabic name 'أحمد منصور' found in text stream!"
    assert "أحمد" not in all_text and "أحمد" not in norm_all_text and "أحمد" not in rev_all_words, "PII LEAK: Arabic first name 'أحمد' found in text stream!"

    # 3. Assert Emails are excised
    pii_emails = [
        "jeanluc.picard@starfleet.fr",
        "ahmed.mansour@tech.ma",
    ]
    for email in pii_emails:
        assert email.lower() not in all_text.lower(), f"PII LEAK: Email '{email}' still present in text stream!"

    # 4. Assert Phone Numbers are excised
    pii_phones = [
        "06 12 34 56 78",
        "+212 6 12 34 56 78",
        "+1 555-0199",
    ]
    for phone in pii_phones:
        assert phone not in all_text, f"PII LEAK: Phone number '{phone}' still present in text stream!"

    # 5. Assert Non-PII legitimate resume content is retained
    legit_terms = ["Kubernetes", "Docker", "Python", "Cloud Architect", "EXPERIENCE"]
    for term in legit_terms:
        assert term in all_text, f"OVER-REDACTION: Legitimate term '{term}' was erroneously deleted!"

    # 6. Assert Profile Photo was redacted
    assert summary["entities_found"]["PROFILE_PHOTO"] >= 1, "Face detector failed to identify avatar profile photo!"

    # 7. Assert PDF metadata was wiped
    meta = sanitized_doc.metadata
    assert not meta.get("author"), f"METADATA LEAK: Author metadata not stripped: {meta.get('author')}"
    assert not meta.get("title"), f"METADATA LEAK: Title metadata not stripped: {meta.get('title')}"

    sanitized_doc.close()
    logger.info(">>> SUCCESS: Searchable CV completely sanitized. All PII unrecoverable via text selection.")

    # === STEP 4: Test Scanned PDF Fallback ===
    logger.info("=== STEP 4: Testing Scanned PDF Fallback Pipeline ===")
    create_scanned_cv_from_pdf(SYNTHETIC_PDF_PATH, SCANNED_PDF_PATH)
    scanned_summary = redactor.redact_pdf(SCANNED_PDF_PATH, SANITIZED_SCANNED_PDF_PATH)
    logger.info(f"Scanned CV Summary: {scanned_summary}")

    assert scanned_summary["is_scanned"] is True, "Scanned detector failed to identify scanned document!"
    assert scanned_summary["total_redactions"] > 0, "No redactions applied to scanned document!"
    assert os.path.exists(SANITIZED_SCANNED_PDF_PATH), "Sanitized scanned PDF was not created!"

    scanned_doc = fitz.open(SANITIZED_SCANNED_PDF_PATH)
    assert len(scanned_doc) == 1, "Sanitized scanned doc has incorrect page count!"
    scanned_doc.close()

    logger.info(">>> SUCCESS: Scanned CV pipeline executed and sanitized successfully.")
    print("\n=======================================================")
    print("ALL VERIFICATION CHECKS PASSED: PIPELINE FULLY VALIDATED")
    print("=======================================================\n")


if __name__ == "__main__":
    run_tests()

