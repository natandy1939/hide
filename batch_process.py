import os
import sys
import time
import json
import logging
import fitz
from typing import Dict, List, Set, Any, Tuple
from redactor import CVRedactor

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('batch_process')

INPUT_ROOT = 'test'
OUTPUT_ROOT = 'result'


def discover_candidate_folders(input_root: str) -> List[Tuple[str, List[str]]]:
    """
    Groups files by candidate directory.
    Returns list of tuples: (candidate_id, [absolute_file_paths])
    """
    candidate_folders = []
    for entry in sorted(os.listdir(input_root)):
        entry_path = os.path.join(input_root, entry)
        if os.path.isdir(entry_path):
            files = []
            for root, _, filenames in os.walk(entry_path):
                for f in filenames:
                    files.append(os.path.join(root, f))
            if files:
                candidate_folders.append((entry, files))
    return candidate_folders


def run_batch():
    if not os.path.exists(INPUT_ROOT):
        logger.error(f'Input root folder "{INPUT_ROOT}" not found!')
        sys.exit(1)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    candidate_folders = discover_candidate_folders(INPUT_ROOT)
    total_candidates = len(candidate_folders)
    total_files = sum(len(files) for _, files in candidate_folders)

    logger.info(f"Discovered {total_candidates} candidate folders containing {total_files} total files.")

    # Initialize CVRedactor with multilingual transformer
    redactor = CVRedactor()

    start_time = time.time()
    successful = 0
    errors = 0
    aggregate_entities = {
        'PERSON': 0,
        'PHONE_NUMBER': 0,
        'EMAIL_ADDRESS': 0,
        'PROFILE_PHOTO': 0,
    }
    total_redactions = 0
    file_records = []
    file_counter = 0

    for cand_idx, (cand_id, file_paths) in enumerate(candidate_folders, 1):
        # 1. Identify candidate's CV in folder to discover identity
        cv_file = None
        for p in file_paths:
            base_lower = os.path.basename(p).lower()
            if '_cv.pdf' in base_lower or 'cv' in base_lower or 'resume' in base_lower:
                cv_file = p
                break
        if not cv_file:
            # Fallback to first PDF
            for p in file_paths:
                if p.lower().endswith('.pdf'):
                    cv_file = p
                    break

        cand_names: Set[str] = set()
        if cv_file:
            try:
                cv_doc = fitz.open(cv_file)
                cand_names = redactor._identify_candidate_names(cv_doc)
                cv_doc.close()
            except Exception as e:
                logger.warning(f"Could not extract candidate identity from {cv_file}: {e}")

        # Inspect any attestation/certificate PDFs in this dossier for recipient names
        for p in file_paths:
            if p != cv_file and p.lower().endswith('.pdf'):
                try:
                    att_doc = fitz.open(p)
                    att_names = redactor._extract_attestation_names(att_doc)
                    att_doc.close()
                    for aname in att_names:
                        cand_names.update(redactor._generate_candidate_search_strings(aname))
                except Exception as e:
                    logger.debug(f"Could not extract attestation names from {p}: {e}")

        logger.info(
            f"[{cand_idx}/{total_candidates}] Candidate {cand_id} ({len(file_paths)} files) | "
            f"Identified Identity: {sorted(list(cand_names))[:5]}"
        )

        # 2. Process all files in this candidate dossier
        for src_path in file_paths:
            file_counter += 1
            rel_path = os.path.relpath(src_path, INPUT_ROOT)
            dst_path = os.path.join(OUTPUT_ROOT, rel_path)

            try:
                summary = redactor.redact_file(src_path, dst_path, candidate_names=cand_names)
                successful += 1

                ents = summary.get('entities_found', {})
                for k in aggregate_entities:
                    aggregate_entities[k] += ents.get(k, 0)
                total_redactions += summary.get('total_redactions', 0)

                file_records.append({
                    'candidate_id': cand_id,
                    'rel_path': rel_path,
                    'status': 'success',
                    'summary': summary,
                })
            except Exception as e:
                errors += 1
                logger.error(f"Error processing [{file_counter}/{total_files}] {rel_path}: {e}", exc_info=True)
                file_records.append({
                    'candidate_id': cand_id,
                    'rel_path': rel_path,
                    'status': 'error',
                    'error': str(e),
                })

    elapsed = time.time() - start_time

    report = {
        'total_candidates': total_candidates,
        'total_files': total_files,
        'successful': successful,
        'errors': errors,
        'total_redactions_applied': total_redactions,
        'entities_masked': aggregate_entities,
        'time_elapsed_seconds': round(elapsed, 2),
        'file_records': file_records,
    }

    report_path = os.path.join(OUTPUT_ROOT, 'redaction_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info('====================================================')
    logger.info(f'BATCH PROCESSING COMPLETED in {elapsed:.2f}s ({elapsed/60:.2f} min)')
    logger.info(f'Candidates: {total_candidates} | Files: {successful}/{total_files} (Errors: {errors})')
    logger.info(f'Total Redactions Applied: {total_redactions}')
    logger.info(f'Entities Masked: {aggregate_entities}')
    logger.info(f'Detailed report saved to: {report_path}')
    logger.info('====================================================')


if __name__ == '__main__':
    run_batch()
