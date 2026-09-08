import os
import urllib.request
import logging
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Cache directory for vision models
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "cv_redactor_models")
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


class ProfilePhotoDetector:
    """
    Locates candidate profile photos, avatars, and headshots on PDF pages
    using MediaPipe FaceDetector with OpenCV DNN (YuNet) fallback.
    Targets only the face/headshot area with tight padding, preserving surrounding
    page content and background banners.
    """

    def __init__(
        self,
        min_face_size_pt: float = 25.0,
        max_face_size_pt: float = 160.0,
        padding_factor: float = 0.20,
        confidence_threshold: float = 0.60,
    ):
        self.min_face_size_pt = min_face_size_pt
        self.max_face_size_pt = max_face_size_pt
        self.padding_factor = padding_factor
        self.confidence_threshold = confidence_threshold

        os.makedirs(CACHE_DIR, exist_ok=True)
        self.mp_detector = None
        self.yunet_detector = None

        self._init_detector()

    def _init_detector(self) -> None:
        """Attempts to initialize MediaPipe FaceDetector, then OpenCV YuNet."""
        try:
            mp_model_path = os.path.join(CACHE_DIR, "blaze_face_short_range.tflite")
            if not os.path.exists(mp_model_path):
                logger.info(f"Downloading MediaPipe face detection model to {mp_model_path}...")
                urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, mp_model_path)

            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=mp_model_path)
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=self.confidence_threshold,
            )
            self.mp_detector = vision.FaceDetector.create_from_options(options)
            logger.info("MediaPipe FaceDetector successfully initialized.")
            return
        except Exception as e:
            logger.warning(f"MediaPipe initialization failed: {e}. Attempting OpenCV YuNet fallback.")

        try:
            import cv2
            yunet_model_path = os.path.join(CACHE_DIR, "face_detection_yunet.onnx")
            if not os.path.exists(yunet_model_path):
                logger.info(f"Downloading OpenCV YuNet model to {yunet_model_path}...")
                urllib.request.urlretrieve(YUNET_MODEL_URL, yunet_model_path)

            self.yunet_model_path = yunet_model_path
            logger.info("OpenCV YuNet face detector model ready.")
        except Exception as e:
            logger.warning(f"YuNet model setup failed: {e}.")

    def detect_faces_in_numpy(
        self, img_bgr_or_rgb: np.ndarray, is_rgb: bool = True
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in a numpy image.
        Returns bounding boxes as (x, y, w, h) in pixel coordinates.
        """
        h_img, w_img = img_bgr_or_rgb.shape[:2]
        if h_img < 40 or w_img < 40:
            return []

        rgb_img = img_bgr_or_rgb if is_rgb else img_bgr_or_rgb[:, :, ::-1]

        # 1. MediaPipe
        if self.mp_detector is not None:
            try:
                import mediapipe as mp
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb_img)
                )
                detection_result = self.mp_detector.detect(mp_image)
                boxes = []
                for detection in detection_result.detections:
                    score = detection.categories[0].score if detection.categories else 1.0
                    if score < self.confidence_threshold:
                        continue
                    bb = detection.bounding_box
                    x = max(0, int(bb.origin_x))
                    y = max(0, int(bb.origin_y))
                    w = min(int(bb.width), w_img - x)
                    h = min(int(bb.height), h_img - y)
                    if w >= 25 and h >= 25:
                        boxes.append((x, y, w, h))
                return boxes
            except Exception as e:
                logger.debug(f"MediaPipe inference failed: {e}")

        # 2. OpenCV YuNet fallback
        if hasattr(self, "yunet_model_path") and os.path.exists(self.yunet_model_path):
            try:
                import cv2
                detector = cv2.FaceDetectorYN_create(
                    self.yunet_model_path,
                    "",
                    (w_img, h_img),
                    score_threshold=self.confidence_threshold,
                    nms_threshold=0.3,
                    top_k=10,
                )
                bgr_img = rgb_img[:, :, ::-1] if is_rgb else rgb_img
                _, faces = detector.detect(bgr_img)
                boxes = []
                if faces is not None:
                    for face in faces:
                        x, y, w, h = map(int, face[0:4])
                        if w >= 25 and h >= 25:
                            boxes.append((max(0, x), max(0, y), min(w, w_img - x), min(h, h_img - y)))
                return boxes
            except Exception as e:
                logger.debug(f"YuNet inference failed: {e}")

        return []

    def detect_faces_on_page(
        self, page: fitz.Page, dpi: int = 150
    ) -> List[fitz.Rect]:
        """
        Scans a PDF page for profile photos and headshots.
        Calculates exact face bounding boxes with tight padding.
        Never expands to cover full page background banners or layout graphics.
        """
        rects: List[fitz.Rect] = []
        scale = 72.0 / dpi
        page_rect = page.rect

        # 1. Detect faces on rendered page canvas
        pix = page.get_pixmap(dpi=dpi)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_rgb = img_np[:, :, :3]
        elif pix.n == 1:
            img_rgb = np.stack([img_np.squeeze()] * 3, axis=-1)
        else:
            img_rgb = img_np

        face_boxes = self.detect_faces_in_numpy(img_rgb, is_rgb=True)

        for (x, y, w, h) in face_boxes:
            pt_w = w * scale
            pt_h = h * scale

            if pt_w < self.min_face_size_pt or pt_h < self.min_face_size_pt:
                continue
            if pt_w > self.max_face_size_pt or pt_h > self.max_face_size_pt:
                continue

            pt_x0 = x * scale
            pt_y0 = y * scale
            pt_x1 = pt_x0 + pt_w
            pt_y1 = pt_y0 + pt_h

            pad_w = pt_w * self.padding_factor
            pad_h = pt_h * (self.padding_factor * 1.25)

            rect = fitz.Rect(
                max(0.0, pt_x0 - pad_w),
                max(0.0, pt_y0 - pad_h),
                min(page_rect.width, pt_x1 + pad_w),
                min(page_rect.height, pt_y1 + pad_h),
            )
            rects.append(rect)

        # 2. Inspect embedded images: map sub-image face coordinates directly to page points
        try:
            image_list = page.get_images(full=True)
            for img_info in image_list:
                xref = img_info[0]
                img_rects = page.get_image_rects(xref)
                if not img_rects:
                    continue

                for r in img_rects:
                    # Skip icons and tiny elements (< 40 pt)
                    if r.width < 40 or r.height < 40:
                        continue

                    # If an already detected face is contained within this region, we already have it
                    if any(existing.intersects(r) for existing in rects):
                        continue

                    try:
                        base_image = page.parent.extract_image(xref)
                        image_bytes = base_image["image"]
                        import io
                        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        w_img, h_img = pil_img.size
                        sub_faces = self.detect_faces_in_numpy(np.array(pil_img), is_rgb=True)

                        for (fx, fy, fw, fh) in sub_faces:
                            # Map sub-image face coordinates to PDF page coordinates
                            px0 = r.x0 + (fx / w_img) * r.width
                            py0 = r.y0 + (fy / h_img) * r.height
                            px1 = px0 + (fw / w_img) * r.width
                            py1 = py0 + (fh / h_img) * r.height

                            pt_w = px1 - px0
                            pt_h = py1 - py0
                            if pt_w < self.min_face_size_pt or pt_w > self.max_face_size_pt:
                                continue

                            pad_w = pt_w * self.padding_factor
                            pad_h = pt_h * (self.padding_factor * 1.25)

                            face_rect = fitz.Rect(
                                max(0.0, px0 - pad_w),
                                max(0.0, py0 - pad_h),
                                min(page_rect.width, px1 + pad_w),
                                min(page_rect.height, py1 + pad_h),
                            )
                            rects.append(face_rect)
                    except Exception as e:
                        logger.debug(f"Error inspecting embedded image xref {xref}: {e}")
        except Exception as e:
            logger.debug(f"Error checking embedded images: {e}")

        return self._merge_rectangles(rects)

    def close(self):
        """Cleanly releases detector resources."""
        if self.mp_detector is not None:
            try:
                self.mp_detector.close()
            except Exception:
                pass
            self.mp_detector = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _merge_rectangles(self, rects: List[fitz.Rect]) -> List[fitz.Rect]:
        """Merges overlapping face rectangles."""
        if not rects:
            return []

        merged: List[fitz.Rect] = []
        for r in rects:
            combined = False
            for i, existing in enumerate(merged):
                if r.intersects(existing):
                    merged[i] = existing | r
                    combined = True
                    break
            if not combined:
                merged.append(r)

        return merged
