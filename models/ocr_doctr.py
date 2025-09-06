# ocr_doctr.py
import io, re
from functools import lru_cache
from typing import List, Tuple, Union
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def _bgr_to_png_bytes(img_bgr) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()

@lru_cache(maxsize=1)
def get_model():
    # assume_straight_pages speeds things up if your crops are already upright
    m = ocr_predictor(pretrained=True, assume_straight_pages=True)
    # Optional (Torch backend only): push to GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            m = m.to("cuda")
    except Exception:
        pass
    m.eval()
    return m

def doctr_text_lines_from_bgr(
    imgs_bgr: Union[any, List[any]],
    min_conf: float = 0.0
) -> List[List[str]]:
    """
    Input: single BGR image or list of BGR images (OpenCV).
    Output: list per-image of text lines (strings), confidence-filtered.
    """
    if not isinstance(imgs_bgr, list):
        imgs_bgr = [imgs_bgr]

    # Encode all crops to raw bytes (DocTR likes bytes/paths)
    bytes_list = []
    for img in imgs_bgr:
        if img is None or img.size == 0:
            raise ValueError("Empty image in batch")
        bytes_list.append(_bgr_to_png_bytes(img))

    model = get_model()
    doc = DocumentFile.from_images(bytes_list)
    pred = model(doc).export()

    out: List[List[str]] = []
    for page in pred.get("pages", []):
        page_lines = []
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                # Keep words >= min_conf
                words = [w["value"] for w in line.get("words", []) if w.get("confidence", 1.0) >= min_conf]
                if words:
                    page_lines.append(" ".join(words))
        out.append(page_lines)
    return out

def doctr_text_join(
    img_bgr,
    min_conf: float = 0.0,
    normalize_spaces: bool = True
) -> str:
    """Single image → one normalized string."""
    lines = doctr_text_lines_from_bgr(img_bgr, min_conf=min_conf)[0]
    text = "\n".join(lines)
    if normalize_spaces:
        text = re.sub(r"[ \t]+", " ", text).strip()
    return text
