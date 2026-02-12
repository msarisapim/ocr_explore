# ocr_doctr.py
import io, re
from functools import lru_cache
from typing import List, Tuple, Union
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, recognition

def _bgr_to_png_bytes(img_bgr) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()

@lru_cache(maxsize=1)
def get_recognizer():
    model = recognition.crnn_vgg16_bn(pretrained=True)
    return model.eval()

def doctr_recognize_line(crop_bgr):
    import torch
    import torchvision.transforms as T
    rec = get_recognizer()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((32, 128)),  # normal input size for recognizer
        T.ToTensor(),
    ])
    img_t = transform(crop_bgr).unsqueeze(0)
    out = rec(img_t)
    return out[0][0]  # predicted string

def get_model(device=None):
    # assume_straight_pages speeds things up if your crops are already upright
    m = ocr_predictor(pretrained=True, assume_straight_pages=True)
    
    # Use specified device or auto-detect
    if device:
        try:
            m = m.to(device)
        except Exception:
            pass
    else:
        # Auto-detect fallback
        try:
            import torch
            if torch.cuda.is_available():
                m = m.to("cuda")
            elif torch.backends.mps.is_available():
                m = m.to("mps")
        except Exception:
            pass
    
    m.eval()
    return m

def doctr_read_line(crop_bgr, min_conf=0.1, device=None):
    _, buf = cv2.imencode(".png", crop_bgr)
    doc = DocumentFile.from_images([buf.tobytes()])
    # model = ocr_predictor(pretrained=True)
    model = get_model(device=device)
    result = model(doc).export()

    # Just return all words from this crop (since it's one line)
    words = []
    for page in result["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                for w in line["words"]:
                    if w["confidence"] >= min_conf:
                        words.append(w["value"])
    return " ".join(words)

def doctr_text_lines_from_bgr(
    imgs_bgr: Union[any, List[any]],
    min_conf: float = 0.0,
    device=None
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

    model = get_model(device=device)
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

def doctr_first_line(img_bgr, min_conf=0.1, device=None):
    _, buf = cv2.imencode(".png", img_bgr)
    doc = DocumentFile.from_images([buf.tobytes()])
    # model = ocr_predictor(pretrained=True)
    model = get_model(device=device)
    result = model(doc).export()

    for page in result["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                words = [w["value"] for w in line["words"] if w["confidence"] >= min_conf]
                if words:
                    return " ".join(words)  # only first line
    return ""

def doctr_text_join(
    img_bgr,
    min_conf: float = 0.0,
    normalize_spaces: bool = True,
    device=None
) -> str:
    """Single image → one normalized string."""
    lines = doctr_text_lines_from_bgr(img_bgr, min_conf=min_conf, device=device)[0]
    text = "\n".join(lines)
    if normalize_spaces:
        text = re.sub(r"[ \t]+", " ", text).strip()
    return text
