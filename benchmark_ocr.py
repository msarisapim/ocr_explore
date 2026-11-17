# python benchmark_ocr.py --mode pickpoints --input data/28012026_2328011.jpg --gt "TIS 50-2565" --whitelist '0123456789TIS.- '
# python benchmark_ocr.py --mode full --input data/for_OCR/7_29-8-2025_14H-11M_warped.jpg --gt "2900173"

import time
import argparse
import re
import cv2
import numpy as np
import math
import os
from PIL import Image
from rapidfuzz import fuzz
import platform
import matplotlib.pyplot as plt

# Configure matplotlib for macOS to support CJK/Thai characters
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]

# Engines
import easyocr
import pytesseract
from pytesseract import Output
from paddleocr import PaddleOCR
from rapidocr import RapidOCR
from models.ocr_doctr import doctr_text_join, doctr_read_line, doctr_text_lines_from_bgr, doctr_first_line
import models.ocr_glm
from ultralytics import solutions as ul_solutions
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# ---------- Utils ----------

def euclidean(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def pick_four_points(image):
    points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click 4 Points", image)

    cv2.namedWindow("Click 4 Points", cv2.WINDOW_NORMAL)
    cv2.imshow("Click 4 Points", image)
    cv2.setMouseCallback("Click 4 Points", click_event)

    while len(points) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            break
    cv2.destroyWindow("Click 4 Points")
    return points

def warp_perspective(image, points):
    (tl, tr, br, bl) = points
    widthA = euclidean(br, bl)
    widthB = euclidean(tr, tl)
    heightA = euclidean(tr, br)
    heightB = euclidean(tl, bl)
    minWidth = int(min(widthA, widthB))
    minHeight = int(min(heightA, heightB))

    pts_src = np.float32(points)
    pts_dst = np.float32([
        [0, 0],
        [minWidth - 1, 0],
        [minWidth - 1, minHeight - 1],
        [0, minHeight - 1]
    ])
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (minWidth, minHeight))
    return warped

def draw_points_overlay(image, points):
    """Return a copy of `image` with the selected points annotated."""
    overlay = image.copy()
    if not points:
        return overlay
    pts = [tuple(map(int, p)) for p in points]
    color = (0, 0, 255)
    for idx, (x, y) in enumerate(pts, start=1):
        cv2.circle(overlay, (x, y), 6, color, -1)
        cv2.putText(
            overlay,
            str(idx),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    if len(pts) >= 2:
        pts_array = np.array(pts, dtype=np.int32)
        cv2.polylines(
            overlay,
            [pts_array.reshape((-1, 1, 2))],
            isClosed=len(pts) >= 3,
            color=color,
            thickness=2,
        )
    return overlay

BOX_COLOR_CYCLE = [
    (231, 76, 60),   # blue-ish red
    (46, 204, 113),  # green
    (52, 152, 219),  # orange-ish blue
    (155, 89, 182),  # purple
    (241, 196, 15),  # teal-ish yellow
    (230, 126, 34),  # orange
    (26, 188, 156),  # teal
    (149, 165, 166), # gray
]

def _points_from_box(box_entry):
    """Normalize a box entry (points or rect) into a 4-point polygon."""
    if not box_entry:
        return None
    if "points" in box_entry and box_entry["points"]:
        pts = np.array(box_entry["points"], dtype=np.int32).reshape(-1, 2)
        return pts
    if "rect" in box_entry and box_entry["rect"]:
        x, y, w, h = [int(v) for v in box_entry["rect"]]
        return np.array(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            dtype=np.int32,
        )
    return None

def annotate_processed_image(image, results):
    """Return image copy with OCR bounding boxes drawn for each engine."""
    if image is None:
        return None
    annotated = image.copy()
    for idx, res in enumerate(results):
        boxes = res.get("boxes") or []
        if not boxes:
            continue
        color = BOX_COLOR_CYCLE[idx % len(BOX_COLOR_CYCLE)]
        label = res.get("name", f"engine_{idx}")
        for box in boxes:
            pts = _points_from_box(box)
            if pts is None or pts.size == 0:
                continue
            cv2.polylines(
                annotated,
                [pts.reshape((-1, 1, 2))],
                True,
                color,
                2,
            )
            min_x = int(np.min(pts[:, 0]))
            min_y = int(np.min(pts[:, 1])) - 5
            cv2.putText(
                annotated,
                label,
                (min_x, max(min_y, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return annotated

def save_results_figure(original_img, overlay_img, warped_img, results, output_path, mode):
    """Render and save a summary figure that visualizes the OCR benchmark."""
    if warped_img is None:
        warped_img = original_img

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    ax_input.set_title("Input (with pick points)" if mode == "pickpoints" else "Input image")
    ax_input.axis("off")

    processed_view = warped_img.copy()
    processed_title = "Warped image" if mode == "pickpoints" else "Processed view"
    has_boxes = any(res.get("boxes") for res in results)
    annotated = annotate_processed_image(warped_img, results) if has_boxes else None
    if annotated is not None:
        processed_view = annotated
        processed_title = f"{processed_title} (detections)"

    ax_warp = fig.add_subplot(gs[0, 1])
    ax_warp.imshow(cv2.cvtColor(processed_view, cv2.COLOR_BGR2RGB))
    ax_warp.set_title(processed_title)
    ax_warp.axis("off")

    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")
    lines = ["====== OCR Benchmark ======"]
    for res in results:
        if res.get("error"):
            lines.append(f"{res['name']:<10} | ERROR: {res['error']}")
            continue
        acc_text = res["acc"] if isinstance(res["acc"], str) else f"{res['acc']:.1f}"
        clean_text = res["text"].replace("\n", " ").strip()
        lines.append(f"{res['name']:<10} | {res['time_ms']:.1f} ms | text: {clean_text or '-'} | acc: {acc_text}")
    ax_text.text(
        0,
        1,
        "\n".join(lines),
        fontfamily="monospace",
        fontsize=10,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

# ---------- OCR Engines ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_OCR_DIR = os.path.join(BASE_DIR, "models", "for_yolo_ocr")
YOLO_OCR_SIZES = ["n", "s", "m", "l", "x"]

# Benchmark order. Each YOLO26-Text detector size is its own engine (yolo-ocr-n ... yolo-ocr-x).
ENGINE_NAMES = [
    "easyocr",
    "tesseract",
    "paddleocr",
    "paddleocr-onnx",
    "rapidocr",
    *[f"yolo-ocr-{s}" for s in YOLO_OCR_SIZES],
    "doctr",
    "trocr",
    "glm-ocr",
]


def resolve_device(device_override=None):
    if device_override:
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OCREngines:
    """Holds OCR engines. Each engine is loaded on first use (or up front via `preload`)."""

    def __init__(self, device_override=None, preload=None):
        self.device = resolve_device(device_override)
        print(f"[INFO] Using device: {self.device.upper()}")
        self._loaded = set()
        self.yolo_ocr = {}
        for name in preload or []:
            self.load(name)

    # ----- loading -----

    def load(self, name):
        """Load `name` if it is not loaded yet. Raises ValueError for unknown engines."""
        if name in self._loaded:
            return
        if name not in ENGINE_NAMES:
            raise ValueError(f"Unknown engine: {name}")
        print(f"[INFO] Loading {name}...")
        if name.startswith("yolo-ocr-"):
            self._load_yolo_ocr(name.rsplit("-", 1)[1])
        else:
            getattr(self, "_load_" + name.replace("-", "_"))()
        self._loaded.add(name)

    def _load_easyocr(self):
        self.easy = easyocr.Reader(['th', 'en'], gpu=(self.device != "cpu"))

    def _load_tesseract(self):
        pass  # external binary, nothing to load

    def _load_paddleocr(self):
        # PaddleOCR uses 'device' argument (e.g., 'gpu', 'cpu')
        # MPS is not supported in PaddleX backend yet, fallback to CPU
        paddle_device = self.device
        if self.device == "cuda":
            paddle_device = "gpu"
        elif self.device == "mps":
            print("[WARN] PaddleOCR (PaddleX) does not support MPS, falling back to CPU.")
            paddle_device = "cpu"

        self.paddle = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            textline_orientation_model_dir=None,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device=paddle_device,
        )

    def _load_rapidocr(self):
        rapid_model_dir = os.path.join(BASE_DIR, "models", "for_rapidocr")
        rapid_params = {
            "Det.model_path": os.path.join(rapid_model_dir, "ch_PP-OCRv5_det_server_infer.onnx"),
            "Det.box_thresh": 0.3,
            "Det.unclip_ratio": 2.0,
            "Cls.cls_thresh": 0.5,
            "Rec.model_path": os.path.join(rapid_model_dir, "ch_PP-OCRv5_server_rec_infer.onnx"),
            "Global.text_score": 0.5
        }
        try:
            self.rapid = RapidOCR(params=rapid_params)
        except Exception as e:
            print(f"[WARN] RapidOCR init failed on GPU/default, retrying CPU… ({e})")
            import onnxruntime as ort
            # force CPU execution provider
            ort.get_available_providers = lambda: ["CPUExecutionProvider"]
            self.rapid = RapidOCR(params=rapid_params)

    def _load_paddleocr_onnx(self):
        # Alias for explicit "PaddleOCR ONNX" benchmark method.
        # This uses PP-OCRv5 ONNX models via ONNXRuntime (same instance as rapidocr).
        self.load("rapidocr")
        self.paddle_onnx = self.rapid

    def _load_yolo_ocr(self, size):
        # Ultralytics YOLO OCR: YOLO26-Text OBB detector + PP-OCRv6 medium recognizer (ONNX Runtime).
        # Weights live in models/for_yolo_ocr/ (auto-downloaded there on first run if missing).
        # The recognizer runs on CPU unless CUDA is available; the detector uses self.device.
        os.makedirs(YOLO_OCR_DIR, exist_ok=True)
        self.yolo_ocr[size] = ul_solutions.OCR(
            model=os.path.join(YOLO_OCR_DIR, f"yolo26{size}-text.pt"),
            recognizer=os.path.join(YOLO_OCR_DIR, "PP-OCRv6_medium_rec"),
            device=self.device,
            verbose=False,
        )

    def _load_doctr(self):
        self.doctr = doctr_text_join

    def _load_trocr(self):
        self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=False)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(self.device)

    def _load_glm_ocr(self):
        self.glm_model, self.glm_processor = models.ocr_glm.get_glm_model(device=self.device)
        if self.glm_model is None:
            raise RuntimeError("GLM-OCR model failed to load (see log above)")

    # ----- running -----

    def run(self, name, img, img_path, min_score=0.4, yolo_ocr_resize=0):
        """Run engine `name`. `img` is a BGR array, `img_path` the same image on disk."""
        self.load(name)
        if name.startswith("yolo-ocr-"):
            return self.run_yolo_ocr(img, name.rsplit("-", 1)[1], min_score, yolo_ocr_resize)
        dispatch = {
            "easyocr": lambda: self.run_easy(img_path, min_score),
            "tesseract": lambda: self.run_tesseract(img, min_score),
            "paddleocr": lambda: self.run_paddle(img_path, min_score),
            "paddleocr-onnx": lambda: self.run_paddle_onnx(img, min_score),
            "rapidocr": lambda: self.run_rapid(img, min_score),
            "doctr": lambda: self.run_doctr(img, min_score),
            "trocr": lambda: self.run_trocr(img_path),
            "glm-ocr": lambda: self.run_glm(img_path),
        }
        return dispatch[name]()

    def run_easy(self, img_path, min_score=0.4):
        ocr_result = self.easy.readtext(img_path)
        texts = []
        boxes = []
        for bbox, text, conf in ocr_result:
            clean = text.strip()
            if conf < min_score or not clean:
                continue
            texts.append(clean)
            pts = [(int(x), int(y)) for x, y in bbox]
            boxes.append({"points": pts, "text": clean, "score": float(conf)})
        return {"text": " ".join(texts), "boxes": boxes}

    def run_tesseract(self, img, min_score=0.4):
        custom_config = r'--oem 3 --psm 6'
        ocr_result = pytesseract.image_to_data(img, output_type=Output.DICT, config=custom_config, lang='eng')#lang='tha+eng')
        texts = []
        boxes = []
        for i in range(len(ocr_result['text'])):
            try:
                conf = float(ocr_result['conf'][i])
                if conf >= min_score:
                    text = ocr_result['text'][i].strip()
                    if text:
                        texts.append(text)
                        x = int(ocr_result["left"][i])
                        y = int(ocr_result["top"][i])
                        w = int(ocr_result["width"][i])
                        h = int(ocr_result["height"][i])
                        boxes.append(
                            {"rect": (x, y, w, h), "text": text, "score": conf}
                        )
            except ValueError:
                continue
        return {"text": " ".join(texts), "boxes": boxes}

    def run_paddle(self, img_path, min_score=0.4):
        result = self.paddle.predict(img_path)
        texts = []
        boxes = []
        for res in result:
            rec_texts = res.get('rec_texts', [])
            rec_scores = res.get('rec_scores', [])
            rec_polys = res.get('rec_polys') or res.get('dt_polys') or []
            for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                clean = text[0] if isinstance(text, (list, tuple)) else text
                clean = clean.strip()
                if score >= min_score and clean:
                    texts.append(clean)
                    pts_array = np.asarray(poly, dtype=np.int32)
                    if pts_array.size == 0:
                        continue
                    pts_array = pts_array.reshape(-1, 2)
                    pts = [(int(x), int(y)) for x, y in pts_array.tolist()]
                    boxes.append({"points": pts, "text": clean, "score": float(score)})
        return {"text": " ".join(texts), "boxes": boxes}

    @staticmethod
    def _rapid_payload(result, min_score):
        texts = []
        boxes = []
        if result and result.txts and result.scores and result.boxes is not None:
            for text, score, box in zip(result.txts, result.scores, result.boxes):
                clean = text.strip()
                if score < min_score or not clean:
                    continue
                texts.append(clean)
                pts = np.asarray(box, dtype=np.int32).reshape(-1, 2)
                boxes.append({"points": [(int(x), int(y)) for x, y in pts], "text": clean, "score": float(score)})
        return {"text": " ".join(texts), "boxes": boxes}

    def run_rapid(self, img, min_score=0.4):
        return self._rapid_payload(self.rapid(img), min_score)

    def run_paddle_onnx(self, img, min_score=0.4):
        return self._rapid_payload(self.paddle_onnx(img), min_score)

    def run_yolo_ocr(self, img, size="s", min_score=0.4, resize_long_side=0):
        # solutions.OCR never upscales, so tiny crops hit the detector at ~64-96 px.
        # resize_long_side > 0 rescales the image so its longest side equals that value.
        scale = 1.0
        if resize_long_side and resize_long_side > 0:
            scale = resize_long_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # result.texts: one dict per word, already sorted by line then left-to-right.
        # Pass a copy: solutions.OCR draws its annotations onto the input array in place,
        # which would corrupt the shared image for later engines (and for repeated calls).
        result = self.yolo_ocr[size](img.copy())
        texts = []
        boxes = []
        for word in result.texts:
            clean = word["text"].strip()
            score = float(word["confidence"])
            if score < min_score or not clean:
                continue
            texts.append(clean)
            # map polygon back to the un-resized image coordinates
            pts = (np.asarray(word["polygon"], dtype=np.float32).reshape(-1, 2) / scale).round().astype(np.int32)
            boxes.append({"points": [(int(x), int(y)) for x, y in pts], "text": clean, "score": score})
        return {"text": " ".join(texts), "boxes": boxes}

    def run_doctr(self, img, min_score=0.1):
        text = self.doctr(img, min_conf=min_score, device=self.device)
        # Join multi-line results into a single line with spaces
        text = text.replace("\n", " ").strip()
        return {"text": text, "boxes": []}

    def run_trocr(self, img_path):
        pil_img = Image.open(img_path).convert("RGB")
        pixel_values = self.trocr_processor(images=pil_img, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.trocr_model.generate(pixel_values, max_length=16, num_beams=4)
        decoded_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"text": decoded_text.strip(), "boxes": []}

    def run_glm(self, img_path):
        text = models.ocr_glm.run_glm_ocr(img_path, self.glm_model, self.glm_processor, self.device)
        # Clean up newlines for single-line output
        text = text.replace("\n", " ").strip()
        return {"text": text, "boxes": []}

# ---------- Benchmark ----------

def benchmark(fn):
    start = time.time()
    payload = fn()
    end = time.time()
    return payload, end - start


def apply_whitelist(text, whitelist):
    if not whitelist:
        return text
    return re.sub(f"[^{re.escape(whitelist)}]", "", text)


def score_text(gt, text):
    """fuzz.ratio accuracy (0-100) against the ground truth, or '-' when there is no ground truth."""
    if not gt:
        return "-"
    return float(fuzz.ratio(gt, text))


def run_benchmark(engines, img, img_path, names, min_score=0.4, gt=None, whitelist=None,
                  yolo_ocr_resize=0, warmup=False, skip=(), log=print):
    """Run each engine in `names` on one image and return a list of result dicts.

    img: BGR array; img_path: the same image on disk (some engines only take a path).
    warmup: run each engine once untimed first, so first-call overhead is not counted.
    """
    results = []
    for name in names:
        if name in skip:
            log(f"{name:<10} | SKIPPED")
            results.append({"name": name, "error": "SKIPPED", "time_ms": 0, "text": "", "acc": "-", "boxes": []})
            continue
        try:
            engines.load(name)  # never count model loading in the timing
            run = lambda: engines.run(name, img.copy(), img_path, min_score, yolo_ocr_resize)
            if warmup:
                run()
            payload, sec = benchmark(run)
            if isinstance(payload, dict):
                text = payload.get("text", "")
                boxes = payload.get("boxes", [])
            else:
                text = payload or ""
                boxes = []
            text = apply_whitelist(text, whitelist)
            acc = score_text(gt, text)
            log(f"{name:<10} | {sec*1000:.1f} ms | text: {text} | acc: {acc}")
            results.append({"name": name, "time_ms": sec * 1000, "text": text or "", "acc": acc, "boxes": boxes})
        except Exception as e:
            log(f"{name:<10} | ERROR: {e}")
            results.append({"name": name, "error": str(e), "time_ms": 0, "text": "", "acc": "-", "boxes": []})
    return results


def main(args):
    original_img = cv2.imread(args.input)
    if original_img is None:
        print(f"[ERROR] Cannot read image at {args.input}")
        return

    processed_img = original_img.copy()
    overlay_img = original_img.copy()
    warped_display = None
    points = []

    # Preprocess based on mode
    if args.mode == "pickpoints":
        points = pick_four_points(original_img.copy())
        if len(points) != 4:
            print("[ERROR] Need exactly 4 points.")
            return
        processed_img = warp_perspective(original_img, points)
        warp_path = os.path.splitext(args.input)[0] + "_warped.jpg"
        cv2.imwrite(warp_path, processed_img)
        print(f"[INFO] Warped saved to: {warp_path}")
        input_path = warp_path
        overlay_img = draw_points_overlay(original_img, points)
        warped_display = processed_img.copy()
    else:
        input_path = args.input
        warped_display = processed_img.copy()

    # Determine device
    device = None
    if args.cpu:
        device = "cpu"
    elif args.gpu:
        device = "cuda"
    elif args.mps:
        device = "mps"

    def parse_names(value):
        return [n.strip().lower() for n in (value.split(",") if value else []) if n.strip()]

    # "yolo-ocr" is shorthand for all YOLO sizes
    def expand(names):
        out = []
        for n in names:
            out.extend([f"yolo-ocr-{s}" for s in YOLO_OCR_SIZES] if n == "yolo-ocr" else [n])
        return out

    only = expand(parse_names(args.engines))
    skip_set = set(expand(parse_names(args.skip_engines)))
    unknown = sorted((set(only) | skip_set) - set(ENGINE_NAMES))
    if unknown:
        print(f"[ERROR] Unknown engine(s): {', '.join(unknown)}")
        return
    names = only or ENGINE_NAMES

    print("[INFO] Loading OCR engines...")
    engines = OCREngines(device_override=device, preload=[n for n in names if n not in skip_set])
    print("[INFO] Engines loaded ✅")

    print("\n====== OCR Benchmark ======")
    results = run_benchmark(
        engines, processed_img, input_path, names,
        min_score=args.min_score, gt=args.gt, whitelist=args.whitelist,
        yolo_ocr_resize=args.yolo_ocr_resize, warmup=args.warmup, skip=skip_set,
    )

    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    figure_name = f"ocr_results_{input_basename}.png"
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    figure_path = os.path.join(output_dir, figure_name)
    save_results_figure(
        original_img,
        overlay_img,
        warped_display,
        results,
        figure_path,
        args.mode,
    )
    print(f"[INFO] Saved summary figure to: {figure_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OCR engines")
    parser.add_argument("--mode", type=str, choices=["pickpoints", "full"], default="full",
                        help="pickpoints = click 4 points to warp, full = read entire image")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--min_score", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--gt", type=str, default=None, help="Ground truth string (optional)")
    engine_help = f"choices: {','.join(ENGINE_NAMES)}; 'yolo-ocr' = all YOLO sizes"
    parser.add_argument(
        "--engines",
        type=str,
        default="",
        help=f"Comma-separated engines to run (default: all). {engine_help}",
    )
    parser.add_argument(
        "--skip_engines",
        type=str,
        default="",
        help=f"Comma-separated engine names to skip ({engine_help})",
    )
    parser.add_argument(
        "--yolo_ocr_resize",
        type=int,
        default=0,
        help="Resize image so its longest side equals this before yolo-ocr (0 = off). ~160 works well for tiny label crops.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run each engine once untimed before timing it (removes first-call overhead).",
    )
    parser.add_argument(
        "--whitelist",
        type=str,
        default=None,
        help="Characters to allow in output (e.g., '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')."
    )

    # Device flags
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true", help="Force CPU usage")
    group.add_argument("--gpu", action="store_true", help="Force GPU (CUDA) usage")
    group.add_argument("--mps", action="store_true", help="Force MPS (Mac) usage")

    args = parser.parse_args()
    main(args)
