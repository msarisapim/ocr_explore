# python benchmark_ocr.py --mode pickpoints --input data/for_OCR/7_29-8-2025_14H-11M.jpg --gt "1200080"
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

class OCREngines:
    def __init__(self, device_override=None):
        print("[INFO] Loading all OCR engines...")
        
        # Determine device
        if device_override:
            self.device = device_override
        else:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
        
        print(f"[INFO] Using device: {self.device.upper()}")

        # EasyOCR
        self.easy = easyocr.Reader(['th', 'en'], gpu=(self.device != "cpu"))
        
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

        # Prepare RapidOCR params with relative paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rapid_model_dir = os.path.join(base_dir, "models", "for_rapidocr")
        
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

        self.doctr = doctr_text_join
        
        # TrOCR
        self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=False)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(self.device)
        
        # GLM-OCR
        self.glm_model, self.glm_processor = models.ocr_glm.get_glm_model(device=self.device)
        print("[INFO] Engines loaded ✅")

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

    def run_rapid(self, img, min_score=0.4):
        result = self.rapid(img)
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
    
    engines = OCREngines(device_override=device)
    img_for_ocr = processed_img.copy()

    candidates = {
        "easyocr": lambda: engines.run_easy(input_path, args.min_score),
        "tesseract": lambda: engines.run_tesseract(img_for_ocr, args.min_score),
        "paddleocr": lambda: engines.run_paddle(input_path, args.min_score),
        "rapidocr": lambda: engines.run_rapid(img_for_ocr, args.min_score),
        "doctr": lambda: engines.run_doctr(img_for_ocr, args.min_score),
        "trocr": lambda: engines.run_trocr(input_path),
        "glm-ocr": lambda: engines.run_glm(input_path),
    }

    skip_set = {
        name.strip().lower()
        for name in (args.skip_engines.split(",") if args.skip_engines else [])
        if name.strip()
    }
    unknown = sorted(skip_set - set(candidates.keys()))
    if unknown:
        print(f"[ERROR] Unknown engine(s) in --skip_engines: {', '.join(unknown)}")
        return

    print("\n====== OCR Benchmark ======")
    results = []
    for name, fn in candidates.items():
        if name in skip_set:
            print(f"{name:<10} | SKIPPED")
            results.append({"name": name, "error": "SKIPPED", "time_ms": 0, "text": "", "acc": "-", "boxes": []})
            continue
        try:
            payload, sec = benchmark(fn)
            if isinstance(payload, dict):
                text = payload.get("text", "")
                boxes = payload.get("boxes", [])
            else:
                text = payload or ""
                boxes = []

            # Whitelist filtering
            if args.whitelist:
                pattern = f"[^{re.escape(args.whitelist)}]"
                text = re.sub(pattern, "", text)

            acc = "-"
            if args.gt:
                acc = float(fuzz.ratio(args.gt, text))
            print(f"{name:<10} | {sec*1000:.1f} ms | text: {text} | acc: {acc}")
            results.append(
                {
                    "name": name,
                    "time_ms": sec * 1000,
                    "text": text or "",
                    "acc": acc,
                    "boxes": boxes,
                }
            )
        except Exception as e:
            print(f"{name:<10} | ERROR: {e}")
            results.append({"name": name, "error": str(e), "time_ms": 0, "text": "", "acc": "-", "boxes": []})

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
    parser.add_argument(
        "--skip_engines",
        type=str,
        default="",
        help="Comma-separated engine names to skip (choices: easyocr,tesseract,paddleocr,rapidocr,doctr,trocr)",
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
