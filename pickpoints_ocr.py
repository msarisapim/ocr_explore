# python .\pickpoints_ocr.py --min_score 0.4 --ocr_engine paddle --input data/for_OCR/7_29-8-2025_14H-11M.jpg
# python .\pickpoints_ocr.py --min_score 0.4 --ocr_engine rapidocr --input data/for_OCR/7_29-8-2025_14H-11M.jpg
# python .\pickpoints_ocr.py --min_score 0.4 --ocr_engine trocr --input data/for_OCR/7_29-8-2025_14H-11M.jpg

import cv2
import numpy as np
import math
import os
import argparse
import easyocr
import pytesseract
from pytesseract import Output
from paddleocr import PaddleOCR
from rapidocr import RapidOCR
from models.ocr_doctr import doctr_text_lines_from_bgr, doctr_text_join
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
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
            cv2.imshow("Click 4 Points (Top-Left to Bottom-Left Clockwise)", image)
            
    # Create the window and set the mouse callback
    cv2.namedWindow("Click 4 Points (Top-Left to Bottom-Left Clockwise)", cv2.WINDOW_NORMAL)
    cv2.imshow("Click 4 Points (Top-Left to Bottom-Left Clockwise)", image)
    cv2.setMouseCallback("Click 4 Points (Top-Left to Bottom-Left Clockwise)", click_event)
    
    # Run the event loop until 4 points are picked
    while len(points) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # Esc key to exit
            break
            
    cv2.destroyWindow("Click 4 Points (Top-Left to Bottom-Left Clockwise)")
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

# ---------- Main Pipeline ----------

def main(args):
    img = cv2.imread(args.input)
    if img is None:
        print(f"[ERROR] Cannot read image at {args.input}")
        return

    points = pick_four_points(img.copy())

    if len(points) != 4:
        print("[ERROR] You must click exactly 4 points or press Esc to exit.")
        return

    warped = warp_perspective(img, points)
    warp_path = os.path.splitext(args.input)[0] + "_warped.jpg"
    cv2.imwrite(warp_path, warped)
    print(f"[INFO] Warped image saved to: {warp_path}")

    print(f"[INFO] Running OCR with '{args.ocr_engine}'...")

    print("\n====== OCR Result ======")

    # Use a separate, dedicated window for the warped image
    cv2.namedWindow("Warped Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Warped Image", warped)
    
    # Wait for a key press to close this window, and then clean up
    print("Press any key to continue with OCR...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if args.ocr_engine == "easy":
        reader = easyocr.Reader(['th', 'en'])
        results = reader.readtext(warp_path)
        for i, (bbox, text, confidence) in enumerate(results):
            if confidence >= args.min_score:
                print(f"text: {text.strip()} (conf: {confidence:.2f})")

    elif args.ocr_engine == "tesseract":
        custom_config = r'--oem 3 --psm 6'
        ocr_result = pytesseract.image_to_data(warped, output_type=Output.DICT, config=custom_config, lang='tha+eng')
        n_boxes = len(ocr_result['text'])
        for i in range(n_boxes):
            try:
                conf = float(ocr_result['conf'][i])
                if conf >= args.min_score:
                    text = ocr_result['text'][i].strip()
                    if text:
                        print(f"text: {text} (conf: {conf:.2f})")
            except ValueError:
                continue
    elif args.ocr_engine == "paddle": #v5 https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det
        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            # det_model_dir=r"C:\Users\msari\.paddlex\official_models\PP-OCRv5_server_det",
            # rec_model_dir=r"C:\Users\msari\.paddlex\official_models\PP-OCRv5_server_rec",
            cls_model_dir=None,   # optional: angle classifier if you want
            # lang='th',            # since you’re using Thai rec model
            use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
            use_doc_unwarping=False, # Disables text image rectification model via this parameter
            use_textline_orientation=True, # Disables text line orientation classification model via this parameter
        )
        result = ocr.predict(warp_path)
        for res in result:  
            res.print()  
            res.save_to_img("output")  
            res.save_to_json("output")
            texts = res['rec_texts']
            scores = res['rec_scores']
            boxes  = res['rec_boxes']   # numpy array
            for text, score, box in zip(texts, scores, boxes):
                if score >= args.min_score:
                    print(f"{text.strip()} (conf: {score:.2f})")
    
    elif args.ocr_engine == "rapid":
        engine = RapidOCR(params={
            # --- Detector ---
            "Det.model_path": r"D:\WORK\SmartSense\Projects\Tools!\PaddlePaddle\models\for_rapidocr\ch_PP-OCRv5_det_server_infer.onnx",
            # "Det.model_path": r"D:\WORK\SmartSense\Projects\DSG-MAERSK\DSG\models\PP-OCRv5\det\ch_PP-OCRv5_det_infer.onnx",
            "Det.box_thresh": 0.3,
            "Det.unclip_ratio": 2.0,

            # --- Direction Classifier (optional) ---
            # "Cls.model_path": r"D:\WORK\SmartSense\Projects\DSG-MAERSK\DSG\models\PP-OCRv5\cls\ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "Cls.cls_thresh": 0.5,

            # --- Recognizer ---
            "Rec.model_path": r"D:\WORK\SmartSense\Projects\Tools!\PaddlePaddle\models\for_rapidocr\ch_PP-OCRv5_server_rec_infer.onnx",
            # "Rec.model_path": r"D:\WORK\SmartSense\Projects\DSG-MAERSK\DSG\models\PP-OCRv5\rec\ch_PP-OCRv5_rec_infer.onnx",
            "Rec.char_dict_path": r"D:\WORK\SmartSense\Projects\DSG-MAERSK\DSG\models\PP-OCRv5\rec\ppocr_keys_v1.txt",
            "Global.text_score": 0.5
        })
        image = cv2.imread(warp_path)
        results = engine(image)
        for text, score in zip(results.txts, results.scores):
            print(f"{text} (conf: {score:.2f})")

        for box, text in zip(results.boxes, results.txts):
            box = np.array(box).astype(int).reshape(-1, 2)
            cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.putText(image, f'{text} ({score:.2f})', tuple(box[0]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # print(text,box)

        cv2.imwrite(f'output/rapid.jpg', image)

    elif args.ocr_engine == "doctr":
        image = cv2.imread(warp_path)
        text = doctr_text_join(image, min_conf=0.1)
        print(text)

    elif args.ocr_engine == "trocr":
        # Load processor & model (can cache globally for speed, but this is simple)
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=False)
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

        # Convert warped image to PIL
        pil_img = Image.open(warp_path).convert("RGB")

        # Preprocess
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values

        # Generate text
        generated_ids = model.generate(pixel_values, max_length=16, num_beams=4)  # adjust for SKU labels
        decoded_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if decoded_text.strip():
            print(f"text: {decoded_text.strip()} (conf: N/A)")
        else:
            print("⚠️ TrOCR produced empty result")


    else:
        print(f"[ERROR] Unsupported OCR engine: {args.ocr_engine}")
        return

    print("==========================================")
    print("[INFO] OCR completed.")

# ---------- Entry ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align document using 4-point perspective transform and OCR it using PaddleOCR.")
    parser.add_argument("--min_score", type=float, default=0.0, help="Minimum confidence threshold for OCR results")
    parser.add_argument("--ocr_engine", type=str, default="tesseract", choices=["easy", "tesseract", "paddle", "rapid", "doctr", "trocr"],
                    help="OCR engine to use: easy | tesseract | paddle | rapid | doctr | trocr")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")

    args = parser.parse_args()
    main(args)