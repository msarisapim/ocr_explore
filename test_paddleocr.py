# วิธีลง paddle ocr
# ```
# pip install paddleocr==3.2.0
# pip install paddlepaddle==3.1.1
# ```

from paddleocr import PaddleOCR
import cv2

# === Initialize OCR engine ===
ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            cls_model_dir=None,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True
        )

# === Run OCR on image ===
img_path = r"data\for_OCR\80_20.jpg"
ocr_result = ocr.predict(img_path)
min_score = 0.4
texts = []
for res in ocr_result:
    for text, score in zip(res['rec_texts'], res['rec_scores']):
        if score >= min_score:
            texts.append(text.strip())
            print(f"Text: {text} | Score: {score:.2f}")

