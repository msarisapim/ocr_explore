# test_batch.py
import cv2
from models.ocr_doctr import doctr_text_lines_from_bgr, doctr_text_join

img1 = cv2.imread(r"data/for_OCR/90_10.jpg")
img2 = cv2.imread(r"data/for_OCR/80_20.jpg")

# 1) Single crop → one string
print(doctr_text_join(img1, min_conf=0.3))

# 2) Batch → list of lines per image
all_lines = doctr_text_lines_from_bgr([img1, img2], min_conf=0.3)
for i, lines in enumerate(all_lines, 1):
    print(f"\n--- Image {i} ---")
    for ln in lines:
        print(ln)
