print("🚀 Script started")
import torch
torch.cuda.is_available = lambda : False  # disable CUDA trick

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image

    print("✅ Imports OK")

    # img_path = r"data\for_OCR\7_29-8-2025_14H-11M_warped.jpg"
    # image = Image.open(img_path).convert("RGB")
    # print("✅ Image loaded:", image.size)

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    print("✅ Model loaded")

    # pixel_values = processor(images=image, return_tensors="pt").pixel_values
    # print("✅ Pixel values ready:", pixel_values.shape)

    # generated_ids = model.generate(pixel_values, max_length=64)
    # print("✅ Generated IDs:", generated_ids.tolist())

    # decoded_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # print("✅ Decoded Text:", decoded_text)

    from glob import glob
    for f in glob(r"data\for_OCR\*.jpg"):
        text = processor.batch_decode(
            model.generate(processor(images=Image.open(f).convert("RGB"), return_tensors="pt").pixel_values),
            skip_special_tokens=True
        )[0]
        print(f, "→", text)


except Exception as e:
    import traceback
    print("❌ Exception caught:", e)
    traceback.print_exc()
