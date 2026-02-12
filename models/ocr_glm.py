from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

def get_glm_model(model_id="zai-org/GLM-OCR", device="cpu"):
    """
    Load GLM-OCR model and processor.
    Note: zai-org/GLM-OCR might be large (0.9B+).
    """
    try:
        print(f"[INFO] Loading GLM-OCR model: {model_id} on {device}...")
        
        # Use float16 for GPU/MPS to save memory if possible
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        return model, processor
    except Exception as e:
        print(f"[ERROR] Failed to load GLM-OCR model: {e}")
        return None, None

def run_glm_ocr(img_path, model, processor, device):
    """Run GLM-OCR on an image using the official API."""
    if model is None or processor is None:
        return ""
    
    try:
        # Format the input as a chat message (official API format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img_path},
                    {"type": "text", "text": "Text Recognition:"}
                ],
            }
        ]
        
        # Apply chat template and prepare inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Remove token_type_ids if present (as shown in official example)
        inputs.pop("token_type_ids", None)
        
        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            # Decode only the generated part (skip input tokens)
            output_text = processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False
            )
        
        return output_text.strip()
        
    except Exception as e:
        print(f"[ERROR] GLM-OCR inference failed: {e}")
        import traceback
        traceback.print_exc()
        return ""
