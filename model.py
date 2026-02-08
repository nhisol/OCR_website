import os
import torch
from PIL import Image
import cv2
import numpy as np
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    AutoTokenizer,
    AutoImageProcessor
)

from config import MODEL_PATH, PROCESSOR_PATH, TORCH_DEVICE


# -----------------------
# Device
# -----------------------
DEVICE = torch.device(TORCH_DEVICE)
print("Using device:", DEVICE)


# -----------------------
# Load model
# -----------------------
print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()


# -----------------------
# Load processor parts
# -----------------------
print("Loading tokenizer from processor_ocr...")
tokenizer = AutoTokenizer.from_pretrained(PROCESSOR_PATH)

print("Loading image processor from processor_ocr...")
image_processor = AutoImageProcessor.from_pretrained(PROCESSOR_PATH)

print("Building TrOCR processor...")
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

print("Loaded successfully!")


# -----------------------
# Force special tokens (important)
# -----------------------
model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id


# -----------------------
# Optional image preprocessing (helps screenshots)
# -----------------------
def preprocess_image(path: str) -> Image.Image:
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    th = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(th)


# -----------------------
# OCR prediction
# -----------------------
@torch.no_grad()
def predict(image_path: str, use_preprocess: bool = True) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    if use_preprocess:
        image = preprocess_image(image_path)
    else:
        image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(DEVICE)

    generated_ids = model.generate(
        pixel_values,
        max_length=64,
        min_length=5,
        num_beams=8,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()
