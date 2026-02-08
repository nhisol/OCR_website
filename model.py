import torch 
from config import MODEL_PATH, PROCESSOR_PATH
from transformers import AutoModelForImageTextToText, AutoTokenizer
from PIL import Image
import os 
import logging

logger = logging.getLogger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
proccessor = AutoTokenizer.from_pretrained(PROCESSOR_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    # Fix the deprecation warning by using 'dtype' instead of 'torch_dtype'
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.to(DEVICE)
model.eval()
logger.info(f"Model loaded on {DEVICE}")


@torch.inference_mode()
def predict(input_image_path:str) -> str:
    
    image = Image.open(input_image_path).convert("RGB")
    inputs = proccessor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    generate_ids = model.generate(pixel_values=inputs, max_length=256, num_beams=4, do_sample=True)

    text = proccessor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip() 

    return text.strip()