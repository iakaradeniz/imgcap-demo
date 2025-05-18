from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import sys, torch

# 1) BLIP-base model (Salesforce tarafından yayınlandı)
MODEL_ID = "Salesforce/blip-image-captioning-base"

# 2) Processor ve model yükle
processor = BlipProcessor.from_pretrained(MODEL_ID)
model     = BlipForConditionalGeneration.from_pretrained(MODEL_ID).eval()

@torch.no_grad()
def caption(img_path: str, beams: int = 5, max_len: int = 30) -> str:
    image  = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs, num_beams=beams, max_length=max_len)
    return processor.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Kullanım: python inference_base.py <gorsel_yolu>")
        sys.exit(1)

    print(caption(sys.argv[1]))
