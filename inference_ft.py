# inference.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import sys, torch

# ——— 1) Sabitler ——————————————————————————————
# Eğer yereldeki blip_ft/ klasöründeki model dosyalarını kullanacaksan:
FINE_TUNE_PATH = "iakaradeniz/blip-flickr8k-ft"                           

# Base processor hâlâ Hugging Face’den:
BASE_PROC      = "Salesforce/blip-image-captioning-base"  

# ——— 2) Cihaz ayarı ————————————————————————————
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ——— 3) Processor & Model ————————————————————————
processor = BlipProcessor.from_pretrained(BASE_PROC)
model     = BlipForConditionalGeneration.from_pretrained(FINE_TUNE_PATH)
model     = model.to(device).eval()

# ——— 4) Caption fonksiyonu ——————————————————————
@torch.no_grad()
def caption(img_path: str, beams: int = 5, max_len: int = 30) -> str:
    image  = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs, num_beams=beams, max_length=max_len)
    return processor.decode(output[0], skip_special_tokens=True)

# ——— 5) Ana blok —————————————————————————————
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    print("🖼  Görsel :", img_path)
    print("📝 Caption:", caption(img_path))
